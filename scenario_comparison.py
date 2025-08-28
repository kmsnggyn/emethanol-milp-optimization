"""
Simple MPC (Model Predictive Control) implementation for the e-methanol plant.

This script demonstrates the three scenarios for the master's thesis:
1. Steady-state operations (always 100% or always 10%)
2. Perfect foresight optimization (full year)
3. MPC rolling horizon (24-hour forecast window)
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time
from main import load_data, build_model as build_full_model, solve_model


class PlantState:
    """Track the current state of the plant for MPC implementation."""
    
    def __init__(self):
        self.current_mode = 0  # 0 = 10% load, 1 = 100% load
        self.stabilization_remaining = 0  # Hours remaining in current state
    
    def update(self, new_mode, min_duration=3):
        """Update plant state based on new operational decision."""
        if new_mode != self.current_mode:
            # Ramp event - need to stay in new state for min_duration
            self.current_mode = new_mode
            self.stabilization_remaining = min_duration
        else:
            # Continue in same state
            if self.stabilization_remaining > 0:
                self.stabilization_remaining -= 1


def build_mpc_model(price_forecast, plant_state, params):
    """Build a 24-hour MPC optimization model."""
    
    model = pyo.ConcreteModel()
    
    # Time horizon: 24 hours (0-23)
    model.T = pyo.RangeSet(0, 23)
    
    # Parameters
    model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(price_forecast)))
    
    # Add all other parameters
    for key, value in params.items():
        setattr(model, key, value)
    
    # Variables
    model.x = pyo.Var(model.T, domain=pyo.Binary)
    
    # Objective
    def obj_rule(model):
        revenue = sum((model.x[t] * model.M_100 + (1-model.x[t]) * model.M_10) * model.Price_Methanol 
                     for t in model.T)
        
        elec_cost = sum((model.x[t] * model.P_100 + (1-model.x[t]) * model.P_10) * model.price_elec[t] 
                       for t in model.T)
        
        co2_cost = sum((model.x[t] * model.C_100 + (1-model.x[t]) * model.C_10) * model.Price_CO2 
                      for t in model.T)
        
        var_opex = sum(model.OPEX_Variable for t in model.T)
        
        # Simple ramping penalty
        ramp_penalty = sum(10000 * (model.x[t] - model.x[t-1])**2 for t in model.T if t > 0)
        
        return revenue - elec_cost - co2_cost - var_opex - ramp_penalty
    
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    # Constraints for plant state
    if plant_state.stabilization_remaining > 0:
        # Must stay in current mode
        for t in range(min(plant_state.stabilization_remaining, 24)):
            model.x[t].fix(plant_state.current_mode)
    else:
        # Can switch immediately, but set initial state
        model.x[0].fix(plant_state.current_mode)
    
    return model


def run_mpc_simulation(prices, params, verbose=True):
    """Run complete MPC simulation with 24-hour rolling horizon."""
    
    total_hours = len(prices)
    plant_state = PlantState()
    
    results = {
        'decisions': [],
        'capacity_values': [],
        'ramp_events': [],
        'solve_times': []
    }
    
    if verbose:
        print(f"Running MPC simulation for {total_hours} hours...")
    
    for hour in range(total_hours):
        
        if verbose and hour % 1000 == 0:
            print(f"Progress: {hour}/{total_hours} hours ({100*hour/total_hours:.1f}%)")
        
        # Get 24-hour forecast
        forecast_end = min(hour + 24, total_hours)
        price_forecast = prices[hour:forecast_end]
        
        # Pad if necessary
        while len(price_forecast) < 24:
            price_forecast.append(np.mean(prices))
        
        # Build and solve MPC model
        start_time = time.time()
        
        try:
            mpc_model = build_mpc_model(price_forecast, plant_state, params)
            
            # Solve
            solver = pyo.SolverFactory('gurobi')
            if not solver.available():
                solver = pyo.SolverFactory('highs')
            
            result = solver.solve(mpc_model, tee=False)
            solve_time = time.time() - start_time
            
            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                decision = int(pyo.value(mpc_model.x[0]))
            else:
                # Fallback: maintain current state
                decision = plant_state.current_mode
                if verbose and hour < 10:
                    print(f"Hour {hour}: Solve failed, maintaining state {decision}")
        
        except Exception as e:
            # Fallback: maintain current state
            decision = plant_state.current_mode
            solve_time = 0
            if verbose and hour < 10:
                print(f"Hour {hour}: Exception during solve: {e}")
        
        # Record results
        results['decisions'].append(decision)
        results['capacity_values'].append(10 if decision == 0 else 100)
        results['solve_times'].append(solve_time)
        
        # Detect ramp
        if hour > 0 and decision != results['decisions'][hour-1]:
            results['ramp_events'].append(1)
        else:
            results['ramp_events'].append(0)
        
        # Update plant state
        plant_state.update(decision)
    
    # Calculate summary metrics
    results['total_hours'] = total_hours
    results['hours_at_100'] = sum(results['decisions'])
    results['hours_at_10'] = total_hours - results['hours_at_100']
    results['capacity_factor'] = np.mean(results['capacity_values'])
    results['total_ramps'] = sum(results['ramp_events'])
    results['avg_solve_time'] = np.mean(results['solve_times'])
    
    if verbose:
        print(f"\nMPC Simulation Complete!")
        print(f"Capacity factor: {results['capacity_factor']:.1f}%")
        print(f"Total ramp events: {results['total_ramps']}")
        print(f"Average solve time: {results['avg_solve_time']:.3f} seconds")
    
    return results


def analyze_steady_state(prices, params):
    """Analyze steady-state operations."""
    
    total_hours = len(prices)
    
    # Always 100%
    revenue_100 = total_hours * params["M_100"] * params["Price_Methanol"]
    elec_cost_100 = sum(params["P_100"] * price for price in prices)
    co2_cost_100 = total_hours * params["C_100"] * params["Price_CO2"]
    var_opex_100 = total_hours * params["OPEX_Variable"]
    profit_100 = revenue_100 - elec_cost_100 - co2_cost_100 - var_opex_100
    
    # Always 10%
    revenue_10 = total_hours * params["M_10"] * params["Price_Methanol"]
    elec_cost_10 = sum(params["P_10"] * price for price in prices)
    co2_cost_10 = total_hours * params["C_10"] * params["Price_CO2"]
    var_opex_10 = total_hours * params["OPEX_Variable"]
    profit_10 = revenue_10 - elec_cost_10 - co2_cost_10 - var_opex_10
    
    return {
        'always_100': {
            'capacity_factor': 100.0,
            'annual_profit': profit_100,
            'ramp_events': 0
        },
        'always_10': {
            'capacity_factor': 10.0,
            'annual_profit': profit_10,
            'ramp_events': 0
        }
    }


def run_perfect_foresight(prices, params):
    """Run perfect foresight optimization."""
    
    print("Running perfect foresight optimization...")
    model = build_full_model(prices, params)
    status = solve_model(model)
    
    if str(status) == 'optimal':
        decisions = []
        for t in range(len(prices)):
            decisions.append(int(pyo.value(model.x[t])))
        
        capacity_values = [10 if d == 0 else 100 for d in decisions]
        capacity_factor = np.mean(capacity_values)
        
        ramp_events = 0
        for i in range(1, len(decisions)):
            if decisions[i] != decisions[i-1]:
                ramp_events += 1
        
        annual_profit = pyo.value(model.objective)
        
        return {
            'decisions': decisions,
            'capacity_factor': capacity_factor,
            'ramp_events': ramp_events,
            'annual_profit': annual_profit
        }
    else:
        print(f"Perfect foresight optimization failed: {status}")
        return None


def compare_all_scenarios():
    """Compare all three scenarios and print results."""
    
    # Load data
    prices, params = load_data()
    if prices is None:
        print("Failed to load data!")
        return
    
    print("="*70)
    print("E-METHANOL PLANT OPTIMIZATION: SCENARIO COMPARISON")
    print("="*70)
    
    # 1. Steady-state analysis
    print("\n1. STEADY-STATE ANALYSIS")
    print("-" * 30)
    steady_results = analyze_steady_state(prices, params)
    
    print(f"Always 100%: CF={steady_results['always_100']['capacity_factor']:.1f}%, " + 
          f"Profit=€{steady_results['always_100']['annual_profit']:,.0f}")
    print(f"Always 10%:  CF={steady_results['always_10']['capacity_factor']:.1f}%, " + 
          f"Profit=€{steady_results['always_10']['annual_profit']:,.0f}")
    
    # 2. Perfect foresight
    print("\n2. PERFECT FORESIGHT OPTIMIZATION")
    print("-" * 35)
    pf_results = run_perfect_foresight(prices, params)
    
    if pf_results:
        print(f"Optimal: CF={pf_results['capacity_factor']:.1f}%, " + 
              f"Ramps={pf_results['ramp_events']}, " + 
              f"Profit=€{pf_results['annual_profit']:,.0f}")
    
    # 3. MPC rolling horizon
    print("\n3. MPC ROLLING HORIZON (24-HOUR FORECAST)")
    print("-" * 42)
    mpc_results = run_mpc_simulation(prices, params)
    
    print(f"MPC: CF={mpc_results['capacity_factor']:.1f}%, " + 
          f"Ramps={mpc_results['total_ramps']}, " + 
          f"Avg solve time={mpc_results['avg_solve_time']:.3f}s")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Scenario':<25} {'Capacity Factor':<15} {'Ramp Events':<12} {'Annual Profit':<15}")
    print("-" * 70)
    
    scenarios = [
        ("Always 100%", steady_results['always_100']['capacity_factor'], 0, steady_results['always_100']['annual_profit']),
        ("Always 10%", steady_results['always_10']['capacity_factor'], 0, steady_results['always_10']['annual_profit']),
    ]
    
    if pf_results:
        scenarios.append(("Perfect Foresight", pf_results['capacity_factor'], pf_results['ramp_events'], pf_results['annual_profit']))
    
    scenarios.append(("MPC (24h horizon)", mpc_results['capacity_factor'], mpc_results['total_ramps'], "TBD"))
    
    for name, cf, ramps, profit in scenarios:
        profit_str = f"€{profit:,.0f}" if isinstance(profit, (int, float)) else str(profit)
        print(f"{name:<25} {cf:<14.1f}% {ramps:<12} {profit_str:<15}")
    
    # Create comparison plot
    create_comparison_plot(steady_results, pf_results, mpc_results)
    
    return {
        'steady_state': steady_results,
        'perfect_foresight': pf_results,
        'mpc': mpc_results
    }


def create_comparison_plot(steady_results, pf_results, mpc_results):
    """Create a comparison plot of the three scenarios."""
    
    plt.figure(figsize=(12, 8))
    
    # Only plot if we have time series data
    if pf_results and 'decisions' in pf_results:
        hours = range(len(pf_results['decisions']))
        
        plt.subplot(2, 1, 1)
        pf_capacity = [10 if d == 0 else 100 for d in pf_results['decisions']]
        plt.plot(hours, pf_capacity, 'b-', alpha=0.7, label='Perfect Foresight')
        plt.ylabel('Capacity (%)')
        plt.title('Plant Operation Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        mpc_capacity = mpc_results['capacity_values']
        plt.plot(hours, mpc_capacity, 'r-', alpha=0.7, label='MPC (24h horizon)')
        plt.xlabel('Hour of Year')
        plt.ylabel('Capacity (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/scenario_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to plots/scenario_comparison.png")
    
    # Summary bar chart
    plt.figure(figsize=(10, 6))
    
    scenarios = ['Always 100%', 'Always 10%']
    capacity_factors = [100, 10]
    ramp_events = [0, 0]
    
    if pf_results:
        scenarios.append('Perfect Foresight')
        capacity_factors.append(pf_results['capacity_factor'])
        ramp_events.append(pf_results['ramp_events'])
    
    scenarios.append('MPC (24h)')
    capacity_factors.append(mpc_results['capacity_factor'])
    ramp_events.append(mpc_results['total_ramps'])
    
    x = range(len(scenarios))
    
    plt.subplot(1, 2, 1)
    plt.bar(x, capacity_factors, color=['blue', 'red', 'green', 'orange'][:len(scenarios)])
    plt.xlabel('Scenario')
    plt.ylabel('Capacity Factor (%)')
    plt.title('Capacity Factor Comparison')
    plt.xticks(x, scenarios, rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(x, ramp_events, color=['blue', 'red', 'green', 'orange'][:len(scenarios)])
    plt.xlabel('Scenario')
    plt.ylabel('Ramp Events')
    plt.title('Operational Flexibility')
    plt.xticks(x, scenarios, rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/scenario_summary.png', dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to plots/scenario_summary.png")


if __name__ == "__main__":
    results = compare_all_scenarios()
