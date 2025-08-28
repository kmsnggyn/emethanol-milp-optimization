"""
Model Predictive Control (MPC) implementation for e-methanol plant optimization.

This module implements a rolling horizon optimization approach where the plant
re-optimizes every hour using only the next 24 hours of price data, simulating
realistic operational conditions with limited forecast horizons.
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from model import build_model


class PlantState:
    """Track the current state of the plant for MPC implementation."""
    
    def __init__(self):
        self.current_mode = 0  # 0 = 10% load, 1 = 100% load
        self.stabilization_remaining = 0  # Hours remaining in current state
        self.last_ramp_time = -1  # Time of last ramp event
    
    def update(self, new_mode, current_time, min_duration=3):
        """Update plant state based on new operational decision."""
        if new_mode != self.current_mode:
            # Ramp event
            self.current_mode = new_mode
            self.stabilization_remaining = min_duration
            self.last_ramp_time = current_time
        else:
            # Continue in same state
            if self.stabilization_remaining > 0:
                self.stabilization_remaining -= 1


def build_mpc_model(price_forecast, plant_state, params):
    """
    Build a 24-hour MPC optimization model starting from current plant state.
    
    Args:
        price_forecast (list): 24-hour price forecast [€/MWh]
        plant_state (PlantState): Current plant operational state
        params (dict): Model parameters
    
    Returns:
        pyo.ConcreteModel: 24-hour optimization model
    """
    
    # Initialize the model for 24 hours
    model = pyo.ConcreteModel()
    
    # =====================================================
    # SETS
    # =====================================================
    
    # Time horizon: hours 0 to 23 (24 hours total)
    model.T = pyo.RangeSet(0, 23)
    
    # =====================================================
    # PARAMETERS
    # =====================================================
    
    # Electricity prices for each hour
    model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(price_forecast)), 
                                doc="Electricity price [€/MWh]")
    
    # Copy all parameters from the main model
    for param_name, param_value in params.items():
        setattr(model, param_name, param_value)
    
    # =====================================================
    # VARIABLES
    # =====================================================
    
    # Binary variable: x[t] = 1 if plant operates at 100% load at time t, 0 if at 10%
    model.x = pyo.Var(model.T, domain=pyo.Binary, doc="Operating mode (0=10%, 1=100%)")
    
    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================
    
    def objective_rule(model):
        return (
            # Revenue from methanol sales
            sum(model.x[t] * model.M_100 * model.Price_Methanol + 
                (1 - model.x[t]) * model.M_10 * model.Price_Methanol for t in model.T)
            
            # Electricity costs
            - sum(model.x[t] * model.P_100 * model.price_elec[t] + 
                  (1 - model.x[t]) * model.P_10 * model.price_elec[t] for t in model.T)
            
            # CO2 costs
            - sum(model.x[t] * model.C_100 * model.Price_CO2 + 
                  (1 - model.x[t]) * model.C_10 * model.Price_CO2 for t in model.T)
            
            # Variable OPEX
            - sum(model.OPEX_Variable for t in model.T)
            
            # Ramping penalties (simplified for MPC)
            - sum(10000 * (model.x[t] - model.x[t-1])**2 for t in model.T if t > 0)
        )
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # =====================================================
    # CONSTRAINTS
    # =====================================================
    
    # Add constraint for current plant state
    if plant_state.stabilization_remaining > 0:
        # Plant must continue in current state for remaining stabilization time
        def initial_state_rule(model, t):
            if t < plant_state.stabilization_remaining:
                return model.x[t] == plant_state.current_mode
            else:
                return pyo.Constraint.Skip
        
        model.initial_state_constraint = pyo.Constraint(model.T, rule=initial_state_rule)
    else:
        # Plant can change state immediately, but fix initial state
        model.x[0].fix(plant_state.current_mode)
    
    return model


def solve_mpc_model(model):
    """Solve the MPC model and return the optimal first-hour decision."""
    
    # Try Gurobi first, then HiGHS
    solver_names = ['gurobi', 'highs']
    
    for name in solver_names:
        try:
            solver = pyo.SolverFactory(name)
            if solver.available():
                # Solve with minimal output for MPC
                results = solver.solve(model, tee=False)
                
                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    # Return optimal decision for first hour
                    return int(pyo.value(model.x[0]))
                else:
                    print(f"MPC solve failed with {name}: {results.solver.termination_condition}")
                    continue
        except:
            continue
    
    # If all solvers fail, return current state (safe fallback)
    print("Warning: MPC optimization failed, maintaining current state")
    return None


def run_mpc_optimization(prices, params, verbose=True):
    """
    Run complete MPC optimization over full year with 24-hour rolling horizon.
    
    Args:
        prices (list): Full year of hourly electricity prices
        params (dict): Model parameters
        verbose (bool): Print progress information
    
    Returns:
        dict: MPC results including decisions, states, and metrics
    """
    
    total_hours = len(prices)
    plant_state = PlantState()
    
    # Results storage
    decisions = []
    capacity_values = []
    ramp_events = []
    solve_times = []
    
    if verbose:
        print("Running MPC optimization with 24-hour rolling horizon...")
        print(f"Total optimization horizon: {total_hours} hours")
        print("Re-optimizing every hour with 24-hour forecast...")
    
    # Progress tracking
    progress_points = [int(total_hours * p) for p in [0.25, 0.5, 0.75, 1.0]]
    
    for hour in range(total_hours):
        
        # Progress reporting
        if verbose and hour in progress_points:
            progress = (hour / total_hours) * 100
            print(f"Progress: {progress:.0f}% ({hour}/{total_hours} hours)")
        
        # Get 24-hour price forecast (or remaining hours if near end)
        forecast_horizon = min(24, total_hours - hour)
        price_forecast = prices[hour:hour + forecast_horizon]
        
        # Pad with average prices if forecast is less than 24 hours
        if len(price_forecast) < 24:
            avg_price = np.mean(prices)
            price_forecast.extend([avg_price] * (24 - len(price_forecast)))
        
        # Build and solve MPC model
        import time
        start_time = time.time()
        
        mpc_model = build_mpc_model(price_forecast, plant_state, params)
        optimal_decision = solve_mpc_model(mpc_model)
        
        solve_time = time.time() - start_time
        solve_times.append(solve_time)
        
        # Handle solve failure
        if optimal_decision is None:
            optimal_decision = plant_state.current_mode
        
        # Record results
        decisions.append(optimal_decision)
        capacity_values.append(10 if optimal_decision == 0 else 100)
        
        # Detect ramp events
        if hour > 0 and decisions[hour] != decisions[hour-1]:
            ramp_events.append(1)
        else:
            ramp_events.append(0)
        
        # Update plant state
        plant_state.update(optimal_decision, hour)
    
    if verbose:
        print(f"MPC optimization completed!")
        print(f"Average solve time per hour: {np.mean(solve_times):.3f} seconds")
    
    # Calculate metrics
    hours_at_100 = sum(decisions)
    hours_at_10 = total_hours - hours_at_100
    capacity_factor = np.mean(capacity_values)
    total_ramps = sum(ramp_events)
    
    results = {
        'decisions': decisions,
        'capacity_values': capacity_values,
        'ramp_events': ramp_events,
        'solve_times': solve_times,
        'hours_at_100': hours_at_100,
        'hours_at_10': hours_at_10,
        'capacity_factor': capacity_factor,
        'total_ramps': total_ramps,
        'avg_solve_time': np.mean(solve_times)
    }
    
    return results


def run_steady_state_analysis(prices, params):
    """Run steady-state analysis (always 100% or always 10%)."""
    
    total_hours = len(prices)
    
    # Calculate economics for steady-state operations
    # Always 100% operation
    revenue_100 = total_hours * params["M_100"] * params["Price_Methanol"]
    electricity_cost_100 = sum(params["P_100"] * price for price in prices)
    co2_cost_100 = total_hours * params["C_100"] * params["Price_CO2"]
    variable_opex_100 = total_hours * params["OPEX_Variable"]
    profit_100 = (revenue_100 - electricity_cost_100 - co2_cost_100 - 
                  variable_opex_100 - params["Annualized_CAPEX"] - params["OPEX_Fixed"])
    
    # Always 10% operation
    revenue_10 = total_hours * params["M_10"] * params["Price_Methanol"]
    electricity_cost_10 = sum(params["P_10"] * price for price in prices)
    co2_cost_10 = total_hours * params["C_10"] * params["Price_CO2"]
    variable_opex_10 = total_hours * params["OPEX_Variable"]
    profit_10 = (revenue_10 - electricity_cost_10 - co2_cost_10 - 
                 variable_opex_10 - params["Annualized_CAPEX"] - params["OPEX_Fixed"])
    
    return {
        'always_100': {
            'capacity_factor': 100.0,
            'annual_profit': profit_100,
            'total_ramps': 0
        },
        'always_10': {
            'capacity_factor': 10.0,
            'annual_profit': profit_10,
            'total_ramps': 0
        }
    }


def compare_scenarios(prices, params):
    """Compare all three scenarios: steady-state, perfect foresight, and MPC."""
    
    print("="*60)
    print("SCENARIO COMPARISON ANALYSIS")
    print("="*60)
    
    # 1. Steady-state analysis
    print("\n1. Running steady-state analysis...")
    steady_state = run_steady_state_analysis(prices, params)
    
    # 2. Perfect foresight (use existing main.py approach)
    print("\n2. Running perfect foresight optimization...")
    from main import load_data, build_model as build_full_model, solve_model
    
    full_model = build_full_model(prices, params)
    status = solve_model(full_model)
    
    if str(status) == 'optimal':
        # Extract perfect foresight results
        pf_decisions = [pyo.value(full_model.x[t]) for t in full_model.T]
        pf_capacity = [10 if d == 0 else 100 for d in pf_decisions]
        pf_capacity_factor = np.mean(pf_capacity)
        pf_ramps = sum(1 for i in range(1, len(pf_decisions)) if pf_decisions[i] != pf_decisions[i-1])
        pf_profit = pyo.value(full_model.objective)
    else:
        print("Perfect foresight optimization failed!")
        return None
    
    # 3. MPC analysis
    print("\n3. Running MPC rolling horizon optimization...")
    mpc_results = run_mpc_optimization(prices, params)
    
    # Calculate MPC profit (simplified)
    mpc_profit = 0  # Would need full economic calculation
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    scenarios = {
        'Always 100%': {
            'capacity_factor': 100.0,
            'ramp_events': 0,
            'annual_profit': steady_state['always_100']['annual_profit']
        },
        'Always 10%': {
            'capacity_factor': 10.0,
            'ramp_events': 0,
            'annual_profit': steady_state['always_10']['annual_profit']
        },
        'Perfect Foresight': {
            'capacity_factor': pf_capacity_factor,
            'ramp_events': pf_ramps,
            'annual_profit': pf_profit
        },
        'MPC (24h horizon)': {
            'capacity_factor': mpc_results['capacity_factor'],
            'ramp_events': mpc_results['total_ramps'],
            'annual_profit': 'TBD'  # Would need economic calculation
        }
    }
    
    print(f"{'Scenario':<20} {'Capacity Factor':<15} {'Ramp Events':<12} {'Annual Profit':<15}")
    print("-" * 65)
    
    for name, results in scenarios.items():
        profit_str = f"€{results['annual_profit']:,.0f}" if isinstance(results['annual_profit'], (int, float)) else str(results['annual_profit'])
        print(f"{name:<20} {results['capacity_factor']:<14.1f}% {results['ramp_events']:<12} {profit_str:<15}")
    
    return {
        'steady_state': steady_state,
        'perfect_foresight': {
            'decisions': pf_decisions,
            'capacity_factor': pf_capacity_factor,
            'ramp_events': pf_ramps,
            'annual_profit': pf_profit
        },
        'mpc': mpc_results
    }


if __name__ == "__main__":
    # Load data and run comparison
    from main import load_data
    
    prices, params = load_data()
    if prices is not None:
        results = compare_scenarios(prices, params)
    else:
        print("Failed to load data!")
