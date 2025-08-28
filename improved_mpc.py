"""
Improved MPC implementation with better decision logic.
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time
from main import load_data


def simple_mpc_strategy(prices, params, forecast_horizon=24):
    """
    Implement a simple but effective MPC strategy.
    
    At each hour, look ahead for forecast_horizon hours and decide:
    - If average price in forecast window < break-even price, run at 100%
    - Otherwise, run at 10%
    - Include ramping constraints and stabilization periods
    """
    
    # Calculate break-even price
    delta_revenue = (params["M_100"] - params["M_10"]) * params["Price_Methanol"]
    delta_power = params["P_100"] - params["P_10"]
    delta_co2 = (params["C_100"] - params["C_10"]) * params["Price_CO2"]
    breakeven_price = (delta_revenue - delta_co2) / delta_power
    
    print(f"Using break-even price: €{breakeven_price:.2f}/MWh")
    print(f"Forecast horizon: {forecast_horizon} hours")
    
    total_hours = len(prices)
    decisions = []
    current_state = 0  # Start at 10%
    stabilization_remaining = 0
    
    for hour in range(total_hours):
        
        # Get forecast window
        forecast_end = min(hour + forecast_horizon, total_hours)
        forecast_prices = prices[hour:forecast_end]
        
        # Pad if necessary
        while len(forecast_prices) < forecast_horizon:
            forecast_prices.append(np.mean(prices))
        
        # Calculate average price in forecast window
        avg_forecast_price = np.mean(forecast_prices)
        
        # Check if we're in stabilization period
        if stabilization_remaining > 0:
            # Must continue in current state
            decision = current_state
            stabilization_remaining -= 1
        else:
            # Can make a decision
            if avg_forecast_price < breakeven_price:
                optimal_state = 1  # 100% capacity
            else:
                optimal_state = 0  # 10% capacity
            
            # Check if this requires a ramp
            if optimal_state != current_state:
                # Ramp event - must stabilize for 3 hours
                decision = optimal_state
                current_state = optimal_state
                stabilization_remaining = 2  # 3 total hours including this one
            else:
                # Continue in same state
                decision = current_state
        
        decisions.append(decision)
    
    # Calculate results
    capacity_values = [10 if d == 0 else 100 for d in decisions]
    capacity_factor = np.mean(capacity_values)
    
    ramp_events = 0
    for i in range(1, len(decisions)):
        if decisions[i] != decisions[i-1]:
            ramp_events += 1
    
    # Calculate economic performance
    annual_profit = 0
    for hour, decision in enumerate(decisions):
        if decision == 1:  # 100% operation
            profit_hour = (params["M_100"] * params["Price_Methanol"] - 
                          params["P_100"] * prices[hour] - 
                          params["C_100"] * params["Price_CO2"] - 
                          params["OPEX_Variable"])
        else:  # 10% operation
            profit_hour = (params["M_10"] * params["Price_Methanol"] - 
                          params["P_10"] * prices[hour] - 
                          params["C_10"] * params["Price_CO2"] - 
                          params["OPEX_Variable"])
        
        annual_profit += profit_hour
    
    # Subtract ramping penalties (simplified)
    ramping_penalty = ramp_events * 10000  # €10k per ramp
    annual_profit -= ramping_penalty
    
    # Subtract fixed costs
    annual_profit -= params["Annualized_CAPEX"] + params["OPEX_Fixed"]
    
    return {
        'decisions': decisions,
        'capacity_values': capacity_values,
        'capacity_factor': capacity_factor,
        'ramp_events': ramp_events,
        'annual_profit': annual_profit,
        'breakeven_price': breakeven_price
    }


def compare_mpc_horizons():
    """Compare MPC performance with different forecast horizons."""
    
    prices, params = load_data()
    if prices is None:
        return
    
    print("="*60)
    print("MPC FORECAST HORIZON COMPARISON")
    print("="*60)
    
    horizons = [6, 12, 24, 48, 72, 168]  # 6h to 1 week
    results = {}
    
    for horizon in horizons:
        print(f"\nRunning MPC with {horizon}-hour forecast horizon...")
        result = simple_mpc_strategy(prices, params, horizon)
        results[horizon] = result
        
        print(f"Results for {horizon}h horizon:")
        print(f"  Capacity factor: {result['capacity_factor']:.1f}%")
        print(f"  Ramp events: {result['ramp_events']}")
        print(f"  Annual profit: €{result['annual_profit']:,.0f}")
    
    # Perfect foresight comparison
    print(f"\nRunning perfect foresight for comparison...")
    perfect_result = simple_mpc_strategy(prices, params, len(prices))
    results['perfect'] = perfect_result
    
    print(f"Perfect foresight results:")
    print(f"  Capacity factor: {perfect_result['capacity_factor']:.1f}%")
    print(f"  Ramp events: {perfect_result['ramp_events']}")
    print(f"  Annual profit: €{perfect_result['annual_profit']:,.0f}")
    
    # Create comparison plots
    create_horizon_comparison_plots(results, prices)
    
    return results


def create_horizon_comparison_plots(results, prices):
    """Create plots comparing different forecast horizons."""
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Capacity factor vs horizon
    plt.subplot(3, 2, 1)
    horizons = [h for h in results.keys() if h != 'perfect']
    cfs = [results[h]['capacity_factor'] for h in horizons]
    
    plt.plot(horizons, cfs, 'o-', color='blue', linewidth=2, markersize=8)
    plt.axhline(y=results['perfect']['capacity_factor'], color='red', linestyle='--', 
                label=f"Perfect foresight: {results['perfect']['capacity_factor']:.1f}%")
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Capacity Factor (%)')
    plt.title('Capacity Factor vs Forecast Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Ramp events vs horizon
    plt.subplot(3, 2, 2)
    ramps = [results[h]['ramp_events'] for h in horizons]
    
    plt.plot(horizons, ramps, 'o-', color='green', linewidth=2, markersize=8)
    plt.axhline(y=results['perfect']['ramp_events'], color='red', linestyle='--',
                label=f"Perfect foresight: {results['perfect']['ramp_events']}")
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Ramp Events')
    plt.title('Operational Flexibility vs Forecast Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Annual profit vs horizon
    plt.subplot(3, 2, 3)
    profits = [results[h]['annual_profit'] for h in horizons]
    
    plt.plot(horizons, profits, 'o-', color='purple', linewidth=2, markersize=8)
    plt.axhline(y=results['perfect']['annual_profit'], color='red', linestyle='--',
                label=f"Perfect foresight: €{results['perfect']['annual_profit']:,.0f}")
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Annual Profit (€)')
    plt.title('Economic Performance vs Forecast Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Example time series comparison (first 168 hours)
    plt.subplot(3, 2, 4)
    hours = range(168)  # First week
    
    # Show price and decisions for 24h vs perfect foresight
    if 24 in results:
        decisions_24h = [10 if d == 0 else 100 for d in results[24]['decisions'][:168]]
        plt.plot(hours, decisions_24h, 'b-', alpha=0.7, label='24h horizon', linewidth=2)
    
    decisions_perfect = [10 if d == 0 else 100 for d in results['perfect']['decisions'][:168]]
    plt.plot(hours, decisions_perfect, 'r--', alpha=0.7, label='Perfect foresight', linewidth=2)
    
    plt.xlabel('Hour')
    plt.ylabel('Capacity (%)')
    plt.title('First Week: MPC vs Perfect Foresight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Profit gap vs horizon
    plt.subplot(3, 2, 5)
    perfect_profit = results['perfect']['annual_profit']
    profit_gaps = [(perfect_profit - results[h]['annual_profit']) / 1000 for h in horizons]  # in k€
    
    plt.plot(horizons, profit_gaps, 'o-', color='orange', linewidth=2, markersize=8)
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Profit Gap from Perfect (k€)')
    plt.title('Value of Perfect Information')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Summary table
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    for h in sorted([h for h in results.keys() if isinstance(h, int)]):
        table_data.append([
            f"{h}h",
            f"{results[h]['capacity_factor']:.1f}%",
            f"{results[h]['ramp_events']}",
            f"€{results[h]['annual_profit']:,.0f}"
        ])
    
    table_data.append([
        "Perfect",
        f"{results['perfect']['capacity_factor']:.1f}%",
        f"{results['perfect']['ramp_events']}",
        f"€{results['perfect']['annual_profit']:,.0f}"
    ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Horizon', 'Capacity Factor', 'Ramps', 'Annual Profit'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('Summary Results', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/mpc_horizon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nMPC horizon comparison saved to plots/mpc_horizon_comparison.png")


if __name__ == "__main__":
    results = compare_mpc_horizons()
