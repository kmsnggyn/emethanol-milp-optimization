#!/usr/bin/env python3
"""
E-Methanol Plant Optimization - Perfect Forecast Analysis

This script runs perfect forecast optimization for the e-methanol plant
with complete knowledge of electricity prices.

Usage:
    python run_optimization.py

Requirements:
    - pyomo, gurobipy, pandas, numpy, matplotlib
    - Excel files in electricity_data/ directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import get_parameters, load_data, build_model, solve_model
import time

def run_perfect_optimization():
    """Run optimization with perfect knowledge of all electricity prices."""
    
    print("E-METHANOL PLANT OPTIMIZATION")
    print("=" * 50)
    print("Perfect Forecast Analysis - Complete Price Knowledge")
    print()
    
    # Load parameters and data
    print("Loading parameters and electricity price data...")
    params = get_parameters()
    data = load_data()
    prices = data['price']
    
    print(f"Loaded {len(prices)} hours of electricity price data")
    print(f"Price range: {min(prices):.2f} - {max(prices):.2f} EUR/MWh")
    print(f"Average price: {np.mean(prices):.2f} EUR/MWh")
    print()
    
    # Build and solve the model
    print("Building optimization model...")
    start_time = time.time()
    
    model = build_model(prices, params)
    
    print("Solving optimization problem...")
    print("(This may take a few minutes for the full year)")
    
    termination_condition = solve_model(model)
    
    solve_time = time.time() - start_time
    print(f"Optimization completed in {solve_time:.1f} seconds")
    print(f"Solver status: {termination_condition}")
    print()
    
    if termination_condition != "optimal":
        print("ERROR: Optimization did not find optimal solution!")
        return None
    
    # Extract results
    print("Extracting optimization results...")
    
    # Get variable values
    x_100_values = [model.x_100[t].value for t in model.T]
    x_10_values = [model.x_10[t].value for t in model.T]
    y_ramp_up_values = [model.y_ramp_up[t].value for t in model.T]
    y_ramp_down_values = [model.y_ramp_down[t].value for t in model.T]
    
    # Calculate statistics
    hours_100 = sum(x_100_values)
    hours_10 = sum(x_10_values)
    hours_ramp_up = sum(y_ramp_up_values)
    hours_ramp_down = sum(y_ramp_down_values)
    total_hours = len(prices)
    
    # Calculate production and costs
    total_production_100 = hours_100 * params["M_100"]  # kg
    total_production_10 = hours_10 * params["M_10"]     # kg
    total_production = total_production_100 + total_production_10
    
    # Calculate electricity costs
    electricity_cost_100 = sum(
        prices[t] * params["P_100"] * x_100_values[t] 
        for t in range(len(prices))
    )
    electricity_cost_10 = sum(
        prices[t] * params["P_10"] * x_10_values[t] 
        for t in range(len(prices))
    )
    electricity_cost_ramp_up = sum(
        prices[t] * params["P_100"] * (params["Energy_Penalty_Up"]/100) * y_ramp_up_values[t]
        for t in range(len(prices))
    )
    electricity_cost_ramp_down = sum(
        prices[t] * params["P_10"] * (params["Energy_Penalty_Down"]/100) * y_ramp_down_values[t]
        for t in range(len(prices))
    )
    total_electricity_cost = electricity_cost_100 + electricity_cost_10 + electricity_cost_ramp_up + electricity_cost_ramp_down
    
    # Calculate other costs
    co2_cost = hours_100 * params["C_100"] * params["Price_CO2"] + hours_10 * params["C_10"] * params["Price_CO2"]
    variable_opex = hours_100 * params["OPEX_Variable"] + hours_10 * params["OPEX_10"]
    
    # Calculate revenue
    revenue = total_production * params["Price_Methanol"]
    
    # Calculate total costs
    total_costs = (total_electricity_cost + co2_cost + variable_opex + 
                   params["Annualized_CAPEX"] + params["OPEX_Fixed"] + params["OPEX_Electrolysis_Stack"])
    
    # Calculate profit
    total_profit = revenue - total_costs
    
    # Display results
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Total Profit: €{total_profit:,.0f}")
    print(f"Total Revenue: €{revenue:,.0f}")
    print(f"Total Costs: €{total_costs:,.0f}")
    print()
    
    print("OPERATION BREAKDOWN")
    print("-" * 30)
    print(f"100% Load Hours: {hours_100:.0f} ({hours_100/total_hours*100:.1f}%)")
    print(f"10% Load Hours:  {hours_10:.0f} ({hours_10/total_hours*100:.1f}%)")
    print(f"Ramp Up Events:  {hours_ramp_up:.0f}")
    print(f"Ramp Down Events: {hours_ramp_down:.0f}")
    print()
    
    print("PRODUCTION")
    print("-" * 30)
    print(f"100% Load Production: {total_production_100/1000:,.0f} tons")
    print(f"10% Load Production:  {total_production_10/1000:,.0f} tons")
    print(f"Total Production:     {total_production/1000:,.0f} tons")
    print(f"Capacity Factor:      {total_production/(total_hours * params['M_100'])*100:.1f}%")
    print()
    
    print("COST BREAKDOWN")
    print("-" * 30)
    print(f"Electricity Cost:     €{total_electricity_cost:,.0f}")
    print(f"  - 100% Load:        €{electricity_cost_100:,.0f}")
    print(f"  - 10% Load:         €{electricity_cost_10:,.0f}")
    print(f"  - Ramp Penalties:   €{electricity_cost_ramp_up + electricity_cost_ramp_down:,.0f}")
    print(f"CO2 Cost:             €{co2_cost:,.0f}")
    print(f"Variable OPEX:        €{variable_opex:,.0f}")
    print(f"Fixed OPEX:           €{params['OPEX_Fixed'] + params['OPEX_Electrolysis_Stack']:,.0f}")
    print(f"CAPEX:                €{params['Annualized_CAPEX']:,.0f}")
    print()
    
    # Calculate breakeven prices
    breakeven_100 = (params["C_100"] * params["Price_CO2"] + params["OPEX_Variable"]) / params["P_100"]
    breakeven_10 = (params["C_10"] * params["Price_CO2"] + params["OPEX_10"]) / params["P_10"]
    
    print("ECONOMIC ANALYSIS")
    print("-" * 30)
    print(f"100% Load Breakeven:  €{breakeven_100:.2f}/MWh")
    print(f"10% Load Breakeven:   €{breakeven_10:.2f}/MWh")
    print()
    
    # Analyze price distribution
    prices_100_hours = [prices[t] for t in range(len(prices)) if x_100_values[t] > 0.5]
    prices_10_hours = [prices[t] for t in range(len(prices)) if x_10_values[t] > 0.5]
    
    if prices_100_hours:
        print(f"100% Load Price Stats:")
        print(f"  Average: €{np.mean(prices_100_hours):.2f}/MWh")
        print(f"  Range: €{min(prices_100_hours):.2f} - €{max(prices_100_hours):.2f}/MWh")
    
    if prices_10_hours:
        print(f"10% Load Price Stats:")
        print(f"  Average: €{np.mean(prices_10_hours):.2f}/MWh")
        print(f"  Range: €{min(prices_10_hours):.2f} - €{max(prices_10_hours):.2f}/MWh")
    
    print()
    
    # Create visualization
    create_optimization_plot(prices, x_100_values, x_10_values, params)
    
    return {
        'total_profit': total_profit,
        'total_production': total_production,
        'hours_100': hours_100,
        'hours_10': hours_10,
        'capacity_factor': total_production/(total_hours * params['M_100'])*100,
        'electricity_cost': total_electricity_cost,
        'revenue': revenue
    }

def create_optimization_plot(prices, x_100_values, x_10_values, params):
    """Create visualization of optimization results."""
    
    # Create a sample of the first 168 hours (1 week) for visualization
    sample_hours = min(168, len(prices))
    sample_prices = prices[:sample_hours]
    sample_x_100 = x_100_values[:sample_hours]
    sample_x_10 = x_10_values[:sample_hours]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Electricity prices and operating states
    hours = range(sample_hours)
    
    # Plot electricity prices
    ax1.plot(hours, sample_prices, 'b-', alpha=0.7, label='Electricity Price')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title('E-Methanol Plant Optimization - Sample Week')
    ax1.grid(True, alpha=0.3)
    
    # Highlight operating states
    for i, (price, x100, x10) in enumerate(zip(sample_prices, sample_x_100, sample_x_10)):
        if x100 > 0.5:  # 100% load
            ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='green', label='100% Load' if i == 0 else "")
        elif x10 > 0.5:  # 10% load
            ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='orange', label='10% Load' if i == 0 else "")
    
    ax1.legend()
    
    # Plot 2: Power consumption
    power_consumption = []
    for i in range(sample_hours):
        if sample_x_100[i] > 0.5:
            power_consumption.append(params["P_100"])
        elif sample_x_10[i] > 0.5:
            power_consumption.append(params["P_10"])
        else:
            power_consumption.append(0)
    
    ax2.plot(hours, power_consumption, 'r-', linewidth=2, label='Power Consumption')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Power (MW)')
    ax2.set_title('Power Consumption Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    print("Optimization visualization saved as 'optimization_results.png'")
    
    return fig

def show_key_results():
    """Display key economic results and insights."""
    print("\nKEY ECONOMIC INSIGHTS")
    print("=" * 50)
    
    params = get_parameters()
    
    print("Plant Configuration:")
    print(f"  • 100% Load: {params['M_100']/1000:.1f} ton/hr, {params['P_100']} MW")
    print(f"  • 10% Load:  {params['M_10']/1000:.1f} ton/hr, {params['P_10']} MW")
    print(f"  • Methanol Price: €{params['Price_Methanol']*1000:.0f}/ton")
    print(f"  • Annual Fixed Costs: €{(params['OPEX_Fixed'] + params['OPEX_Electrolysis_Stack'] + params['Annualized_CAPEX'])/1e6:.1f}M")
    print()
    
    # Calculate breakeven prices
    breakeven_100 = (params["C_100"] * params["Price_CO2"] + params["OPEX_Variable"]) / params["P_100"]
    breakeven_10 = (params["C_10"] * params["Price_CO2"] + params["OPEX_10"]) / params["P_10"]
    
    print("Economic Analysis:")
    print(f"  100% Load Mode:")
    print(f"    • Revenue: €{params['M_100'] * params['Price_Methanol']:.0f}/hour")
    print(f"    • Variable Costs: €{(params['C_100'] * params['Price_CO2'] + params['OPEX_Variable']):.0f}/hour")
    print(f"    • Breakeven Electricity: €{breakeven_100:.2f}/MWh")
    print(f"  10% Load Mode:")
    print(f"    • Revenue: €{params['M_10'] * params['Price_Methanol']:.0f}/hour")
    print(f"    • Variable Costs: €{(params['C_10'] * params['Price_CO2'] + params['OPEX_10']):.0f}/hour")
    print(f"    • Breakeven Electricity: €{breakeven_10:.2f}/MWh")
    print()
    
    print("Key Insights:")
    print("  • Plant operates at 100% when electricity < €{:.2f}/MWh".format(breakeven_100))
    print("  • Plant operates at 10% when electricity < €{:.2f}/MWh".format(breakeven_10))
    print("  • Optimization enables selective operation during profitable periods")
    print("  • Binary operation provides operational flexibility")
    print("  • Perfect forecasting enables optimal load selection")

def main():
    """Main execution function."""
    print("E-METHANOL PLANT OPTIMIZATION")
    print("=" * 50)
    print("Perfect Forecast Analysis")
    print("Complete knowledge of electricity prices")
    print()
    
    # Run optimization
    results = run_perfect_optimization()
    
    if results:
        # Show key results
        show_key_results()
        
        print("\nOPTIMIZATION COMPLETE")
        print("=" * 50)
        print("This represents the theoretical maximum profit achievable")
        print("with perfect knowledge of all electricity prices.")
    else:
        print("Optimization failed!")

if __name__ == "__main__":
    main()