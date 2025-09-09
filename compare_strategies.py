#!/usr/bin/env python3
"""
E-Methanol Plant Strategy Comparison
===================================

Compare three strategies:
1. 100% Load All Year (baseline)
2. 10% Load All Year (minimum)
3. Dynamic Optimization (current)

This shows the value of flexible operation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import get_parameters, load_data, build_model, solve_model
import time

def calculate_fixed_strategy_costs(prices, params, load_level):
    """Calculate costs for a fixed load strategy."""
    
    if load_level == 100:
        power = params["P_100"]
        production = params["M_100"]
        co2_cost_per_hour = params["C_100"] * params["Price_CO2"]
        opex_per_hour = params["OPEX_Variable"]
    elif load_level == 10:
        power = params["P_10"]
        production = params["M_10"]
        co2_cost_per_hour = params["C_10"] * params["Price_CO2"]
        opex_per_hour = params["OPEX_10"]
    else:
        raise ValueError("Load level must be 100 or 10")
    
    # Calculate costs
    total_hours = len(prices)
    total_production = production * total_hours
    total_revenue = total_production * params["Price_Methanol"]
    
    # Electricity costs
    electricity_costs = sum(prices[t] * power for t in range(total_hours))
    
    # Other costs
    co2_costs = co2_cost_per_hour * total_hours
    variable_opex = opex_per_hour * total_hours
    fixed_opex = params["OPEX_Fixed"] + params["OPEX_Electrolysis_Stack"]
    capex = params["Annualized_CAPEX"]
    
    total_costs = electricity_costs + co2_costs + variable_opex + fixed_opex + capex
    total_profit = total_revenue - total_costs
    
    return {
        'total_profit': total_profit,
        'total_revenue': total_revenue,
        'total_costs': total_costs,
        'electricity_costs': electricity_costs,
        'co2_costs': co2_costs,
        'variable_opex': variable_opex,
        'total_production': total_production,
        'capacity_factor': total_production / (total_hours * params["M_100"]) * 100,
        'avg_electricity_price': np.mean(prices),
        'power_consumption': power
    }

def run_dynamic_optimization(prices, params):
    """Run the dynamic optimization strategy."""
    
    print("Running dynamic optimization...")
    start_time = time.time()
    
    model = build_model(prices, params)
    termination_condition = solve_model(model)
    
    solve_time = time.time() - start_time
    print(f"Dynamic optimization completed in {solve_time:.1f} seconds")
    
    if termination_condition != "optimal":
        print("ERROR: Dynamic optimization failed!")
        return None
    
    # Extract results
    x_100_values = [model.x_100[t].value for t in model.T]
    x_10_values = [model.x_10[t].value for t in model.T]
    
    # Calculate statistics
    hours_100 = sum(x_100_values)
    hours_10 = sum(x_10_values)
    total_hours = len(prices)
    
    # Calculate production and costs
    total_production_100 = hours_100 * params["M_100"]
    total_production_10 = hours_10 * params["M_10"]
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
    total_electricity_cost = electricity_cost_100 + electricity_cost_10
    
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
    
    return {
        'total_profit': total_profit,
        'total_revenue': revenue,
        'total_costs': total_costs,
        'electricity_costs': total_electricity_cost,
        'co2_costs': co2_cost,
        'variable_opex': variable_opex,
        'total_production': total_production,
        'capacity_factor': total_production / (total_hours * params["M_100"]) * 100,
        'hours_100': hours_100,
        'hours_10': hours_10,
        'avg_electricity_price_100': np.mean([prices[t] for t in range(len(prices)) if x_100_values[t] > 0.5]) if hours_100 > 0 else 0,
        'avg_electricity_price_10': np.mean([prices[t] for t in range(len(prices)) if x_10_values[t] > 0.5]) if hours_10 > 0 else 0,
        'x_100_values': x_100_values,
        'x_10_values': x_10_values
    }

def create_comparison_plots(prices, results_100, results_10, results_dynamic, params):
    """Create comprehensive comparison plots."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Economic comparison
    ax1 = plt.subplot(2, 3, 1)
    strategies = ['100% All Year', '10% All Year', 'Dynamic Opt']
    profits = [results_100['total_profit']/1e6, results_10['total_profit']/1e6, results_dynamic['total_profit']/1e6]
    revenues = [results_100['total_revenue']/1e6, results_10['total_revenue']/1e6, results_dynamic['total_revenue']/1e6]
    costs = [results_100['total_costs']/1e6, results_10['total_costs']/1e6, results_dynamic['total_costs']/1e6]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    bars1 = ax1.bar(x - width, profits, width, label='Profit', color='green', alpha=0.7)
    bars2 = ax1.bar(x, revenues, width, label='Revenue', color='blue', alpha=0.7)
    bars3 = ax1.bar(x + width, costs, width, label='Total Costs', color='red', alpha=0.7)
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Value (Million EUR)')
    ax1.set_title('Economic Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'€{height:.1f}M', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Cost breakdown
    ax2 = plt.subplot(2, 3, 2)
    cost_categories = ['Electricity', 'CO2', 'Variable OPEX', 'Fixed OPEX', 'CAPEX']
    
    costs_100 = [results_100['electricity_costs']/1e6, results_100['co2_costs']/1e6, 
                 results_100['variable_opex']/1e6, (params['OPEX_Fixed'] + params['OPEX_Electrolysis_Stack'])/1e6, 
                 params['Annualized_CAPEX']/1e6]
    costs_10 = [results_10['electricity_costs']/1e6, results_10['co2_costs']/1e6, 
                results_10['variable_opex']/1e6, (params['OPEX_Fixed'] + params['OPEX_Electrolysis_Stack'])/1e6, 
                params['Annualized_CAPEX']/1e6]
    costs_dynamic = [results_dynamic['electricity_costs']/1e6, results_dynamic['co2_costs']/1e6, 
                     results_dynamic['variable_opex']/1e6, (params['OPEX_Fixed'] + params['OPEX_Electrolysis_Stack'])/1e6, 
                     params['Annualized_CAPEX']/1e6]
    
    x = np.arange(len(cost_categories))
    width = 0.25
    
    ax2.bar(x - width, costs_100, width, label='100% All Year', color='red', alpha=0.7)
    ax2.bar(x, costs_10, width, label='10% All Year', color='orange', alpha=0.7)
    ax2.bar(x + width, costs_dynamic, width, label='Dynamic Opt', color='green', alpha=0.7)
    
    ax2.set_xlabel('Cost Category')
    ax2.set_ylabel('Cost (Million EUR)')
    ax2.set_title('Cost Breakdown Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cost_categories, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Production comparison
    ax3 = plt.subplot(2, 3, 3)
    productions = [results_100['total_production']/1000, results_10['total_production']/1000, 
                   results_dynamic['total_production']/1000]
    capacity_factors = [results_100['capacity_factor'], results_10['capacity_factor'], 
                        results_dynamic['capacity_factor']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, productions, width, label='Production (kton)', color='blue', alpha=0.7)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, capacity_factors, width, label='Capacity Factor (%)', color='green', alpha=0.7)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Production (kton)', color='blue')
    ax3_twin.set_ylabel('Capacity Factor (%)', color='green')
    ax3.set_title('Production Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{height:.0f}k', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3_twin.text(bar.get_x() + bar.get_width()/2., height + 2,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Electricity price analysis
    ax4 = plt.subplot(2, 3, 4)
    sample_hours = min(168, len(prices))  # First week
    hours = range(sample_hours)
    
    ax4.plot(hours, prices[:sample_hours], 'b-', alpha=0.7, label='Electricity Price')
    ax4.axhline(y=params['P_100']*9.42, color='green', linestyle='--', alpha=0.7, label='100% Breakeven')
    ax4.axhline(y=params['P_10']*14.07, color='orange', linestyle='--', alpha=0.7, label='10% Breakeven')
    
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Price (EUR/MWh)')
    ax4.set_title('Electricity Price vs Breakeven (Sample Week)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Dynamic operation profile
    ax5 = plt.subplot(2, 3, 5)
    if 'x_100_values' in results_dynamic:
        sample_x_100 = results_dynamic['x_100_values'][:sample_hours]
        sample_x_10 = results_dynamic['x_10_values'][:sample_hours]
        
        power_profile = []
        for i in range(sample_hours):
            if sample_x_100[i] > 0.5:
                power_profile.append(params["P_100"])
            elif sample_x_10[i] > 0.5:
                power_profile.append(params["P_10"])
            else:
                power_profile.append(0)
        
        ax5.plot(hours, power_profile, 'r-', linewidth=2, label='Power Consumption')
        ax5.fill_between(hours, power_profile, alpha=0.3, color='red')
        
        ax5.set_xlabel('Hour')
        ax5.set_ylabel('Power (MW)')
        ax5.set_title('Dynamic Operation Profile (Sample Week)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Profit improvement
    ax6 = plt.subplot(2, 3, 6)
    baseline_profit = results_100['total_profit']
    improvements = [
        0,  # 100% all year baseline
        (results_10['total_profit'] - baseline_profit) / 1e6,  # 10% all year
        (results_dynamic['total_profit'] - baseline_profit) / 1e6  # Dynamic
    ]
    
    colors = ['red', 'orange', 'green']
    bars = ax6.bar(strategies, improvements, color=colors, alpha=0.7)
    ax6.set_ylabel('Profit Improvement (Million EUR)')
    ax6.set_title('Profit vs 100% All Year Baseline')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.1,
                 f'€{improvement:.1f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("Strategy comparison saved as 'strategy_comparison.png'")
    
    return fig

def main():
    """Main comparison function."""
    
    print("E-METHANOL PLANT STRATEGY COMPARISON")
    print("=" * 60)
    print("Comparing three operational strategies:")
    print("1. 100% Load All Year (baseline)")
    print("2. 10% Load All Year (minimum)")
    print("3. Dynamic Optimization (current)")
    print()
    
    # Load data
    print("Loading data...")
    params = get_parameters()
    data = load_data()
    prices = data['price']
    
    print(f"Loaded {len(prices)} hours of electricity price data")
    print(f"Average electricity price: €{np.mean(prices):.2f}/MWh")
    print()
    
    # Calculate fixed strategies
    print("Calculating 100% All Year strategy...")
    results_100 = calculate_fixed_strategy_costs(prices, params, 100)
    
    print("Calculating 10% All Year strategy...")
    results_10 = calculate_fixed_strategy_costs(prices, params, 10)
    
    # Run dynamic optimization
    results_dynamic = run_dynamic_optimization(prices, params)
    
    if results_dynamic is None:
        print("Dynamic optimization failed!")
        return
    
    # Display results
    print("\nSTRATEGY COMPARISON RESULTS")
    print("=" * 60)
    
    strategies = ['100% All Year', '10% All Year', 'Dynamic Opt']
    results = [results_100, results_10, results_dynamic]
    
    print(f"{'Strategy':<15} {'Profit (M€)':<12} {'Revenue (M€)':<12} {'Costs (M€)':<12} {'Production (kton)':<15} {'Cap. Factor (%)':<12}")
    print("-" * 90)
    
    for strategy, result in zip(strategies, results):
        print(f"{strategy:<15} {result['total_profit']/1e6:<12.1f} {result['total_revenue']/1e6:<12.1f} "
              f"{result['total_costs']/1e6:<12.1f} {result['total_production']/1000:<15.1f} {result['capacity_factor']:<12.1f}")
    
    print()
    
    # Calculate improvements
    baseline_profit = results_100['total_profit']
    improvement_10 = (results_10['total_profit'] - baseline_profit) / 1e6
    improvement_dynamic = (results_dynamic['total_profit'] - baseline_profit) / 1e6
    
    print("PROFIT IMPROVEMENTS vs 100% All Year:")
    print(f"10% All Year:     €{improvement_10:+.1f}M ({improvement_10/baseline_profit*1e6:+.1f}%)")
    print(f"Dynamic Opt:      €{improvement_dynamic:+.1f}M ({improvement_dynamic/baseline_profit*1e6:+.1f}%)")
    print()
    
    # Electricity cost analysis
    print("ELECTRICITY COST ANALYSIS:")
    print(f"100% All Year:    €{results_100['electricity_costs']/1e6:.1f}M")
    print(f"10% All Year:     €{results_10['electricity_costs']/1e6:.1f}M")
    print(f"Dynamic Opt:      €{results_dynamic['electricity_costs']/1e6:.1f}M")
    print()
    
    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(prices, results_100, results_10, results_dynamic, params)
    
    print("\nCOMPARISON COMPLETE!")
    print("=" * 60)
    print("Key insights:")
    print(f"• Dynamic optimization saves €{abs(improvement_dynamic):.1f}M vs 100% all year")
    print(f"• Dynamic optimization saves €{abs(improvement_dynamic - improvement_10):.1f}M vs 10% all year")
    print(f"• Capacity factor: {results_dynamic['capacity_factor']:.1f}% (vs 100% for 100% all year)")
    print(f"• Electricity cost savings: €{(results_100['electricity_costs'] - results_dynamic['electricity_costs'])/1e6:.1f}M")

if __name__ == "__main__":
    main()
