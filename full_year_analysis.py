#!/usr/bin/env python3
"""
Full Year E-Methanol Plant Analysis
==================================

Comprehensive analysis of the full year (2023) with detailed insights
and visualizations for all three operational strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import get_parameters, load_data, build_model, solve_model
import time
from datetime import datetime, timedelta

def create_full_year_plots(prices, results_100, results_10, results_dynamic, params):
    """Create comprehensive full-year analysis plots."""
    
    # Create date range for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(len(prices))]
    
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Full year electricity prices with operation zones
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(dates, prices, 'b-', alpha=0.7, linewidth=0.8, label='Electricity Price')
    ax1.axhline(y=9.42, color='green', linestyle='--', alpha=0.7, label='100% Breakeven (€9.42/MWh)')
    ax1.axhline(y=14.07, color='orange', linestyle='--', alpha=0.7, label='10% Breakeven (€14.07/MWh)')
    ax1.axhline(y=np.mean(prices), color='red', linestyle='-', alpha=0.7, label=f'Average (€{np.mean(prices):.1f}/MWh)')
    
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title('Full Year 2023 Electricity Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Monthly price distribution
    ax2 = plt.subplot(3, 3, 2)
    monthly_prices = []
    monthly_labels = []
    for month in range(1, 13):
        month_hours = [i for i, d in enumerate(dates) if d.month == month]
        if month_hours:
            monthly_prices.append([prices[i] for i in month_hours])
            monthly_labels.append(f'{month:02d}')
    
    ax2.boxplot(monthly_prices, labels=monthly_labels)
    ax2.axhline(y=9.42, color='green', linestyle='--', alpha=0.7, label='100% Breakeven')
    ax2.axhline(y=14.07, color='orange', linestyle='--', alpha=0.7, label='10% Breakeven')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Price (EUR/MWh)')
    ax2.set_title('Monthly Price Distribution 2023')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=9.42, color='green', linestyle='--', alpha=0.7, label='100% Breakeven')
    ax3.axvline(x=14.07, color='orange', linestyle='--', alpha=0.7, label='10% Breakeven')
    ax3.axvline(x=np.mean(prices), color='red', linestyle='-', alpha=0.7, label=f'Average')
    ax3.set_xlabel('Price (EUR/MWh)')
    ax3.set_ylabel('Frequency (hours)')
    ax3.set_title('Price Distribution Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Economic comparison
    ax4 = plt.subplot(3, 3, 4)
    strategies = ['100% All Year', '10% All Year', 'Dynamic Opt']
    profits = [results_100['total_profit']/1e6, results_10['total_profit']/1e6, results_dynamic['total_profit']/1e6]
    revenues = [results_100['total_revenue']/1e6, results_10['total_revenue']/1e6, results_dynamic['total_revenue']/1e6]
    costs = [results_100['total_costs']/1e6, results_10['total_costs']/1e6, results_dynamic['total_costs']/1e6]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    bars1 = ax4.bar(x - width, profits, width, label='Profit', color='green', alpha=0.7)
    bars2 = ax4.bar(x, revenues, width, label='Revenue', color='blue', alpha=0.7)
    bars3 = ax4.bar(x + width, costs, width, label='Total Costs', color='red', alpha=0.7)
    
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Value (Million EUR)')
    ax4.set_title('Full Year Economic Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 0.5,
                     f'€{height:.1f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Plot 5: Cost breakdown
    ax5 = plt.subplot(3, 3, 5)
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
    
    ax5.bar(x - width, costs_100, width, label='100% All Year', color='red', alpha=0.7)
    ax5.bar(x, costs_10, width, label='10% All Year', color='orange', alpha=0.7)
    ax5.bar(x + width, costs_dynamic, width, label='Dynamic Opt', color='green', alpha=0.7)
    
    ax5.set_xlabel('Cost Category')
    ax5.set_ylabel('Cost (Million EUR)')
    ax5.set_title('Full Year Cost Breakdown')
    ax5.set_xticks(x)
    ax5.set_xticklabels(cost_categories, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Production comparison
    ax6 = plt.subplot(3, 3, 6)
    productions = [results_100['total_production']/1000, results_10['total_production']/1000, 
                   results_dynamic['total_production']/1000]
    capacity_factors = [results_100['capacity_factor'], results_10['capacity_factor'], 
                        results_dynamic['capacity_factor']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, productions, width, label='Production (kton)', color='blue', alpha=0.7)
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x + width/2, capacity_factors, width, label='Capacity Factor (%)', color='green', alpha=0.7)
    
    ax6.set_xlabel('Strategy')
    ax6.set_ylabel('Production (kton)', color='blue')
    ax6_twin.set_ylabel('Capacity Factor (%)', color='green')
    ax6.set_title('Full Year Production Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(strategies, rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Dynamic operation profile (sample months)
    ax7 = plt.subplot(3, 3, 7)
    if 'x_100_values' in results_dynamic:
        # Show first 3 months
        sample_hours = min(2160, len(prices))  # 3 months
        sample_dates = dates[:sample_hours]
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
        
        ax7.plot(sample_dates, power_profile, 'r-', linewidth=1, label='Power Consumption')
        ax7.fill_between(sample_dates, power_profile, alpha=0.3, color='red')
        
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Power (MW)')
        ax7.set_title('Dynamic Operation Profile (First 3 Months)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Profit improvement over time
    ax8 = plt.subplot(3, 3, 8)
    baseline_profit = results_100['total_profit']
    improvements = [
        0,  # 100% all year baseline
        (results_10['total_profit'] - baseline_profit) / 1e6,  # 10% all year
        (results_dynamic['total_profit'] - baseline_profit) / 1e6  # Dynamic
    ]
    
    colors = ['red', 'orange', 'green']
    bars = ax8.bar(strategies, improvements, color=colors, alpha=0.7)
    ax8.set_ylabel('Profit Improvement (Million EUR)')
    ax8.set_title('Profit vs 100% All Year Baseline')
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.1,
                 f'€{improvement:.1f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Plot 9: Electricity cost savings analysis
    ax9 = plt.subplot(3, 3, 9)
    elec_costs = [results_100['electricity_costs']/1e6, results_10['electricity_costs']/1e6, 
                  results_dynamic['electricity_costs']/1e6]
    savings_vs_100 = [0, results_100['electricity_costs'] - results_10['electricity_costs'], 
                      results_100['electricity_costs'] - results_dynamic['electricity_costs']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax9.bar(x - width/2, elec_costs, width, label='Electricity Cost (M€)', color='red', alpha=0.7)
    ax9_twin = ax9.twinx()
    bars2 = ax9_twin.bar(x + width/2, [s/1e6 for s in savings_vs_100], width, label='Savings vs 100% (M€)', color='green', alpha=0.7)
    
    ax9.set_xlabel('Strategy')
    ax9.set_ylabel('Electricity Cost (Million EUR)', color='red')
    ax9_twin.set_ylabel('Savings vs 100% All Year (Million EUR)', color='green')
    ax9.set_title('Electricity Cost Analysis')
    ax9.set_xticks(x)
    ax9.set_xticklabels(strategies, rotation=45)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_year_analysis.png', dpi=300, bbox_inches='tight')
    print("Full year analysis saved as 'full_year_analysis.png'")
    
    return fig

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
    
    print("Running full year dynamic optimization...")
    start_time = time.time()
    
    model = build_model(prices, params)
    termination_condition = solve_model(model)
    
    solve_time = time.time() - start_time
    print(f"Full year dynamic optimization completed in {solve_time:.1f} seconds")
    
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

def main():
    """Main full year analysis function."""
    
    print("FULL YEAR E-METHANOL PLANT ANALYSIS")
    print("=" * 60)
    print("Comprehensive analysis of 2023 operational strategies")
    print()
    
    # Load data
    print("Loading full year 2023 data...")
    params = get_parameters()
    data = load_data()
    prices = data['price']
    
    print(f"Loaded {len(prices)} hours of electricity price data")
    print(f"Period: {len(prices)/24:.1f} days ({len(prices)/24/30:.1f} months)")
    print(f"Average electricity price: €{np.mean(prices):.2f}/MWh")
    print(f"Price range: €{min(prices):.2f} - €{max(prices):.2f}/MWh")
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
    
    # Display comprehensive results
    print("\nFULL YEAR 2023 RESULTS")
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
    
    # Price analysis
    print("PRICE ANALYSIS:")
    print(f"Hours below 100% breakeven (€9.42/MWh): {sum(1 for p in prices if p < 9.42)} ({sum(1 for p in prices if p < 9.42)/len(prices)*100:.1f}%)")
    print(f"Hours below 10% breakeven (€14.07/MWh): {sum(1 for p in prices if p < 14.07)} ({sum(1 for p in prices if p < 14.07)/len(prices)*100:.1f}%)")
    print(f"Hours above 10% breakeven: {sum(1 for p in prices if p >= 14.07)} ({sum(1 for p in prices if p >= 14.07)/len(prices)*100:.1f}%)")
    print()
    
    # Dynamic operation analysis
    if 'avg_electricity_price_100' in results_dynamic:
        print("DYNAMIC OPERATION ANALYSIS:")
        print(f"100% Load Hours: {results_dynamic['hours_100']:.0f} ({results_dynamic['hours_100']/len(prices)*100:.1f}%)")
        print(f"10% Load Hours:  {results_dynamic['hours_10']:.0f} ({results_dynamic['hours_10']/len(prices)*100:.1f}%)")
        print(f"100% Load Avg Price: €{results_dynamic['avg_electricity_price_100']:.2f}/MWh")
        print(f"10% Load Avg Price:  €{results_dynamic['avg_electricity_price_10']:.2f}/MWh")
        print()
    
    # Create comprehensive plots
    print("Creating full year analysis plots...")
    create_full_year_plots(prices, results_100, results_10, results_dynamic, params)
    
    print("\nFULL YEAR ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key insights:")
    print(f"• Dynamic optimization saves €{abs(improvement_dynamic):.1f}M vs 100% all year")
    print(f"• Dynamic optimization saves €{abs(improvement_dynamic - improvement_10):.1f}M vs 10% all year")
    print(f"• Capacity factor: {results_dynamic['capacity_factor']:.1f}% (vs 100% for 100% all year)")
    print(f"• Electricity cost savings: €{(results_100['electricity_costs'] - results_dynamic['electricity_costs'])/1e6:.1f}M")
    print(f"• Plant operates at 100% load {results_dynamic['hours_100']/len(prices)*100:.1f}% of the time")
    print(f"• Plant operates at 10% load {results_dynamic['hours_10']/len(prices)*100:.1f}% of the time")

if __name__ == "__main__":
    main()
