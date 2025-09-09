#!/usr/bin/env python3
"""
Custom E-Methanol Plant Analysis with Modified Visualizations
============================================================

Creates custom plot layouts and cost per ton analysis as requested.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import get_parameters, load_data, build_model, solve_model
import time
from datetime import datetime, timedelta

# KTH color scheme - defined once for consistency across all plots
PLOT_COLORS = [
    (0/255, 71/255, 145/255),   # KTH blue
    (161/255, 89/255, 0/255),   # Orange-brown
    (71/255, 156/255, 158/255), # Teal
    (117/255, 0/255, 27/255),   # Deep red
    (25/255, 75/255, 32/255),   # Dark green
    (50/255, 50/255, 50/255),   # Dark gray
]

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
        'fixed_opex': fixed_opex,
        'capex': capex,
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
    fixed_opex = params["OPEX_Fixed"] + params["OPEX_Electrolysis_Stack"]
    capex = params["Annualized_CAPEX"]
    
    # Calculate revenue
    revenue = total_production * params["Price_Methanol"]
    
    # Calculate total costs
    total_costs = total_electricity_cost + co2_cost + variable_opex + fixed_opex + capex
    
    # Calculate profit
    total_profit = revenue - total_costs
    
    return {
        'total_profit': total_profit,
        'total_revenue': revenue,
        'total_costs': total_costs,
        'electricity_costs': total_electricity_cost,
        'co2_costs': co2_cost,
        'variable_opex': variable_opex,
        'fixed_opex': fixed_opex,
        'capex': capex,
        'total_production': total_production,
        'capacity_factor': total_production / (total_hours * params["M_100"]) * 100,
        'hours_100': hours_100,
        'hours_10': hours_10,
        'avg_electricity_price_100': np.mean([prices[t] for t in range(len(prices)) if x_100_values[t] > 0.5]) if hours_100 > 0 else 0,
        'avg_electricity_price_10': np.mean([prices[t] for t in range(len(prices)) if x_10_values[t] > 0.5]) if hours_10 > 0 else 0,
        'x_100_values': x_100_values,
        'x_10_values': x_10_values
    }

def create_plot_1(prices, results_100, results_10, results_dynamic, params):
    """Create plot 1: Full year electricity prices with operation zones."""
    
    # Create date range for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(len(prices))]
    
    # A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(dates, prices, color=PLOT_COLORS[0], linewidth=0.8, label='Electricity Price')
    ax.axhline(y=9.42, color=PLOT_COLORS[4], linestyle='--', label='100% Breakeven (€9.42/MWh)')
    ax.axhline(y=14.07, color=PLOT_COLORS[1], linestyle='--', label='10% Breakeven (€14.07/MWh)')
    ax.axhline(y=np.mean(prices), color=PLOT_COLORS[3], linestyle='-', label=f'Average (€{np.mean(prices):.1f}/MWh)')
    
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plot1 Full year 2023 electricity prices with breakeven thresholds.png', dpi=300, bbox_inches='tight')
    print("Plot 1 saved as 'plot1 Full year 2023 electricity prices with breakeven thresholds.png'")
    
    return fig

def create_plot_2(prices, results_100, results_10, results_dynamic, params):
    """Create plot 2: Monthly price distribution."""
    
    # Create date range for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(len(prices))]
    
    # A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    monthly_prices = []
    monthly_labels = []
    for month in range(1, 13):
        month_hours = [i for i, d in enumerate(dates) if d.month == month]
        if month_hours:
            monthly_prices.append([prices[i] for i in month_hours])
            monthly_labels.append(f'{month:02d}')
    
    # Create boxplot with KTH blue median lines
    bp = ax.boxplot(monthly_prices, tick_labels=monthly_labels, patch_artist=True)
    
    # Style the boxplot elements
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor(PLOT_COLORS[0])
    
    for median in bp['medians']:
        median.set_color(PLOT_COLORS[0])
        median.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color(PLOT_COLORS[0])
    
    for cap in bp['caps']:
        cap.set_color(PLOT_COLORS[0])
    
    for flier in bp['fliers']:
        flier.set_markerfacecolor(PLOT_COLORS[0])
        flier.set_markeredgecolor(PLOT_COLORS[0])
    
    ax.axhline(y=9.42, color=PLOT_COLORS[4], linestyle='--', label='100% Breakeven')
    ax.axhline(y=14.07, color=PLOT_COLORS[1], linestyle='--', label='10% Breakeven')
    ax.set_xlabel('Month')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot2 Monthly electricity price distribution with breakeven lines.png', dpi=300, bbox_inches='tight')
    print("Plot 2 saved as 'plot2 Monthly electricity price distribution with breakeven lines.png'")
    
    return fig

def create_plot_3(prices, results_100, results_10, results_dynamic, params):
    """Create plot 3: Price histogram."""
    
    # A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(prices, bins=50, alpha=0.7, color=PLOT_COLORS[0], edgecolor='black')
    ax.axvline(x=9.42, color=PLOT_COLORS[4], linestyle='--', label='100% Breakeven')
    ax.axvline(x=14.07, color=PLOT_COLORS[1], linestyle='--', label='10% Breakeven')
    ax.axvline(x=np.mean(prices), color=PLOT_COLORS[3], linestyle='-', label=f'Average')
    ax.set_xlabel('Price (EUR/MWh)')
    ax.set_ylabel('Frequency (hours)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot3 Electricity price histogram with statistical indicators.png', dpi=300, bbox_inches='tight')
    print("Plot 3 saved as 'plot3 Electricity price histogram with statistical indicators.png'")
    
    return fig

def create_plot_4(prices, results_100, results_10, results_dynamic, params):
    """Create plot 4 separately (economic comparison)."""
    
    # A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    strategies = ['100% All Year', '10% All Year', 'Dynamic Opt']
    profits = [results_100['total_profit']/1e6, results_10['total_profit']/1e6, results_dynamic['total_profit']/1e6]
    revenues = [results_100['total_revenue']/1e6, results_10['total_revenue']/1e6, results_dynamic['total_revenue']/1e6]
    costs = [results_100['total_costs']/1e6, results_10['total_costs']/1e6, results_dynamic['total_costs']/1e6]  # Back to positive values
    
    x = np.arange(len(strategies))
    width = 0.25
    
    bars1 = ax.bar(x - width, costs, width, label='Total Costs', color=PLOT_COLORS[3])
    bars2 = ax.bar(x, revenues, width, label='Revenue', color=PLOT_COLORS[0])
    bars3 = ax.bar(x + width, profits, width, label='Profit', color=PLOT_COLORS[4])
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Value (Million EUR)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 0.5,
                     f'€{height:.1f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plot4 Annual economic performance comparison by strategy.png', dpi=300, bbox_inches='tight')
    print("Plot 4 saved as 'plot4 Annual economic performance comparison by strategy.png'")
    
    return fig

def create_plots_5_7_8_9(prices, results_100, results_10, results_dynamic, params):
    """Create plots 5, 7, 8, 9 in one image (excluding 6)."""
    
    # Create date range for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(len(prices))]
    
    # A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ((ax5, ax7), (ax8, ax9)) = plt.subplots(2, 2, figsize=(10, 7))
    
    # Plot 5: Cost breakdown
    cost_categories = ['Electricity', 'CO2', 'Variable OPEX', 'Fixed OPEX', 'CAPEX']
    
    costs_100 = [results_100['electricity_costs']/1e6, results_100['co2_costs']/1e6, 
                 results_100['variable_opex']/1e6, results_100['fixed_opex']/1e6, 
                 results_100['capex']/1e6]
    costs_10 = [results_10['electricity_costs']/1e6, results_10['co2_costs']/1e6, 
                results_10['variable_opex']/1e6, results_10['fixed_opex']/1e6, 
                results_10['capex']/1e6]
    costs_dynamic = [results_dynamic['electricity_costs']/1e6, results_dynamic['co2_costs']/1e6, 
                     results_dynamic['variable_opex']/1e6, results_dynamic['fixed_opex']/1e6, 
                     results_dynamic['capex']/1e6]
    
    x = np.arange(len(cost_categories))
    width = 0.25
    
    ax5.bar(x - width, costs_100, width, label='100% All Year', color=PLOT_COLORS[3])
    ax5.bar(x, costs_10, width, label='10% All Year', color=PLOT_COLORS[1])
    ax5.bar(x + width, costs_dynamic, width, label='Dynamic Opt', color=PLOT_COLORS[4])
    
    ax5.set_xlabel('Cost Category')
    ax5.set_ylabel('Cost (Million EUR)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(cost_categories, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 7: Dynamic operation profile (sample months)
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
        
        ax7.plot(sample_dates, power_profile, color=PLOT_COLORS[0], linewidth=1, label='Power Consumption')
        ax7.fill_between(sample_dates, power_profile, alpha=0.3, color=PLOT_COLORS[0])
        
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Power (MW)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Profit improvement over time
    baseline_profit = results_100['total_profit']
    improvements = [
        0,  # 100% all year baseline
        (results_10['total_profit'] - baseline_profit) / 1e6,  # 10% all year
        (results_dynamic['total_profit'] - baseline_profit) / 1e6  # Dynamic
    ]
    
    strategies = ['100% All Year', '10% All Year', 'Dynamic Opt']
    colors = [PLOT_COLORS[3], PLOT_COLORS[1], PLOT_COLORS[4]]
    bars = ax8.bar(strategies, improvements, color=colors)
    ax8.set_ylabel('Profit Improvement (Million EUR)')
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.1,
                 f'€{improvement:.1f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Plot 9: Electricity cost savings analysis
    elec_costs = [results_100['electricity_costs']/1e6, results_10['electricity_costs']/1e6, 
                  results_dynamic['electricity_costs']/1e6]
    savings_vs_100 = [0, results_100['electricity_costs'] - results_10['electricity_costs'], 
                      results_100['electricity_costs'] - results_dynamic['electricity_costs']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax9.bar(x - width/2, elec_costs, width, label='Electricity Cost (M€)', color=PLOT_COLORS[3])
    ax9_twin = ax9.twinx()
    bars2 = ax9_twin.bar(x + width/2, [s/1e6 for s in savings_vs_100], width, label='Savings vs 100% (M€)', color=PLOT_COLORS[4])
    
    ax9.set_xlabel('Strategy')
    ax9.set_ylabel('Electricity Cost (Million EUR)', color='red')
    ax9_twin.set_ylabel('Savings vs 100% All Year (Million EUR)', color='green')
    ax9.set_xticks(x)
    ax9.set_xticklabels(strategies, rotation=45)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot5 Detailed analysis cost breakdown operation profile and savings.png', dpi=300, bbox_inches='tight')
    print("Plot 5 saved as 'plot5 Detailed analysis cost breakdown operation profile and savings.png'")
    
    return fig

def create_cost_per_ton_analysis(results_100, results_10, results_dynamic, params):
    """Create cost per ton analysis with stacked bars (100% vs Dynamic only)."""
    
    strategies = ['100% All Year', 'Dynamic Opt']
    results = [results_100, results_dynamic]
    
    # Calculate cost per ton for each strategy
    cost_per_ton_data = []
    
    for result in results:
        total_production_ton = result['total_production'] / 1000  # Convert kg to tons
        cost_per_ton = {
            'Electricity': result['electricity_costs'] / total_production_ton,
            'CO2 Purchase': result['co2_costs'] / total_production_ton,
            'Variable OPEX': result['variable_opex'] / total_production_ton,
            'Fixed OPEX': result['fixed_opex'] / total_production_ton,
            'CAPEX': result['capex'] / total_production_ton
        }
        cost_per_ton_data.append(cost_per_ton)
    
    # Create stacked bar chart - A4-sized rectangular format (width > height) - smaller for better text readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for each cost component using KTH scheme
    colors = {
        'Electricity': PLOT_COLORS[3],      # Deep red
        'CO2 Purchase': PLOT_COLORS[2],     # Teal
        'Variable OPEX': PLOT_COLORS[0],    # KTH blue
        'Fixed OPEX': PLOT_COLORS[4],       # Dark green
        'CAPEX': PLOT_COLORS[5]             # Dark gray
    }
    
    # Prepare data for stacked bars
    categories = list(cost_per_ton_data[0].keys())
    x = np.arange(len(strategies))
    width = 0.4  # Thinner bars
    
    # Calculate bottom positions for stacking
    bottoms = np.zeros(len(strategies))
    
    # Plot each cost component
    for i, category in enumerate(categories):
        values = [cost_per_ton_data[j][category] for j in range(len(strategies))]
        ax.bar(x, values, width, bottom=bottoms, label=category, color=colors[category])
        bottoms += values
    
    # Customize the plot
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Cost per Tonne Methanol (EUR/tonne)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45)
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limit to provide headroom for labels
    ax.set_ylim(0, 1500)
    
    # Add total cost per tonne labels
    total_costs_per_tonne = [sum(cost_per_ton_data[i].values()) for i in range(len(strategies))]
    for i, total in enumerate(total_costs_per_tonne):
        ax.text(i, total + 50, f'€{total:.0f}/tonne', ha='center', va='bottom', fontweight='bold')
    
    # Add value labels for each component
    for i, strategy in enumerate(strategies):
        bottom = 0
        for category in categories:
            value = cost_per_ton_data[i][category]
            if value > 50:  # Only label if significant
                ax.text(i, bottom + value/2, f'€{value:.0f}', ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
            bottom += value
    
    plt.tight_layout()
    plt.savefig('plot6 Cost breakdown per tonne methanol by strategy.png', dpi=300, bbox_inches='tight')
    print("Plot 6 saved as 'plot6 Cost breakdown per tonne methanol by strategy.png'")
    
    # Print summary
    print("\nCOST PER TONNE ANALYSIS (100% vs Dynamic)")
    print("=" * 50)
    for i, strategy in enumerate(strategies):
        print(f"\n{strategy}:")
        total = 0
        for category in categories:
            value = cost_per_ton_data[i][category]
            total += value
            print(f"  {category}: €{value:.0f}/tonne")
        print(f"  TOTAL: €{total:.0f}/tonne")
    
    # Calculate savings
    savings = cost_per_ton_data[0]['Electricity'] - cost_per_ton_data[1]['Electricity']
    total_savings = sum(cost_per_ton_data[0].values()) - sum(cost_per_ton_data[1].values())
    print(f"\nDynamic vs 100% All Year:")
    print(f"  Electricity savings: €{savings:.0f}/tonne")
    print(f"  Total cost savings: €{total_savings:.0f}/tonne")
    
    return fig

def main():
    """Main custom analysis function."""
    
    print("CUSTOM E-METHANOL PLANT ANALYSIS")
    print("=" * 60)
    print("Creating custom visualizations as requested...")
    print()
    
    # Load data
    print("Loading full year 2023 data...")
    params = get_parameters()
    data = load_data()
    prices = data['price']
    
    print(f"Loaded {len(prices)} hours of electricity price data")
    print()
    
    # Calculate strategies
    print("Calculating strategies...")
    results_100 = calculate_fixed_strategy_costs(prices, params, 100)
    results_10 = calculate_fixed_strategy_costs(prices, params, 10)
    results_dynamic = run_dynamic_optimization(prices, params)
    
    if results_dynamic is None:
        print("Dynamic optimization failed!")
        return
    
    # Create custom visualizations
    print("\nCreating custom visualizations...")
    
    # Individual plots 1, 2, 3
    print("Creating plot 1...")
    create_plot_1(prices, results_100, results_10, results_dynamic, params)
    
    print("Creating plot 2...")
    create_plot_2(prices, results_100, results_10, results_dynamic, params)
    
    print("Creating plot 3...")
    create_plot_3(prices, results_100, results_10, results_dynamic, params)
    
    # Plot 4 separately
    print("Creating plot 4...")
    create_plot_4(prices, results_100, results_10, results_dynamic, params)
    
    # Plots 5, 7, 8, 9 in one image (excluding 6)
    print("Creating plots 5, 7, 8, 9...")
    create_plots_5_7_8_9(prices, results_100, results_10, results_dynamic, params)
    
    # New cost per ton analysis (100% vs Dynamic only)
    print("Creating cost per ton analysis...")
    create_cost_per_ton_analysis(results_100, results_10, results_dynamic, params)
    
    print("\nCUSTOM ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("• plot1 Full year 2023 electricity prices with breakeven thresholds.png")
    print("• plot2 Monthly electricity price distribution with breakeven lines.png")
    print("• plot3 Electricity price histogram with statistical indicators.png")
    print("• plot4 Annual economic performance comparison by strategy.png")
    print("• plot5 Detailed analysis cost breakdown operation profile and savings.png")
    print("• plot6 Cost breakdown per tonne methanol by strategy.png")

if __name__ == "__main__":
    main()
