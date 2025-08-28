"""
Integration module to add plotting capabilities to the main optimization workflow.
This extends main.py to generate visualizations of actual optimization results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Import the main optimization functions
from main import load_data, build_model, solve_model
from visualize import create_plots_directory
import matplotlib.pyplot as plt
import seaborn as sns

def extract_results_from_model(model):
    """Extract optimization results from solved Pyomo model."""
    
    # Create results DataFrame
    time_indices = list(model.T)
    results = pd.DataFrame(index=time_indices)
    
    # Extract decision variables
    results['operational_state'] = [model.x[t].value for t in time_indices]
    results['ramp_up'] = [model.y_up[t].value for t in time_indices] 
    results['ramp_down'] = [model.y_down[t].value for t in time_indices]
    
    # Calculate derived metrics
    results['capacity_mw'] = results['operational_state'].map({0: 10, 1: 100})
    results['capacity_percent'] = results['operational_state'].map({0: 10, 1: 100})
    results['ramp_event'] = results['ramp_up'] + results['ramp_down']
    
    # Add price data
    price_data = pd.read_csv('data/dummy_prices.csv')
    results['price'] = price_data['price_eur_per_mwh'].values[:len(results)]
    
    # Add datetime index
    start_date = datetime(2024, 1, 1)
    results['datetime'] = [start_date + timedelta(hours=i) for i in range(len(results))]
    results.set_index('datetime', inplace=True)
    
    return results

def plot_optimization_results(results, show_plots=True):
    """Create plots specifically for actual optimization results."""
    create_plots_directory()
    
    # Calculate key metrics
    total_hours = len(results)
    high_capacity_hours = (results['operational_state'] == 1).sum()
    capacity_factor = (results['capacity_percent'].mean())
    total_ramps = results['ramp_event'].sum()
    avg_price = results['price'].mean()
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*60)
    print(f"Total simulation period: {total_hours:,} hours ({total_hours/24:.0f} days)")
    print(f"Average capacity factor: {capacity_factor:.1f}%")
    print(f"Hours at high capacity (100%): {high_capacity_hours:,} ({high_capacity_hours/total_hours*100:.1f}%)")
    print(f"Hours at low capacity (10%): {total_hours-high_capacity_hours:,} ({(total_hours-high_capacity_hours)/total_hours*100:.1f}%)")
    print(f"Total ramp events: {total_ramps}")
    print(f"Ramp frequency: {total_ramps/total_hours*100:.2f}% of hours")
    print(f"Average electricity price: €{avg_price:.2f}/MWh")
    
    # Economic calculations
    power_high = 100  # MW
    power_low = 10    # MW
    methanol_production_high = 1.0  # tons/hour (placeholder)
    methanol_production_low = 0.1   # tons/hour (placeholder)
    methanol_price = 400  # €/ton (placeholder)
    ramp_penalty = 1625  # € per ramp
    
    results['power_consumption'] = results['operational_state'].map({0: power_low, 1: power_high})
    results['electricity_cost'] = results['power_consumption'] * results['price']
    results['methanol_production'] = results['operational_state'].map({0: methanol_production_low, 1: methanol_production_high})
    results['methanol_revenue'] = results['methanol_production'] * methanol_price
    results['ramp_costs'] = results['ramp_event'] * ramp_penalty
    results['hourly_profit'] = results['methanol_revenue'] - results['electricity_cost'] - results['ramp_costs']
    
    annual_revenue = results['methanol_revenue'].sum()
    annual_electricity_cost = results['electricity_cost'].sum() 
    annual_ramp_costs = results['ramp_costs'].sum()
    annual_profit = results['hourly_profit'].sum()
    
    print(f"\nECONOMIC ANALYSIS (with placeholder prices):")
    print(f"Annual methanol revenue: €{annual_revenue:,.0f}")
    print(f"Annual electricity cost: €{annual_electricity_cost:,.0f}")
    print(f"Annual ramp penalties: €{annual_ramp_costs:,.0f}")
    print(f"Annual net profit: €{annual_profit:,.0f}")
    print(f"ROI from optimization: {(annual_profit/annual_electricity_cost)*100:.1f}%")
    
    if not show_plots:
        return results
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: First month operational overview
    month_data = results.iloc[:744]  # First 31 days
    
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    # Price line
    line1 = ax1.plot(month_data.index, month_data['price'], 
                     color='blue', linewidth=1, alpha=0.8, label='Price')
    ax1.axhline(y=86.6, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax1.set_ylabel('Price (€/MWh)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Capacity fill
    ax1_twin.fill_between(month_data.index, month_data['capacity_percent'], 
                         color='green', alpha=0.4, step='post', label='Capacity')
    ax1_twin.set_ylabel('Capacity (%)', color='green')
    ax1_twin.set_ylim(0, 110)
    ax1_twin.tick_params(axis='y', labelcolor='green')
    
    ax1.set_title('Operational Overview (First Month)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily capacity factor distribution
    daily_capacity = results['capacity_percent'].resample('D').mean()
    axes[0, 1].hist(daily_capacity, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Daily Average Capacity Factor (%)')
    axes[0, 1].set_ylabel('Number of Days')
    axes[0, 1].set_title('Distribution of Daily Capacity Factors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Hourly operational pattern
    hourly_pattern = results.groupby(results.index.hour)['operational_state'].mean()
    bars = axes[1, 0].bar(range(24), hourly_pattern * 100, 
                         color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('% Time at High Capacity')
    axes[1, 0].set_title('Daily Operational Pattern')
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative profit
    cumulative_profit = results['hourly_profit'].cumsum() / 1000  # Convert to k€
    sample_indices = range(0, len(cumulative_profit), max(1, len(cumulative_profit)//100))
    sampled_data = cumulative_profit.iloc[sample_indices]
    sampled_times = results.index[sample_indices]
    
    axes[1, 1].plot(sampled_times, sampled_data, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Cumulative Profit (k€)')
    axes[1, 1].set_title('Cumulative Profit Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/actual_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Monthly summary
    monthly_data = results.groupby(results.index.month).agg({
        'capacity_percent': 'mean',
        'price': 'mean', 
        'ramp_event': 'sum',
        'hourly_profit': 'sum'
    })
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    x = np.arange(12)
    width = 0.2
    
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    bars1 = ax.bar(x - width, monthly_data['capacity_percent'], width, 
                   label='Avg Capacity (%)', color='lightblue', alpha=0.7)
    bars2 = ax2.bar(x, monthly_data['ramp_event'], width, 
                    label='Ramp Events', color='orange', alpha=0.7)
    line1 = ax3.plot(x + width/2, monthly_data['hourly_profit']/1000, 
                     color='green', marker='o', linewidth=2, markersize=6,
                     label='Monthly Profit (k€)')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Capacity Factor (%)', color='blue')
    ax2.set_ylabel('Ramp Events', color='orange')
    ax3.set_ylabel('Monthly Profit (k€)', color='green')
    
    ax.set_title('Monthly Performance Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() 
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/monthly_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def run_optimization_with_plots():
    """Run the complete optimization and generate visualizations."""
    print("Running e-methanol plant optimization with visualization...")
    
    # Load data and build model
    print("\n1. Loading data and building model...")
    prices, params = load_data()
    if prices is None or params is None:
        return None
        
    model = build_model(prices, params)
    
    # Solve the model
    print("2. Solving optimization model...")
    status = solve_model(model)
    
    if str(status) == 'optimal':
        print("3. Extracting and analyzing results...")
        
        # Extract results from model
        optimization_results = extract_results_from_model(model)
        
        # Generate plots
        print("4. Creating visualization plots...")
        plot_optimization_results(optimization_results)
        
        print("\n✅ Optimization completed successfully!")
        print("📊 Visualization plots saved to 'plots/' directory:")
        print("   - actual_optimization_results.png")
        print("   - monthly_performance_summary.png")
        
        return optimization_results
        
    else:
        print(f"❌ Optimization failed with status: {status}")
        return None

if __name__ == "__main__":
    # Run optimization with plotting
    results = run_optimization_with_plots()
