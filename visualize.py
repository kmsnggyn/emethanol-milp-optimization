"""
Visualization module for e-methanol plant optimization results.
Creates comprehensive plots to analyze optimization performance and operational patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_results_data():
    """Load optimization results and price data."""
    # Load price data
    price_data = pd.read_csv('data/dummy_prices.csv')
    
    # Create datetime index (assuming 2024 as base year)
    start_date = datetime(2024, 1, 1)
    price_data['datetime'] = [start_date + timedelta(hours=i) for i in range(len(price_data))]
    price_data.set_index('datetime', inplace=True)
    
    return price_data

def generate_sample_results(price_data):
    """Generate sample optimization results for plotting (replace with actual results)."""
    # This is a placeholder - in real usage, you'd load actual optimization results
    np.random.seed(42)  # For reproducible results
    
    n_hours = len(price_data)
    results = pd.DataFrame(index=price_data.index)
    
    # Simulate operational decisions based on price thresholds with some noise
    price_threshold = 86.6  # Break-even price from analysis
    base_decisions = (price_data['price_eur_per_mwh'] > price_threshold).astype(int)
    
    # Add some realistic operational constraints (minimum run times, etc.)
    operational_state = []
    current_state = 0
    state_duration = 0
    min_duration = 3  # Minimum 3 hours in any state
    
    for i, should_run in enumerate(base_decisions):
        if state_duration < min_duration:
            # Continue current state
            operational_state.append(current_state)
            state_duration += 1
        else:
            # Can change state
            if should_run != current_state:
                current_state = should_run
                state_duration = 1
            else:
                state_duration += 1
            operational_state.append(current_state)
    
    results['operational_state'] = operational_state
    results['capacity_mw'] = results['operational_state'].map({0: 10, 1: 100})
    results['capacity_percent'] = results['operational_state'].map({0: 10, 1: 100})
    results['price'] = price_data['price_eur_per_mwh']
    
    # Calculate ramp events
    results['ramp_up'] = (results['operational_state'].diff() == 1).astype(int)
    results['ramp_down'] = (results['operational_state'].diff() == -1).astype(int)
    results['ramp_event'] = results['ramp_up'] + results['ramp_down']
    
    return results

def plot_operational_overview(results, time_period='week'):
    """Plot operational overview showing price and capacity decisions."""
    create_plots_directory()
    
    if time_period == 'week':
        # Show first week for detailed view
        plot_data = results.iloc[:168]  # First 7 days
        title_suffix = "First Week"
        filename = "operational_overview_week.png"
    elif time_period == 'month':
        # Show first month
        plot_data = results.iloc[:720]  # First 30 days
        title_suffix = "First Month"
        filename = "operational_overview_month.png"
    else:
        # Show full year but sampled
        plot_data = results.iloc[::24]  # Daily samples
        title_suffix = "Full Year (Daily Samples)"
        filename = "operational_overview_year.png"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Electricity price
    ax1.plot(plot_data.index, plot_data['price'], color='blue', linewidth=1, alpha=0.8)
    ax1.axhline(y=86.6, color='red', linestyle='--', alpha=0.7, label='Break-even price (€86.6/MWh)')
    ax1.set_ylabel('Electricity Price (€/MWh)', fontsize=12)
    ax1.set_title(f'E-methanol Plant Optimization Results - {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Operational capacity
    ax2.fill_between(plot_data.index, plot_data['capacity_percent'], 
                     color='green', alpha=0.6, step='post', label='Operating Capacity')
    ax2.set_ylabel('Operating Capacity (%)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add ramp event markers
    ramp_events = plot_data[plot_data['ramp_event'] == 1]
    if not ramp_events.empty:
        for time in ramp_events.index:
            ax2.axvline(x=time, color='orange', linestyle='-', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_operational_statistics(results):
    """Plot various operational statistics and metrics."""
    create_plots_directory()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Capacity factor distribution
    capacity_factor_daily = results['capacity_percent'].resample('D').mean()
    ax1.hist(capacity_factor_daily, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Daily Average Capacity Factor (%)')
    ax1.set_ylabel('Frequency (Days)')
    ax1.set_title('Distribution of Daily Capacity Factors')
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs Operating Decision
    price_bins = np.arange(0, results['price'].max() + 10, 10)
    price_groups = pd.cut(results['price'], bins=price_bins)
    operation_by_price = results.groupby(price_groups)['operational_state'].mean()
    
    ax2.bar(range(len(operation_by_price)), operation_by_price.values * 100, 
           color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Price Range (€/MWh)')
    ax2.set_ylabel('% Time at High Capacity')
    ax2.set_title('Operating Decisions by Price Range')
    ax2.set_xticks(range(len(operation_by_price)))
    ax2.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                        for interval in operation_by_price.index], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly operational patterns
    monthly_stats = results.groupby(results.index.month).agg({
        'capacity_percent': 'mean',
        'price': 'mean',
        'ramp_event': 'sum'
    })
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax3_twin = ax3.twinx()
    bars = ax3.bar(range(12), monthly_stats['capacity_percent'], 
                   color='lightgreen', alpha=0.7, label='Avg Capacity Factor (%)')
    line = ax3_twin.plot(range(12), monthly_stats['price'], 
                        color='red', marker='o', linewidth=2, label='Avg Price (€/MWh)')
    
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Capacity Factor (%)', color='green')
    ax3_twin.set_ylabel('Average Price (€/MWh)', color='red')
    ax3.set_title('Monthly Operational Patterns')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(month_names)
    ax3.grid(True, alpha=0.3)
    
    # 4. Ramp frequency analysis
    ramp_by_hour = results.groupby(results.index.hour)['ramp_event'].sum()
    ax4.bar(range(24), ramp_by_hour.values, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Total Ramp Events')
    ax4.set_title('Ramp Events by Hour of Day')
    ax4.set_xticks(range(0, 24, 2))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/operational_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_economic_analysis(results):
    """Plot economic analysis including revenue and costs."""
    create_plots_directory()
    
    # Calculate economic metrics
    # Assume some economic parameters (replace with actual values)
    power_consumption_high = 100  # MW
    power_consumption_low = 10   # MW
    methanol_production_high = 100  # tons/h (placeholder)
    methanol_production_low = 10   # tons/h (placeholder)
    methanol_price = 400  # €/ton (placeholder)
    ramp_cost = 1625  # € per ramp event
    
    results['power_consumption'] = results['capacity_mw']
    results['electricity_cost'] = results['power_consumption'] * results['price']
    results['methanol_production'] = results['operational_state'].map({0: methanol_production_low, 1: methanol_production_high})
    results['methanol_revenue'] = results['methanol_production'] * methanol_price
    results['ramp_costs'] = results['ramp_event'] * ramp_cost
    results['hourly_profit'] = results['methanol_revenue'] - results['electricity_cost'] - results['ramp_costs']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cumulative profit over time
    cumulative_profit = results['hourly_profit'].cumsum() / 1000  # Convert to k€
    ax1.plot(results.index[::24], cumulative_profit[::24], color='green', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Profit (k€)')
    ax1.set_title('Cumulative Profit Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily profit distribution
    daily_profit = results['hourly_profit'].resample('D').sum() / 1000
    ax2.hist(daily_profit, bins=30, color='lightblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Daily Profit (k€)')
    ax2.set_ylabel('Frequency (Days)')
    ax2.set_title('Distribution of Daily Profits')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost breakdown
    total_electricity_cost = results['electricity_cost'].sum() / 1000000  # M€
    total_ramp_costs = results['ramp_costs'].sum() / 1000  # k€
    total_revenue = results['methanol_revenue'].sum() / 1000000  # M€
    
    costs = [total_electricity_cost, total_ramp_costs/1000]  # Convert ramp costs to M€
    revenues = [total_revenue]
    
    x = ['Electricity Cost', 'Ramp Costs']
    ax3.bar(x, costs, color=['red', 'orange'], alpha=0.7)
    ax3.bar(['Methanol Revenue'], revenues, color='green', alpha=0.7)
    ax3.set_ylabel('Amount (M€)')
    ax3.set_title('Annual Cost and Revenue Breakdown')
    ax3.grid(True, alpha=0.3)
    
    # 4. Profit vs price scatter
    scatter_data = results.sample(n=min(1000, len(results)))  # Sample for readability
    scatter = ax4.scatter(scatter_data['price'], scatter_data['hourly_profit'], 
                         c=scatter_data['operational_state'], cmap='RdYlGn', alpha=0.6)
    ax4.set_xlabel('Electricity Price (€/MWh)')
    ax4.set_ylabel('Hourly Profit (€)')
    ax4.set_title('Hourly Profit vs Electricity Price')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Operational State')
    
    plt.tight_layout()
    plt.savefig('plots/economic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_summary_dashboard(results):
    """Create a comprehensive summary dashboard."""
    create_plots_directory()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main time series plot (top row, spans 3 columns)
    ax_main = fig.add_subplot(gs[0, :3])
    
    # Show a representative month (e.g., January)
    plot_data = results.iloc[:744]  # First 31 days
    
    # Create twin axis for price and capacity
    ax_price = ax_main
    ax_capacity = ax_main.twinx()
    
    # Plot price
    line1 = ax_price.plot(plot_data.index, plot_data['price'], 
                         color='blue', linewidth=1, alpha=0.8, label='Price')
    ax_price.axhline(y=86.6, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax_price.set_ylabel('Electricity Price (€/MWh)', color='blue')
    ax_price.tick_params(axis='y', labelcolor='blue')
    
    # Plot capacity as step function
    line2 = ax_capacity.fill_between(plot_data.index, plot_data['capacity_percent'], 
                                    color='green', alpha=0.4, step='post', label='Capacity')
    ax_capacity.set_ylabel('Operating Capacity (%)', color='green')
    ax_capacity.set_ylim(0, 110)
    ax_capacity.tick_params(axis='y', labelcolor='green')
    
    ax_main.set_title('E-methanol Plant Operations (January 2024)', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # Key metrics (top right)
    ax_metrics = fig.add_subplot(gs[0, 3])
    ax_metrics.axis('off')
    
    # Calculate key metrics
    total_ramps = results['ramp_event'].sum()
    avg_capacity = results['capacity_percent'].mean()
    high_capacity_hours = (results['operational_state'] == 1).sum()
    capacity_factor = high_capacity_hours / len(results) * 100
    
    metrics_text = f"""
    KEY METRICS
    
    Annual Capacity Factor: {avg_capacity:.1f}%
    High Capacity Hours: {high_capacity_hours:,}
    Total Ramp Events: {total_ramps}
    Avg Ramps/Month: {total_ramps/12:.1f}
    
    Break-even Price: €86.6/MWh
    Max Price: €{results['price'].max():.1f}/MWh
    Min Price: €{results['price'].min():.1f}/MWh
    """
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Operational pattern (middle left)
    ax_pattern = fig.add_subplot(gs[1, :2])
    hourly_pattern = results.groupby(results.index.hour).agg({
        'operational_state': 'mean',
        'price': 'mean'
    })
    
    ax_pattern_twin = ax_pattern.twinx()
    bars = ax_pattern.bar(range(24), hourly_pattern['operational_state'] * 100, 
                         color='lightgreen', alpha=0.7, label='% High Capacity')
    line = ax_pattern_twin.plot(range(24), hourly_pattern['price'], 
                               color='red', marker='o', linewidth=2, label='Avg Price')
    
    ax_pattern.set_xlabel('Hour of Day')
    ax_pattern.set_ylabel('% Time at High Capacity', color='green')
    ax_pattern_twin.set_ylabel('Average Price (€/MWh)', color='red')
    ax_pattern.set_title('Daily Operational Pattern')
    ax_pattern.set_xticks(range(0, 24, 2))
    ax_pattern.grid(True, alpha=0.3)
    
    # Price distribution (middle right)
    ax_price_dist = fig.add_subplot(gs[1, 2:])
    
    # Create price histogram with operational overlay
    price_bins = np.arange(0, results['price'].max() + 10, 5)
    
    # Separate prices by operational state
    prices_low = results[results['operational_state'] == 0]['price']
    prices_high = results[results['operational_state'] == 1]['price']
    
    ax_price_dist.hist([prices_low, prices_high], bins=price_bins, 
                      color=['red', 'green'], alpha=0.6, 
                      label=['Low Capacity (10%)', 'High Capacity (100%)'],
                      stacked=False)
    ax_price_dist.axvline(x=86.6, color='black', linestyle='--', linewidth=2, label='Break-even')
    ax_price_dist.set_xlabel('Electricity Price (€/MWh)')
    ax_price_dist.set_ylabel('Hours')
    ax_price_dist.set_title('Price Distribution by Operation Mode')
    ax_price_dist.legend()
    ax_price_dist.grid(True, alpha=0.3)
    
    # Monthly summary (bottom)
    ax_monthly = fig.add_subplot(gs[2, :])
    
    monthly_data = results.groupby(results.index.month).agg({
        'capacity_percent': 'mean',
        'price': 'mean',
        'ramp_event': 'sum'
    })
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    x = np.arange(12)
    width = 0.25
    
    bars1 = ax_monthly.bar(x - width, monthly_data['capacity_percent'], width, 
                          label='Avg Capacity Factor (%)', color='lightblue', alpha=0.7)
    bars2 = ax_monthly.bar(x, monthly_data['price'], width, 
                          label='Avg Price (€/MWh)', color='orange', alpha=0.7)
    bars3 = ax_monthly.bar(x + width, monthly_data['ramp_event'], width, 
                          label='Ramp Events', color='red', alpha=0.7)
    
    ax_monthly.set_xlabel('Month')
    ax_monthly.set_ylabel('Value')
    ax_monthly.set_title('Monthly Summary Statistics')
    ax_monthly.set_xticks(x)
    ax_monthly.set_xticklabels(month_names)
    ax_monthly.legend()
    ax_monthly.grid(True, alpha=0.3)
    
    plt.savefig('plots/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_all_plots():
    """Generate all visualization plots."""
    print("Loading data and generating optimization results...")
    
    # Load price data
    price_data = load_results_data()
    
    # Generate sample results (replace with actual optimization results)
    results = generate_sample_results(price_data)
    
    print("Creating visualization plots...")
    
    # Generate all plots
    print("1. Creating operational overview plots...")
    plot_operational_overview(results, 'week')
    plot_operational_overview(results, 'month')
    
    print("2. Creating operational statistics...")
    plot_operational_statistics(results)
    
    print("3. Creating economic analysis...")
    plot_economic_analysis(results)
    
    print("4. Creating summary dashboard...")
    plot_summary_dashboard(results)
    
    print("\nAll plots saved to 'plots/' directory!")
    print("Generated plots:")
    print("- operational_overview_week.png")
    print("- operational_overview_month.png") 
    print("- operational_statistics.png")
    print("- economic_analysis.png")
    print("- summary_dashboard.png")
    
    return results

if __name__ == "__main__":
    # Generate all plots
    results = generate_all_plots()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total hours simulated: {len(results):,}")
    print(f"Average capacity factor: {results['capacity_percent'].mean():.1f}%")
    print(f"Hours at high capacity: {(results['operational_state']==1).sum():,}")
    print(f"Total ramp events: {results['ramp_event'].sum()}")
    print(f"Average ramps per month: {results['ramp_event'].sum()/12:.1f}")
    print(f"Price range: €{results['price'].min():.1f} - €{results['price'].max():.1f}/MWh")
    print(f"Hours above break-even: {(results['price'] > 86.6).sum():,}")
