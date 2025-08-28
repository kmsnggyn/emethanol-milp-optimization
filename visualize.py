"""
Simple visualization for e-methanol plant optimization results.
Creates a single plot showing capacity decisions over the full year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

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
    """Generate sample optimization results for plotting."""
    np.random.seed(42)  # For reproducible results
    
    n_hours = len(price_data)
    results = pd.DataFrame(index=price_data.index)
    
    # Simulate operational decisions based on price thresholds
    price_threshold = 86.6  # Break-even price
    base_decisions = (price_data['price_eur_per_mwh'] > price_threshold).astype(int)
    
    # Add operational constraints (minimum run times)
    operational_state = []
    current_state = 0
    state_duration = 0
    min_duration = 3  # Minimum 3 hours in any state
    
    for i, should_run in enumerate(base_decisions):
        if state_duration < min_duration:
            operational_state.append(current_state)
            state_duration += 1
        else:
            if should_run != current_state:
                current_state = should_run
                state_duration = 1
            else:
                state_duration += 1
            operational_state.append(current_state)
    
    results['operational_state'] = operational_state
    results['capacity'] = results['operational_state'].map({0: 10, 1: 100})
    
    return results

def plot_capacity_over_year(results):
    """Create a single plot showing capacity over the full year."""
    create_plots_directory()
    
    # Create the plot
    plt.figure(figsize=(16, 6))
    
    # Plot capacity as step function
    plt.step(results.index, results['capacity'], where='post', linewidth=0.8, color='steelblue')
    
    # Formatting
    plt.xlabel('Time (2024)', fontsize=12)
    plt.ylabel('Plant Capacity (%)', fontsize=12)
    plt.title('E-methanol Plant Capacity Over Full Year', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Capacity')
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% Capacity')
    
    # Format x-axis to show months
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/capacity_over_year.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    hours_at_100 = (results['capacity'] == 100).sum()
    hours_at_10 = (results['capacity'] == 10).sum()
    capacity_factor = results['capacity'].mean()
    
    print(f"\nCapacity Utilization Summary:")
    print(f"Hours at 100% capacity: {hours_at_100:,} ({hours_at_100/len(results)*100:.1f}%)")
    print(f"Hours at 10% capacity: {hours_at_10:,} ({hours_at_10/len(results)*100:.1f}%)")
    print(f"Average capacity factor: {capacity_factor:.1f}%")
    print(f"Plot saved as: plots/capacity_over_year.png")

if __name__ == "__main__":
    print("Loading data and generating capacity plot...")
    
    # Load price data
    price_data = load_results_data()
    
    # Generate sample results
    results = generate_sample_results(price_data)
    
    # Create the plot
    plot_capacity_over_year(results)
