"""
Integration module to add simple plotting capability to the main optimization workflow.
This extends main.py to generate a single capacity over time visualization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Import the main optimization functions
from main import load_data, build_model, solve_model
import matplotlib.pyplot as plt

def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def extract_results_from_model(model):
    """Extract optimization results from solved Pyomo model."""
    
    # Create results DataFrame
    time_indices = list(model.T)
    results = pd.DataFrame(index=time_indices)
    
    # Extract decision variables
    results['operational_state'] = [model.x[t].value for t in time_indices]
    results['capacity'] = results['operational_state'].map({0: 10, 1: 100})
    
    # Add datetime index
    start_date = datetime(2024, 1, 1)
    results['datetime'] = [start_date + timedelta(hours=i) for i in range(len(results))]
    results.set_index('datetime', inplace=True)
    
    return results

def plot_capacity_over_year(results):
    """Create a simple plot showing capacity over the full year."""
    create_plots_directory()
    
    # Create the plot
    plt.figure(figsize=(16, 6))
    
    # Plot capacity as step function
    plt.step(results.index, results['capacity'], where='post', linewidth=0.8, color='steelblue')
    
    # Formatting
    plt.xlabel('Time (2024)', fontsize=12)
    plt.ylabel('Plant Capacity (%)', fontsize=12)
    plt.title('E-methanol Plant Capacity Over Full Year (Optimization Results)', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show months
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    
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

def run_optimization_with_plots():
    """Run the complete optimization and generate capacity visualization."""
    print("Running e-methanol plant optimization with capacity visualization...")
    
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
        
        # Generate capacity plot
        print("4. Creating capacity visualization...")
        plot_capacity_over_year(optimization_results)
        
        print("\n✅ Optimization completed successfully!")
        print("📊 Capacity plot saved as: plots/capacity_over_year.png")
        
        return optimization_results
        
    else:
        print(f"❌ Optimization failed with status: {status}")
        return None

if __name__ == "__main__":
    # Run optimization with plotting
    results = run_optimization_with_plots()
