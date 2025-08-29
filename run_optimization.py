#!/usr/bin/env python3
"""
E-Methanol Plant Optimization - Minimal Analysis Suite

This consolidated script demonstrates the adaptive MPC system with minimal complexity.

Usage:
    python run_optimization.py

Requirements:
    - pyomo, gurobipy, pandas, numpy, matplotlib
    - Excel files in electricity_data/ directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Core optimization imports
from mpc import ForecastingMPC, AdaptivePatternForecaster
from model import get_parameters


def run_mpc_demo():
    """Run a demonstration of the Adaptive MPC system."""
    print("ADAPTIVE MPC DEMONSTRATION")
    print("=" * 50)
    
    # Load electricity data
    print("Loading 2019 electricity price data...")
    try:
        data = pd.read_excel('electricity_data/elspot_prices_2019.xlsx')
        data = data[data['Date'].notna() & (data['Date'] != 'ID')]
        data['datetime'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)
        data = data.set_index('datetime')
        prices = data['SE3'].dropna().values
        
        print(f"Loaded {len(prices)} hours of price data")
        print(f"  Average price: €{np.mean(prices):.2f}/MWh")
        print(f"  Price range: €{np.min(prices):.2f} - €{np.max(prices):.2f}/MWh")
        
    except Exception as e:
        print(f"Failed to load electricity data: {e}")
        return None
    
    # Get parameters
    params = get_parameters()
    
    # Initialize MPC controller
    print("\nInitializing Adaptive MPC Controller...")
    mpc_controller = ForecastingMPC()
    
    # Run MPC simulation for first week
    week_hours = 168
    print(f"Running MPC simulation (first {week_hours} hours)...")
    try:
        result = mpc_controller.run_mpc_simulation(
            actual_prices=prices[:week_hours],
            params=params,
            verbose=False
        )
        
        if result:
            print(f"\nMPC Simulation Complete!")
            print(f"  Total Profit: €{result['total_profit']:,.0f}")
            print(f"  Production: {result['total_production']:,.0f} tons")
            print(f"  Capacity Factor: {result['capacity_factor']:.1f}%")
            print(f"  Ramp Events: {result['total_ramps']}")
            
            return result
        else:
            print("MPC simulation failed")
            return None
            
    except Exception as e:
        print(f"MPC simulation error: {e}")
        return None


def analyze_forecasting():
    """Analyze the adaptive forecasting capabilities."""
    print("\nFORECASTING ANALYSIS")
    print("=" * 30)
    
    try:
        # Load price data
        data = pd.read_excel('electricity_data/elspot_prices_2019.xlsx')
        data = data[data['Date'].notna() & (data['Date'] != 'ID')]
        data['datetime'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)
        data = data.set_index('datetime')
        prices = data['SE3'].dropna()
        
        # Test forecaster learning
        forecaster = AdaptivePatternForecaster()
        errors = []
        
        # Test first week - matches our progress bar intervals  
        week_hours = 168
        test_hours = min(week_hours, len(prices) - 1)
        for hour in range(test_hours):
            if hour < len(prices) - 1:
                actual_price = prices.iloc[hour]
                next_actual = prices.iloc[hour + 1]
                
                # Generate forecast
                forecast = forecaster.forecast(hour, 1)
                if forecast:
                    error = abs(forecast[0] - next_actual)
                    errors.append(error)
                
                # Update with observation
                forecaster.update_with_observation(hour, actual_price)
        
        if errors:
            print(f"Forecast Performance ({test_hours} hours):")
            print(f"   • Mean error: €{np.mean(errors):.2f}/MWh")
            print(f"   • Std error: €{np.std(errors):.2f}/MWh")
            print(f"   • Best forecast: €{np.min(errors):.2f}/MWh error")
            print(f"   • Worst forecast: €{np.max(errors):.2f}/MWh error")
        
        # Calculate dynamic price thresholds based on data
        price_75th = np.percentile(prices, 75)
        price_25th = np.percentile(prices, 25) 
        
        # Market insights with dynamic thresholds
        print(f"\nMarket Insights:")
        print(f"   • Average price: €{prices.mean():.2f}/MWh")
        print(f"   • Hours above €{price_75th:.0f}/MWh: {(prices > price_75th).sum()} ({(prices > price_75th).mean()*100:.1f}%)")
        print(f"   • Hours below €{price_25th:.0f}/MWh: {(prices < price_25th).sum()} ({(prices < price_25th).mean()*100:.1f}%)")
        
        return errors
        
    except Exception as e:
        print(f"Forecasting analysis failed: {e}")
        return None


def show_key_results():
    """Display the key results from our full system analysis."""
    params = get_parameters()
    
    # Calculate breakeven price dynamically
    methanol_production = 2.56  # ton/hr from params M_100
    revenue_per_hour = methanol_production * params['Price_Methanol']
    fixed_costs_per_hour = (params['Annualized_CAPEX'] + params['OPEX_Fixed']) / 8760
    other_costs = params['C_100'] * params['Price_CO2'] + params['OPEX_Variable'] + fixed_costs_per_hour
    net_before_electricity = revenue_per_hour - other_costs
    breakeven_price = net_before_electricity / params['P_100']
    
    print("\nKEY SYSTEM RESULTS")
    print("=" * 40)
    print("Current Plant Economics (based on parameters):")
    print()
    print("Plant Configuration:")
    print(f"  • Methanol Production: {methanol_production} ton/hr")
    print(f"  • Power Consumption: {params['P_100']} MW")
    print(f"  • Methanol Price: €{params['Price_Methanol']}/ton")
    print(f"  • Annual Fixed Costs: €{(params['Annualized_CAPEX'] + params['OPEX_Fixed'])/1e6:.1f}M")
    print()
    print("Economic Analysis:")
    print(f"  • Revenue: €{revenue_per_hour:.0f}/hour")
    print(f"  • Fixed Costs: €{fixed_costs_per_hour:.0f}/hour")
    print(f"  • Variable Costs: €{other_costs - fixed_costs_per_hour:.0f}/hour")
    print(f"  • Breakeven Electricity: €{breakeven_price:.1f}/MWh")
    print()
    print("Key Insights:")
    print("  • Plant operates only when electricity < breakeven price")
    print("  • Optimization enables selective operation during profitable periods")
    print("  • Shutdown capability prevents losses during expensive periods")
    print("  • Adaptive forecasting enables intelligent operation")


def create_simple_visualization(mpc_result=None, forecast_errors=None):
    """Create a simple visualization if matplotlib is available."""
    try:
        params = get_parameters()
        
        # Calculate actual values based on current parameters
        methanol_production = 2.56
        revenue_per_hour = methanol_production * params['Price_Methanol']
        fixed_costs_per_hour = (params['Annualized_CAPEX'] + params['OPEX_Fixed']) / 8760
        other_costs = params['C_100'] * params['Price_CO2'] + params['OPEX_Variable'] + fixed_costs_per_hour
        net_before_electricity = revenue_per_hour - other_costs
        breakeven_price = net_before_electricity / params['P_100']
        
        # Create economic analysis chart
        categories = ['Revenue', 'Fixed Costs', 'Variable Costs', 'Net (before elec.)']
        values = [revenue_per_hour, -fixed_costs_per_hour, -(other_costs - fixed_costs_per_hour), net_before_electricity]
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Economic breakdown
        plt.subplot(1, 2, 1)
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        bars = plt.bar(categories, values, color=colors)
        
        plt.title('Hourly Economics Breakdown', fontsize=12, fontweight='bold')
        plt.ylabel('€/hour', fontsize=10)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (20 if height > 0 else -40),
                    f'€{value:.0f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Breakeven analysis
        plt.subplot(1, 2, 2)
        electricity_prices = [0, 10, 20, breakeven_price, 40, 60]
        profits = [(net_before_electricity - params['P_100'] * price) for price in electricity_prices]
        
        plt.plot(electricity_prices, profits, 'o-', linewidth=2, markersize=6, color='#3498db')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break-even')
        plt.axvline(x=breakeven_price, color='g', linestyle='--', alpha=0.7, label=f'Break-even price: €{breakeven_price:.1f}/MWh')
        
        plt.title('Profit vs Electricity Price', fontsize=12, fontweight='bold')
        plt.xlabel('Electricity Price (€/MWh)', fontsize=10)
        plt.ylabel('Profit (€/hour)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        print(f"\nEconomic analysis chart saved as 'optimization_results.png'")
        plt.close()
        
    except Exception as e:
        print(f"Note: Visualization not available ({e})")


def run_shutdown_mpc_test():
    """Test the new shutdown-capable MPC system."""
    print("\nSHUTDOWN-CAPABLE MPC TEST")
    print("=" * 50)
    
    # Load data using new shutdown parameters
    print("Loading 2019 electricity price data...")
    try:
        data = pd.read_excel('electricity_data/elspot_prices_2019.xlsx')
        data = data[data['Date'].notna() & (data['Date'] != 'ID')]
        data['datetime'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)
        data = data.set_index('datetime')
        prices = data['SE3'].dropna().values
        print(f"Loaded {len(prices)} hours of price data")
    except Exception as e:
        print(f"Failed to load electricity data: {e}")
        return None
    
    params = get_parameters()  # This now includes shutdown parameters
    
    print("\nNew Shutdown Parameters:")
    print(f"  Startup Duration: {params['startup_duration']} hours")
    print(f"  Shutdown Duration: {params['shutdown_duration']} hours")
    print(f"  Power during startup: {params['P_startup']} MW")
    print(f"  Power during shutdown transition: {params['P_shutdown_trans']} MW")
    print(f"  OPEX during shutdown state: €{params['OPEX_shutdown_state']}/hour")
    
    print("\nInitializing Shutdown-Capable MPC Controller...")
    mpc_controller = ForecastingMPC()
    
    # Run MPC simulation for full quarter (Q1 2019)
    quarter_days = 90
    simulation_hours = min(quarter_days * 24, len(prices))
    print(f"Running Shutdown MPC simulation ({simulation_hours} hours = {simulation_hours//24} days)...")
    try:
        result = mpc_controller.run_mpc_simulation(
            actual_prices=prices[:simulation_hours],
            params=params,
            verbose=True
        )
        
        if result:
            print(f"\nShutdown MPC Simulation Complete!")
            print(f"  Total Profit: €{result['total_profit']:,.0f}")
            print(f"  Production: {result['total_production']:,.0f} tons")
            print(f"  Capacity Factor: {result['capacity_factor']:.1f}%")
            print(f"  Transition Events: {result['total_ramps']}")
            
            # Analyze state distribution
            states = result['decisions']
            running_hours = sum(1 for s in states if s == 'running')
            shutdown_hours = sum(1 for s in states if s == 'shutdown')
            startup_hours = sum(1 for s in states if s == 'startup')
            shutdown_trans_hours = sum(1 for s in states if s == 'shutdown_trans')
            
            print(f"  Running Hours: {running_hours}")
            print(f"  Shutdown Hours: {shutdown_hours}")
            print(f"  Startup Hours: {startup_hours}")
            print(f"  Shutdown Transition Hours: {shutdown_trans_hours}")
            
            # Additional analysis for longer simulation
            total_hours = simulation_hours
            uptime_percentage = (running_hours / total_hours) * 100
            print(f"\nExtended Simulation Analysis ({total_hours//24} days):")
            print(f"  Plant Uptime: {uptime_percentage:.1f}%")
            print(f"  Avg Daily Production: {result['total_production']/(total_hours/24):.1f} tons/day")
            if result['total_ramps'] > 0:
                hours_per_cycle = total_hours / result['total_ramps']
                print(f"  Avg Cycle Length: {hours_per_cycle:.1f} hours")
            
            return result
        else:
            print("Shutdown MPC simulation failed")
            return None
            
    except Exception as e:
        print(f"Shutdown MPC simulation error: {e}")
        return None


def main():
    """Main demonstration function for shutdown-capable optimization."""
    print("E-METHANOL OPTIMIZATION WITH SHUTDOWN CAPABILITY")
    print("=" * 60)
    print("Enhanced MILP-based optimization with complete shutdown option")
    print("Demonstrating adaptive MPC with shutdown capability...\n")
    
    # Test the new shutdown-capable system
    shutdown_result = run_shutdown_mpc_test()
    
    # Analyze forecasting (unchanged)
    forecast_errors = analyze_forecasting()
    
    # Show key results from enhanced analysis
    if shutdown_result:
        print(f"\nENHANCED SYSTEM RESULTS")
        print("=" * 40)
        print(f"Shutdown-Capable Strategy:")
        print(f"  • Total Profit: €{shutdown_result['total_profit']:,.0f}")
        print(f"  • Production: {shutdown_result['total_production']:,.0f} tons")
        print(f"  • Capacity Factor: {shutdown_result['capacity_factor']:.1f}%")
        print(f"  • Transition Events: {shutdown_result['total_ramps']}")
        
        # Calculate efficiency metrics
        states = shutdown_result['decisions']
        running_hours = sum(1 for s in states if s == 'running')
        print(f"  • Actual running time: {running_hours}/{len(states)} hours")
        
        if running_hours > 0:
            production_efficiency = shutdown_result['total_production'] / running_hours
            print(f"  • Production efficiency: {production_efficiency:.1f} tons/running_hour")
    
    # Create simple visualization showing state distribution
    if shutdown_result:
        print("\nState Distribution Analysis:")
        states = shutdown_result['decisions']
        state_counts = {
            'running': sum(1 for s in states if s == 'running'),
            'shutdown': sum(1 for s in states if s == 'shutdown'),
            'startup': sum(1 for s in states if s == 'startup'),
            'shutdown_trans': sum(1 for s in states if s == 'shutdown_trans')
        }
        
        for state, count in state_counts.items():
            percentage = count / len(states) * 100
            print(f"  {state}: {count} hours ({percentage:.1f}%)")
    
    # Show economic analysis
    show_key_results()
    
    # Create visualization
    create_simple_visualization(shutdown_result, forecast_errors)
    
    print(f"\nENHANCED DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key enhancements in shutdown-capable system:")
    print("  • Complete plant shutdown (0% vs minimum 10%)")
    print("  • Realistic transition sequences and costs")
    print("  • No production during transition periods")
    print("  • Enhanced operational flexibility")
    print("  • Better adaptation to extreme market conditions")


if __name__ == "__main__":
    main()
