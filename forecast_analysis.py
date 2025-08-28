"""
Analysis script to understand MPC behavior and forecast horizon impact.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import load_data


def analyze_forecast_horizons():
    """Analyze how different forecast horizons affect decision making."""
    
    prices, params = load_data()
    if prices is None:
        return
    
    # Calculate break-even electricity price
    # At break-even: profit_100 = profit_10
    # (M_100 * Price_Methanol - P_100 * price_elec - C_100 * Price_CO2 - OPEX_Variable) = 
    # (M_10 * Price_Methanol - P_10 * price_elec - C_10 * Price_CO2 - OPEX_Variable)
    
    delta_revenue = (params["M_100"] - params["M_10"]) * params["Price_Methanol"]
    delta_power = params["P_100"] - params["P_10"]
    delta_co2 = (params["C_100"] - params["C_10"]) * params["Price_CO2"]
    
    breakeven_price = (delta_revenue - delta_co2) / delta_power
    
    print(f"Break-even electricity price: €{breakeven_price:.2f}/MWh")
    print(f"Price statistics:")
    print(f"  Mean: €{np.mean(prices):.2f}/MWh")
    print(f"  Median: €{np.median(prices):.2f}/MWh")
    print(f"  Min: €{np.min(prices):.2f}/MWh")
    print(f"  Max: €{np.max(prices):.2f}/MWh")
    
    # Analyze price distribution relative to break-even
    hours_below_breakeven = sum(1 for p in prices if p < breakeven_price)
    hours_above_breakeven = len(prices) - hours_below_breakeven
    
    print(f"\nPrice distribution relative to break-even:")
    print(f"  Hours below break-even (should run 100%): {hours_below_breakeven} ({100*hours_below_breakeven/len(prices):.1f}%)")
    print(f"  Hours above break-even (should run 10%): {hours_above_breakeven} ({100*hours_above_breakeven/len(prices):.1f}%)")
    
    # Analyze forecast horizon impact
    print("\nForecast horizon analysis:")
    horizons = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 1d, 2d, 3d, 1week
    
    for horizon in horizons:
        favorable_periods = 0
        total_periods = len(prices) - horizon + 1
        
        for start in range(total_periods):
            window_prices = prices[start:start + horizon]
            avg_price = np.mean(window_prices)
            
            if avg_price < breakeven_price:
                favorable_periods += 1
        
        favorable_pct = 100 * favorable_periods / total_periods if total_periods > 0 else 0
        print(f"  {horizon:3d}h horizon: {favorable_pct:.1f}% of periods have avg price below break-even")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price time series with break-even line
    plt.subplot(3, 2, 1)
    hours = range(len(prices))
    plt.plot(hours, prices, alpha=0.7, color='blue', linewidth=0.5)
    plt.axhline(y=breakeven_price, color='red', linestyle='--', linewidth=2, label=f'Break-even: €{breakeven_price:.1f}/MWh')
    plt.xlabel('Hour of Year')
    plt.ylabel('Electricity Price (€/MWh)')
    plt.title('Electricity Prices vs Break-even')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Price histogram
    plt.subplot(3, 2, 2)
    plt.hist(prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=breakeven_price, color='red', linestyle='--', linewidth=2, label=f'Break-even: €{breakeven_price:.1f}/MWh')
    plt.xlabel('Electricity Price (€/MWh)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Rolling average prices for different horizons
    plt.subplot(3, 2, 3)
    for horizon in [24, 48, 168]:
        rolling_avg = []
        for i in range(len(prices) - horizon + 1):
            rolling_avg.append(np.mean(prices[i:i+horizon]))
        
        plt.plot(range(len(rolling_avg)), rolling_avg, alpha=0.7, label=f'{horizon}h rolling avg')
    
    plt.axhline(y=breakeven_price, color='red', linestyle='--', linewidth=2, label=f'Break-even')
    plt.xlabel('Starting Hour')
    plt.ylabel('Average Price (€/MWh)')
    plt.title('Rolling Average Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Forecast horizon impact on favorable periods
    plt.subplot(3, 2, 4)
    favorable_percentages = []
    
    for horizon in range(6, 169, 6):  # Every 6 hours from 6h to 1 week
        favorable_periods = 0
        total_periods = len(prices) - horizon + 1
        
        for start in range(total_periods):
            window_prices = prices[start:start + horizon]
            avg_price = np.mean(window_prices)
            
            if avg_price < breakeven_price:
                favorable_periods += 1
        
        favorable_pct = 100 * favorable_periods / total_periods if total_periods > 0 else 0
        favorable_percentages.append(favorable_pct)
    
    horizon_range = list(range(6, 169, 6))
    plt.plot(horizon_range, favorable_percentages, 'o-', color='green')
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('% Periods with Favorable Avg Price')
    plt.title('Impact of Forecast Horizon')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Simple decision analysis for 24h horizon
    plt.subplot(3, 2, 5)
    decisions_24h = []
    
    for start in range(len(prices) - 24 + 1):
        window_prices = prices[start:start + 24]
        avg_price = np.mean(window_prices)
        
        # Simple decision: run 100% if avg price < break-even
        decision = 100 if avg_price < breakeven_price else 10
        decisions_24h.append(decision)
    
    plt.plot(range(len(decisions_24h)), decisions_24h, alpha=0.7, color='orange')
    plt.xlabel('Starting Hour')
    plt.ylabel('Capacity Decision (%)')
    plt.title('Simple 24h Forecast Decisions')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Economic impact comparison
    plt.subplot(3, 2, 6)
    
    # Calculate simple economics for different strategies
    always_100_profit = len(prices) * (params["M_100"] * params["Price_Methanol"] - 
                                      np.mean(prices) * params["P_100"] - 
                                      params["C_100"] * params["Price_CO2"] - 
                                      params["OPEX_Variable"])
    
    always_10_profit = len(prices) * (params["M_10"] * params["Price_Methanol"] - 
                                     np.mean(prices) * params["P_10"] - 
                                     params["C_10"] * params["Price_CO2"] - 
                                     params["OPEX_Variable"])
    
    # Simple 24h strategy profit
    simple_24h_profit = 0
    for start in range(len(prices) - 24 + 1):
        window_prices = prices[start:start + 24]
        avg_price = np.mean(window_prices)
        
        if avg_price < breakeven_price:
            # Run at 100%
            simple_24h_profit += (params["M_100"] * params["Price_Methanol"] - 
                                 prices[start] * params["P_100"] - 
                                 params["C_100"] * params["Price_CO2"] - 
                                 params["OPEX_Variable"])
        else:
            # Run at 10%
            simple_24h_profit += (params["M_10"] * params["Price_Methanol"] - 
                                 prices[start] * params["P_10"] - 
                                 params["C_10"] * params["Price_CO2"] - 
                                 params["OPEX_Variable"])
    
    strategies = ['Always 100%', 'Always 10%', 'Simple 24h MPC']
    profits = [always_100_profit, always_10_profit, simple_24h_profit]
    colors = ['blue', 'red', 'orange']
    
    plt.bar(strategies, profits, color=colors, alpha=0.7)
    plt.ylabel('Annual Profit (€)')
    plt.title('Economic Comparison')
    plt.xticks(rotation=45)
    
    for i, profit in enumerate(profits):
        plt.text(i, profit + max(profits)*0.01, f'€{profit:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/forecast_horizon_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nForecast horizon analysis saved to plots/forecast_horizon_analysis.png")


if __name__ == "__main__":
    analyze_forecast_horizons()
