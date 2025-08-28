"""
Script to generate dummy electricity price data for the e-methanol optimization model.
Creates realistic hourly prices for one year (8760 hours) with daily and seasonal variations.
"""

import numpy as np
import pandas as pd

def generate_dummy_prices():
    """
    Generate realistic electricity price data for 8760 hours (1 year).
    
    Returns:
        pd.DataFrame: DataFrame with 'price_eur_per_mwh' column
    """
    # Number of hours in a year
    hours = 8760
    
    # Time array for the year
    t = np.arange(hours)
    
    # Base price
    base_price = 50.0  # €/MWh
    
    # Seasonal variation (higher in winter)
    seasonal = 15 * np.sin(2 * np.pi * t / (24 * 365) + np.pi)  # Phase shift for winter peak
    
    # Daily variation (higher during day, lower at night)
    daily = 20 * np.sin(2 * np.pi * t / 24 + np.pi/2)  # Phase shift for peak around noon
    
    # Weekly variation (slightly lower on weekends)
    weekly = -5 * np.sin(2 * np.pi * t / (24 * 7))
    
    # Random noise to simulate market volatility
    np.random.seed(42)  # For reproducible results
    noise = np.random.normal(0, 8, hours)
    
    # Combine all components
    prices = base_price + seasonal + daily + weekly + noise
    
    # Ensure no negative prices (set minimum at 5 €/MWh)
    prices = np.maximum(prices, 5.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price_eur_per_mwh': prices
    })
    
    return df

if __name__ == "__main__":
    # Generate the data
    price_data = generate_dummy_prices()
    
    # Save to CSV
    price_data.to_csv('data/dummy_prices.csv', index=False)
    
    print(f"Generated {len(price_data)} hours of price data")
    print(f"Price range: {price_data['price_eur_per_mwh'].min():.2f} - {price_data['price_eur_per_mwh'].max():.2f} €/MWh")
    print(f"Average price: {price_data['price_eur_per_mwh'].mean():.2f} €/MWh")
    print("Data saved to data/dummy_prices.csv")
