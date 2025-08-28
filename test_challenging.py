"""
Create a more challenging optimization problem to test solve times.
"""

import pandas as pd
import numpy as np
from model import build_model
import pyomo.environ as pyo


def create_challenging_prices():
    """Create more volatile, challenging price data."""
    np.random.seed(123)  # Different seed for more challenging pattern
    hours = 8760
    t = np.arange(hours)
    
    # Much more volatile prices with frequent spikes
    base_price = 50.0
    seasonal = 20 * np.sin(2 * np.pi * t / (24 * 365))
    daily = 30 * np.sin(2 * np.pi * t / 24)
    
    # Add random price spikes (much more volatile)
    spikes = np.random.exponential(0.1, hours) * 200 * np.random.choice([1, -1], hours)
    
    # High frequency variations
    high_freq = 15 * np.sin(2 * np.pi * t / 6) * np.random.normal(1, 0.3, hours)
    
    prices = base_price + seasonal + daily + spikes + high_freq
    prices = np.maximum(prices, 1.0)  # Minimum 1 €/MWh
    
    return prices.tolist()


def create_challenging_params():
    """Create more challenging parameters."""
    return {
        "P_100": 100.0, "M_100": 8.5, "C_100": 6.2,
        "P_10": 15.0, "M_10": 0.85, "C_10": 0.62,
        
        # Much higher ramp penalties (makes ramping very expensive)
        "Production_Loss_Up": 20.0,     # Higher penalty
        "Energy_Penalty_Up": 50.0,      # Higher penalty
        "Production_Loss_Down": 15.0,   # Higher penalty  
        "Energy_Penalty_Down": 30.0,    # Higher penalty
        
        "Price_Methanol": 750.0, "Price_CO2": 50.0,
        "Annualized_CAPEX": 8.5e6, "OPEX_Fixed": 2.5e6, "OPEX_Variable": 150.0,
        
        # Longer stabilization time (more constraints)
        "T_stab": 8  # Much longer stabilization requirement
    }


def test_challenging_model():
    """Test with a more challenging model."""
    print("Creating challenging optimization problem...")
    
    # Create challenging data
    prices = create_challenging_prices()
    params = create_challenging_params()
    
    print(f"Price range: {min(prices):.1f} - {max(prices):.1f} €/MWh")
    print(f"Price volatility (std dev): {np.std(prices):.1f} €/MWh")
    print(f"Stabilization time: {params['T_stab']} hours")
    print(f"High ramp penalties: Up={params['Production_Loss_Up']} tons, Down={params['Production_Loss_Down']} tons")
    
    # Build and solve
    model = build_model(prices, params)
    
    solver = pyo.SolverFactory('gurobi')
    print("\nSolving challenging model...")
    results = solver.solve(model, tee=True)
    
    print(f"\nSolver status: {results.solver.termination_condition}")


if __name__ == "__main__":
    test_challenging_model()
