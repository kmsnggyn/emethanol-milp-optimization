"""
Main script for running the e-methanol plant optimization model.

This script loads electricity price data, builds the Pyomo MILP model,
solves it using Gurobi or HiGHS, and prints comprehensive results including profit,
production metrics, and operational statistics.
"""

import pandas as pd
import pyomo.environ as pyo
from model import build_model


def load_data():
    """
    Load electricity price data and model parameters.
    
    Returns:
        tuple: (prices_list, params_dict)
    """
    
    # Load electricity price data
    try:
        price_df = pd.read_csv('data/dummy_prices.csv')
        prices = price_df['price_eur_per_mwh'].tolist()
        print(f"Loaded {len(prices)} hours of electricity price data")
        print(f"Price range: {min(prices):.2f} - {max(prices):.2f} €/MWh")
    except FileNotFoundError:
        print("Error: data/dummy_prices.csv not found. Run generate_prices.py first.")
        return None, None
    
    # Model parameters (dummy data for testing)
    params = {
        # == PLANT TECHNICAL PARAMETERS ==
        # Based on a nominal 100 MW electrolyzer input capacity plant
        "P_100": 100.0,  # Power consumption at 100% load [MW]
        "M_100": 8.5,    # Methanol production at 100% load [ton/hr]
        "C_100": 6.2,    # CO2 consumption at 100% load [ton/hr]
        
        "P_10": 15.0,     # Power consumption at 10% load [MW]
        "M_10": 0.85,     # Methanol production at 10% load [ton/hr]
        "C_10": 0.62,     # CO2 consumption at 10% load [ton/hr]

        # == DYNAMIC RAMP PENALTIES ==
        # These represent the total deviation from steady-state over the ramp event duration
        # Ramp-Up Event (e.g., takes 3 hours total)
        "Production_Loss_Up": 4.0,   # Total tons of methanol NOT produced vs. staying at 100%
        "Energy_Penalty_Up": 10.0,    # Extra MWh consumed vs. staying at 100%
        
        # Ramp-Down Event (e.g., takes 2 hours total)
        "Production_Loss_Down": 1.5, # Total tons of methanol NOT produced vs. staying at 10%
        "Energy_Penalty_Down": 5.0,   # Extra MWh consumed vs. staying at 10%

        # == ECONOMIC PARAMETERS ==
        "Price_Methanol": 750.0,     # €/ton
        "Price_CO2": 50.0,           # €/ton
        "Annualized_CAPEX": 8.5e6,   # €/year (e.g., from a €120M CAPEX, 25-year life, 8% discount)
        "OPEX_Fixed": 2.5e6,       # €/year (staff, maintenance, etc.)
        "OPEX_Variable": 150.0,      # Additional variable costs per hour of operation [€/hr]

        # == OPERATIONAL CONSTRAINTS ==
        "T_stab": 3  # Minimum hours plant must stay in a state after ramping
    }
    
    return prices, params


def solve_model(model):
    """
    Solve the optimization model using available solver (Gurobi or HiGHS).
    
    Args:
        model: Pyomo model instance
        
    Returns:
        str: Solver status
    """
    
    # Try different solvers in order of preference 
    # Gurobi first (fastest with academic license), then HiGHS as backup
    solver_names = ['gurobi', 'highs']
    solver = None
    solver_name = None
    
    for name in solver_names:
        try:
            test_solver = pyo.SolverFactory(name)
            if test_solver.available():
                solver = test_solver
                solver_name = name
                break
        except:
            continue
    
    if solver is None:
        print("Error: No compatible MILP solver found.")
        print("Please install one of: Gurobi or HiGHS")
        print("")
        print("Installation options:")
        print("1. For Gurobi: pip install gurobipy (fastest, requires academic license)")
        print("2. For HiGHS: pip install highspy (open-source, no license needed)")
        print("")
        print("Model structure created successfully but cannot solve without a solver.")
        return "no_solver"
    
    print(f"Starting optimization with {solver_name.upper() if solver_name else 'UNKNOWN'} solver...")
    print("This may take several minutes for 8760 variables and constraints...")
    
    # Solve the model
    results = solver.solve(model, tee=True)
    
    # Check solver status
    status = results.solver.termination_condition
    print(f"\nSolver finished with status: {status}")
    
    return status


def print_results(model, params, status):
    """
    Extract and print comprehensive results from the solved model.
    
    Args:
        model: Solved Pyomo model
        params: Model parameters dictionary  
        status: Solver termination status
    """
    
    if status == "no_solver":
        print("\n" + "="*60)
        print("MODEL STRUCTURE SUMMARY")
        print("="*60)
        print("✓ Model built successfully with:")
        print(f"  - {len(model.T)} time periods (hours)")
        print(f"  - {len(model.x)} binary operating variables")
        print(f"  - {len(model.y_up)} ramp-up indicator variables")
        print(f"  - {len(model.y_down)} ramp-down indicator variables")
        print(f"  - {len(list(model.component_objects(pyo.Constraint)))} constraint sets")
        print("\n✓ Objective function: Maximize annual profit")
        print("✓ Constraints: Ramp logic + Stabilization requirements")
        print("\nTo solve the model, please install a compatible solver (see instructions above).")
        print("="*60)
        return
    
    if status != pyo.TerminationCondition.optimal:
        print(f"Warning: Solution may not be optimal. Status: {status}")
        return
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    # Extract objective value (total annual profit)
    total_profit = pyo.value(model.objective)
    print(f"Total Annual Profit: €{total_profit:,.0f}")
    
    # Extract decision variables
    x_values = [pyo.value(model.x[t]) for t in model.T]
    y_up_values = [pyo.value(model.y_up[t]) for t in model.T]
    y_down_values = [pyo.value(model.y_down[t]) for t in model.T]
    
    # Calculate operational metrics
    hours_at_100 = sum(x_values)
    hours_at_10 = 8760 - hours_at_100
    capacity_factor = (hours_at_100 * 100 + hours_at_10 * 10) / (8760 * 100) * 100
    
    print(f"\nOPERATIONAL METRICS:")
    print(f"Hours at 100% load: {hours_at_100:,.0f} ({hours_at_100/8760*100:.1f}%)")
    print(f"Hours at 10% load: {hours_at_10:,.0f} ({hours_at_10/8760*100:.1f}%)")
    print(f"Overall capacity factor: {capacity_factor:.1f}%")
    
    # Calculate production metrics
    methanol_100 = hours_at_100 * params["M_100"]
    methanol_10 = hours_at_10 * params["M_10"]
    total_methanol = methanol_100 + methanol_10
    
    print(f"\nPRODUCTION METRICS:")
    print(f"Methanol from 100% operation: {methanol_100:,.0f} tons")
    print(f"Methanol from 10% operation: {methanol_10:,.0f} tons")
    print(f"Total methanol production: {total_methanol:,.0f} tons/year")
    
    # Calculate ramp events
    total_ramp_ups = sum(y_up_values)
    total_ramp_downs = sum(y_down_values)
    
    print(f"\nRAMP EVENTS:")
    print(f"Total ramp-up events: {total_ramp_ups:.0f}")
    print(f"Total ramp-down events: {total_ramp_downs:.0f}")
    print(f"Total state changes: {total_ramp_ups + total_ramp_downs:.0f}")
    
    # Calculate energy consumption
    energy_100 = hours_at_100 * params["P_100"]
    energy_10 = hours_at_10 * params["P_10"]
    total_energy = energy_100 + energy_10
    
    print(f"\nENERGY CONSUMPTION:")
    print(f"Energy at 100% load: {energy_100:,.0f} MWh")
    print(f"Energy at 10% load: {energy_10:,.0f} MWh") 
    print(f"Total energy consumption: {total_energy:,.0f} MWh/year")
    
    # Calculate average electricity price during operation
    prices = [pyo.value(model.price_elec[t]) for t in model.T]
    weighted_avg_price = sum(
        prices[t] * (x_values[t] * params["P_100"] + (1-x_values[t]) * params["P_10"])
        for t in range(8760)
    ) / total_energy
    
    print(f"Weighted average electricity price: {weighted_avg_price:.2f} €/MWh")
    
    print("\n" + "="*60)


def main():
    """
    Main execution function.
    """
    print("E-Methanol Plant Optimization Model")
    print("="*40)
    
    # Load data
    prices, params = load_data()
    if prices is None:
        return
    
    # Build model
    print("\nBuilding optimization model...")
    model = build_model(prices, params)
    print(f"Model built with {len(model.T)} time periods")
    print(f"Variables: {len(model.x)} binary operating decisions")
    print(f"           {len(model.y_up)} ramp-up indicators") 
    print(f"           {len(model.y_down)} ramp-down indicators")
    
    # Solve model
    status = solve_model(model)
    
    # Print results
    print_results(model, params, status)


if __name__ == "__main__":
    main()
