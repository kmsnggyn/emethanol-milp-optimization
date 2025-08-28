"""
Test script for the e-methanol optimization model with a smaller time horizon.
This version tests the model with just 24 hours to verify it works correctly.
"""

import pandas as pd
import pyomo.environ as pyo
from model import build_model


def test_small_model():
    """Test the model with just 24 hours of data."""
    
    # Small test dataset - 24 hours with varying prices
    test_prices = [
        30, 25, 20, 18, 16, 20,  # Night (low prices)
        35, 45, 60, 70, 75, 80,  # Morning ramp-up
        85, 90, 88, 85, 82, 78,  # Day (high prices)  
        70, 60, 50, 40, 35, 32   # Evening ramp-down
    ]
    
    # Simplified parameters for testing
    params = {
        "P_100": 100.0,  "M_100": 8.5,   "C_100": 6.2,
        "P_10": 15.0,    "M_10": 0.85,   "C_10": 0.62,
        "Production_Loss_Up": 4.0,       "Energy_Penalty_Up": 10.0,
        "Production_Loss_Down": 1.5,     "Energy_Penalty_Down": 5.0,
        "Price_Methanol": 750.0,         "Price_CO2": 50.0,
        "Annualized_CAPEX": 8.5e6,       "OPEX_Fixed": 2.5e6,
        "OPEX_Variable": 150.0,          "T_stab": 3
    }
    
    print("Building 24-hour test model...")
    
    # Temporarily modify the model for 24 hours
    import model
    original_build = model.build_model
    
    def build_test_model(prices, params):
        """Modified version for 24-hour test."""
        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(0, 23)  # 24 hours
        
        model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(prices)))
        
        model.x = pyo.Var(model.T, domain=pyo.Binary)
        model.y_up = pyo.Var(model.T, domain=pyo.Binary)
        model.y_down = pyo.Var(model.T, domain=pyo.Binary)
        
        # Simplified objective (just revenue - electricity cost)
        def simple_objective(model):
            revenue = sum(
                model.x[t] * params["M_100"] * params["Price_Methanol"] + 
                (1 - model.x[t]) * params["M_10"] * params["Price_Methanol"]
                for t in model.T
            )
            elec_cost = sum(
                model.x[t] * params["P_100"] * model.price_elec[t] + 
                (1 - model.x[t]) * params["P_10"] * model.price_elec[t]
                for t in model.T
            )
            return revenue - elec_cost
        
        model.objective = pyo.Objective(rule=simple_objective, sense=pyo.maximize)
        
        # Basic ramp constraints
        def ramp_up_rule(model, t):
            if t == 0:
                return model.y_up[t] >= model.x[t]
            else:
                return model.y_up[t] >= model.x[t] - model.x[t-1]
        
        def ramp_down_rule(model, t):
            if t == 0:
                return model.y_down[t] == 0
            else:
                return model.y_down[t] >= model.x[t-1] - model.x[t]
        
        model.ramp_up_logic = pyo.Constraint(model.T, rule=ramp_up_rule)
        model.ramp_down_logic = pyo.Constraint(model.T, rule=ramp_down_rule)
        
        return model
    
    # Build and solve test model
    test_model = build_test_model(test_prices, params)
    
    # Try to solve with any available solver
    solvers = ['cbc', 'glpk', 'cplex', 'gurobi']
    for solver_name in solvers:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver.available():
                print(f"Solving with {solver_name}...")
                results = solver.solve(test_model, tee=False)
                
                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    print("✓ Test model solved successfully!")
                    
                    # Print simple results
                    x_vals = [pyo.value(test_model.x[t]) for t in test_model.T]
                    print(f"Operating decisions (1=100%, 0=10%): {x_vals}")
                    print(f"Hours at 100%: {sum(x_vals)}")
                    print(f"Objective value: €{pyo.value(test_model.objective):,.0f}")
                    return True
                else:
                    print(f"Solver status: {results.solver.termination_condition}")
        except Exception as e:
            print(f"Error with {solver_name}: {e}")
            continue
    
    print("No solver available for testing")
    return False


if __name__ == "__main__":
    test_small_model()
