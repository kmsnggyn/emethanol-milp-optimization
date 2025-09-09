"""
Pyomo model for e-methanol plant optimization.

This module contains the core MILP model for determining the optimal hourly 
operating schedule of a green e-methanol plant over one year (8760 hours).
The model maximizes annual profit by deciding when to operate at full load (100%) 
versus minimum turndown load (10%), considering ramping penalties and stabilization constraints.
"""

import pyomo.environ as pyo
import pandas as pd
import os


def get_parameters():
    """Get realistic 2019 parameters for e-methanol optimization with binary operation (100% or 10%)."""
    return {
        # Technical parameters (based on 2023-2024 market data: 52 kWh/kg H2, 77% LHV efficiency)
        "P_electrolysis_100": 31.4,  # Electrolysis power at 100% [MW] - H2 production (604.8 kg/hr × 52 kWh/kg ÷ 1000)
        "P_methanol_100": 1.0,  # Methanol plant power at 100% [MW] - compressors and auxiliaries
        "P_100": 31.4 + 1.0,  # Total power consumption at 100% [MW] - electrolysis + methanol plant
        "M_100": 2689.14,       # Methanol production at 100% [kg/hr] - actual production rate
        "C_100": 4401.0,         # CO2 consumption at 100% [kg/hr] - matches your feed rate
        
        # Minimum turndown parameters (10% of full capacity)
        "P_electrolysis_10": 3.14,  # Electrolysis power at 10% [MW] - 10% of full capacity
        "P_methanol_10": 0.2,  # Methanol plant power at 10% [MW] - 20% due to lower efficiency
        "P_10": 3.14 + 0.2,  # Total power consumption at 10% [MW] - electrolysis + methanol plant
        "M_10": 268.91,         # Methanol production at 10% [kg/hr] - 10% of full production
        "C_10": 440.1,          # CO2 consumption at 10% [kg/hr] - 10% of full consumption
        
        # Ramping penalties (simplified - no shutdown transitions)
        "Production_Loss_Up": 100.0,  # No production during ramp up [%]
        "Energy_Penalty_Up": 50.0,   # Additional energy penalty during ramp up [%]
        "Production_Loss_Down": 100.0, # No production during ramp down [%]
        "Energy_Penalty_Down": 50.0,  # Additional energy penalty during ramp down [%]
        
        # Economic parameters (2023-2024 market conditions with green methanol premium)
        "Price_Methanol": 0.8,        # €/kg (800 €/ton ÷ 1000 kg/ton)
        "Price_CO2": 0.05,             # €/kg (50 €/ton ÷ 1000 kg/ton)
        "Methanol_Plant_CAPEX": 226399.12, # €/year (methanol plant only)
        "Electrolysis_CAPEX": 5.9e6,  # €/year (31.4 MW × €1,666/kW × 7% discount rate, 20yr)
        "Annualized_CAPEX": 226399.12 + 5.9e6, # €/year (total system CAPEX)
        "OPEX_Fixed": 1.35e6,         # €/year (methanol plant fixed O&M)
        "OPEX_Electrolysis_Stack": 2.79e6, # €/year (stack replacement every 7.5 years)
        "OPEX_Variable": 85.0,        # €/hour at 100% operation
        "OPEX_10": 25.0,              # €/hour at 10% operation (higher due to lower efficiency)
        
        # Operational constraints
        "T_stab": 4  # Minimum stabilization hours after ramping
    }


def load_data():
    """Load 2023 electricity price data."""
    data_file = 'electricity_data/elspot_prices_2023.xlsx'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading 2023 electricity data from {data_file}...")
    df = pd.read_excel(data_file)
    
    # Use SE3 (Swedish zone 3) prices, skip first row (metadata)
    price_series = df['SE3'].iloc[1:]
    prices = price_series.dropna().tolist()
    prices = [float(p) for p in prices if isinstance(p, (int, float)) and not pd.isna(p) and p > 0]
    
    print(f"  elspot_prices_2023.xlsx: {len(prices)} prices from column 'SE3'")
    print(f"Total loaded: {len(prices)} hours of 2023 electricity price data")
    print(f"Price range: {min(prices):.2f} - {max(prices):.2f} €/MWh")
    
    return {'price': prices}


def solve_model(model):
    """Solve the optimization model using Gurobi or HiGHS."""
    
    # Try Gurobi first, then HiGHS
    for solver_name in ['gurobi', 'highs']:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver.available():
                # Solve with minimal output
                results = solver.solve(model, tee=False)
                
                # Check if solution is optimal
                if (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                    return results.solver.termination_condition
                else:
                    print(f"Solver failed with status: {results.solver.termination_condition}")
                    return results.solver.termination_condition
        except:
            continue
    
    print("Error: No compatible MILP solver found. Please install Gurobi or HiGHS.")
    return "no_solver"


def build_model(prices, params=None):
    """
    Build the Pyomo MILP model for e-methanol plant optimization with binary operation.
    
    Args:
        prices (list): Hourly electricity prices [€/MWh] for 8760 hours
        params (dict): Model parameters (optional, uses defaults if None)
    
    Returns:
        pyomo.ConcreteModel: The optimization model ready for solving
    """
    
    if params is None:
        params = get_parameters()
    
    # Initialize the model
    model = pyo.ConcreteModel()
    
    # =====================================================
    # SETS
    # =====================================================
    
    # Time horizon: hours 0 to len(prices)-1 (dynamic based on data)
    model.T = pyo.RangeSet(0, len(prices)-1)
    
    # =====================================================
    # PARAMETERS
    # =====================================================
    
    # Electricity prices for each hour
    model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(prices)), 
                                doc="Electricity price [€/MWh]")
    
    # =====================================================
    # DECISION VARIABLES
    # =====================================================
    
    # Operating state variables (mutually exclusive)
    model.x_100 = pyo.Var(model.T, domain=pyo.Binary, 
                         doc="Running at 100% load")
    
    model.x_10 = pyo.Var(model.T, domain=pyo.Binary, 
                        doc="Running at 10% load (minimum turndown)")
    
    # Ramping variables
    model.y_ramp_up = pyo.Var(model.T, domain=pyo.Binary, 
                             doc="Ramp up from 10% to 100%")
    
    model.y_ramp_down = pyo.Var(model.T, domain=pyo.Binary, 
                               doc="Ramp down from 100% to 10%")
    
    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================
    
    def objective_rule(model):
        """
        Maximize annual profit = Revenue - Costs
        
        States:
        - 100% load: Full production and consumption
        - 10% load: Minimum turndown production and consumption
        - Ramping: Additional penalties during transitions
        """
        
        # Revenue from methanol production (no production during ramps)
        methanol_revenue = sum(
            model.x_100[t] * params["M_100"] * params["Price_Methanol"] +
            model.x_10[t] * params["M_10"] * params["Price_Methanol"]
            # Note: y_ramp_up and y_ramp_down represent transition states with zero production
            for t in model.T
        )
        
        # Electricity costs
        electricity_cost = sum(
            model.price_elec[t] * (
                model.x_100[t] * params["P_100"] +
                model.x_10[t] * params["P_10"] +
                model.y_ramp_up[t] * params["P_100"] * (params["Energy_Penalty_Up"]/100) +
                model.y_ramp_down[t] * params["P_10"] * (params["Energy_Penalty_Down"]/100)
            )
            for t in model.T
        )
        
        # CO2 costs
        co2_cost = sum(
            model.x_100[t] * params["C_100"] * params["Price_CO2"] +
            model.x_10[t] * params["C_10"] * params["Price_CO2"]
            for t in model.T
        )
        
        # Variable OPEX (excluding electricity - calculated separately)
        variable_opex = sum(
            model.x_100[t] * params["OPEX_Variable"] +
            model.x_10[t] * params["OPEX_10"]
            for t in model.T
        )
        
        # Total annual profit
        annual_profit = (
            methanol_revenue 
            - electricity_cost 
            - co2_cost 
            - variable_opex
            - params["Annualized_CAPEX"]
            - params["OPEX_Fixed"]
            - params["OPEX_Electrolysis_Stack"]
        )
        
        return annual_profit
    
    # Set objective to maximize profit
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # =====================================================
    # CONSTRAINTS
    # =====================================================
    
    def state_exclusivity_rule(model, t):
        """
        Exactly one operating state must be active at any time:
        100% load OR 10% load (mutually exclusive)
        """
        return (model.x_100[t] + model.x_10[t]) == 1
    
    model.state_exclusivity = pyo.Constraint(model.T, rule=state_exclusivity_rule)
    
    def ramp_up_logic_rule(model, t):
        """
        Ramp up can only occur when transitioning from 10% to 100%
        """
        if t == 0:
            return model.y_ramp_up[t] == 0  # Can't ramp at t=0
        else:
            return model.y_ramp_up[t] <= model.x_10[t-1] * model.x_100[t]
    
    model.ramp_up_logic = pyo.Constraint(model.T, rule=ramp_up_logic_rule)
    
    def ramp_down_logic_rule(model, t):
        """
        Ramp down can only occur when transitioning from 100% to 10%
        """
        if t == 0:
            return model.y_ramp_down[t] == 0  # Can't ramp at t=0
        else:
            return model.y_ramp_down[t] <= model.x_100[t-1] * model.x_10[t]
    
    model.ramp_down_logic = pyo.Constraint(model.T, rule=ramp_down_logic_rule)
    
    def stabilization_rule(model, t):
        """
        After ramping, stay in the new state for at least T_stab hours
        """
        T_stab = params["T_stab"]
        
        # Only apply if there are enough hours remaining
        if t + T_stab < len(model.T):
            return sum(model.y_ramp_up[tau] + model.y_ramp_down[tau] 
                      for tau in range(t, t + T_stab + 1)) <= 1
        else:
            return sum(model.y_ramp_up[tau] + model.y_ramp_down[tau] 
                      for tau in range(t, len(model.T))) <= 1
    
    model.stabilization = pyo.Constraint(model.T, rule=stabilization_rule)
    
    def initial_state_rule(model, t):
        """
        Set initial state: assume plant starts at 10% load
        """
        if t == 0:
            return model.x_10[t] == 1
        else:
            return pyo.Constraint.Skip
    
    model.initial_state = pyo.Constraint(model.T, rule=initial_state_rule)
    
    return model
