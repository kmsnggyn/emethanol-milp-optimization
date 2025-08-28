"""
Pyomo model for e-methanol plant optimization.

This module contains the core MILP model for determining the optimal hourly 
operating schedule of a green e-methanol plant over one year (8760 hours).
The model maximizes annual profit by deciding when to operate at full load (100%) 
versus minimum turndown load (10%), considering ramping penalties and stabilization constraints.
"""

import pyomo.environ as pyo


def build_model(prices, params):
    """
    Build the Pyomo MILP model for e-methanol plant optimization.
    
    Args:
        prices (list): Hourly electricity prices [€/MWh] for 8760 hours
        params (dict): Model parameters including technical and economic data
    
    Returns:
        pyo.ConcreteModel: Configured Pyomo model ready for solving
    """
    
    # Initialize the model
    model = pyo.ConcreteModel()
    
    # =====================================================
    # SETS
    # =====================================================
    
    # Time horizon: hours 0 to 8759 (8760 hours total)
    model.T = pyo.RangeSet(0, 8759)
    
    # =====================================================
    # PARAMETERS
    # =====================================================
    
    # Electricity prices for each hour
    model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(prices)), 
                                doc="Electricity price [€/MWh]")
    
    # =====================================================
    # DECISION VARIABLES
    # =====================================================
    
    # Primary operating decision: 1 = 100% load, 0 = 10% load
    model.x = pyo.Var(model.T, domain=pyo.Binary, 
                     doc="Operating state (1=100% load, 0=10% load)")
    
    # Ramp event indicators
    model.y_up = pyo.Var(model.T, domain=pyo.Binary, 
                        doc="Start of ramp-up event (0→1)")
    
    model.y_down = pyo.Var(model.T, domain=pyo.Binary, 
                          doc="Start of ramp-down event (1→0)")
    
    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================
    
    def objective_rule(model):
        """
        Maximize annual profit = Revenue - Costs
        
        Revenue: Methanol sales
        Costs: Electricity, CO2, Variable OPEX, Ramp penalties, Fixed costs
        """
        
        # Revenue from methanol production
        methanol_revenue = sum(
            # Revenue at 100% load
            model.x[t] * params["M_100"] * params["Price_Methanol"] + 
            # Revenue at 10% load  
            (1 - model.x[t]) * params["M_10"] * params["Price_Methanol"]
            for t in model.T
        )
        
        # Electricity costs
        electricity_cost = sum(
            # Cost at 100% load
            model.x[t] * params["P_100"] * model.price_elec[t] + 
            # Cost at 10% load
            (1 - model.x[t]) * params["P_10"] * model.price_elec[t]
            for t in model.T
        )
        
        # CO2 costs
        co2_cost = sum(
            # Cost at 100% load
            model.x[t] * params["C_100"] * params["Price_CO2"] + 
            # Cost at 10% load
            (1 - model.x[t]) * params["C_10"] * params["Price_CO2"]
            for t in model.T
        )
        
        # Variable OPEX (applies when plant is operating at any level)
        variable_opex = sum(params["OPEX_Variable"] for t in model.T)
        
        # Ramp-up penalties (production loss + energy penalty costs)
        ramp_up_cost = sum(
            model.y_up[t] * (
                params["Production_Loss_Up"] * params["Price_Methanol"] +
                params["Energy_Penalty_Up"] * model.price_elec[t]
            )
            for t in model.T
        )
        
        # Ramp-down penalties  
        ramp_down_cost = sum(
            model.y_down[t] * (
                params["Production_Loss_Down"] * params["Price_Methanol"] +
                params["Energy_Penalty_Down"] * model.price_elec[t]
            )
            for t in model.T
        )
        
        # Total annual profit
        annual_profit = (
            methanol_revenue 
            - electricity_cost 
            - co2_cost 
            - variable_opex
            - ramp_up_cost 
            - ramp_down_cost
            - params["Annualized_CAPEX"]
            - params["OPEX_Fixed"]
        )
        
        return annual_profit
    
    # Set objective to maximize profit
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # =====================================================
    # CONSTRAINTS
    # =====================================================
    
    def ramp_up_logic_rule(model, t):
        """
        Ramp-up logic: y_up[t] = 1 when x[t]=1 and x[t-1]=0
        For t > 0: y_up[t] >= x[t] - x[t-1]
        """
        if t == 0:
            # For first hour, assume plant starts at 10% load
            return model.y_up[t] >= model.x[t]
        else:
            return model.y_up[t] >= model.x[t] - model.x[t-1]
    
    model.ramp_up_logic = pyo.Constraint(model.T, rule=ramp_up_logic_rule)
    
    def ramp_down_logic_rule(model, t):
        """
        Ramp-down logic: y_down[t] = 1 when x[t]=0 and x[t-1]=1  
        For t > 0: y_down[t] >= x[t-1] - x[t]
        """
        if t == 0:
            # For first hour, no ramp-down possible
            return model.y_down[t] == 0
        else:
            return model.y_down[t] >= model.x[t-1] - model.x[t]
    
    model.ramp_down_logic = pyo.Constraint(model.T, rule=ramp_down_logic_rule)
    
    def stabilization_rule(model, t):
        """
        Stabilization constraint: After any ramp event, the plant must stay 
        in that state for at least T_stab hours.
        
        Within any window of (T_stab + 1) consecutive hours, there can be 
        at most one ramp event (either up or down).
        """
        T_stab = params["T_stab"]
        
        # Only apply constraint if there are enough hours remaining
        if t + T_stab < len(model.T):
            return sum(model.y_up[tau] + model.y_down[tau] 
                      for tau in range(t, t + T_stab + 1)) <= 1
        else:
            # For the last few hours, apply constraint for remaining hours
            return sum(model.y_up[tau] + model.y_down[tau] 
                      for tau in range(t, len(model.T))) <= 1
    
    model.stabilization = pyo.Constraint(model.T, rule=stabilization_rule)
    
    return model
