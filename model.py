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
    """Get realistic 2019 parameters for e-methanol optimization with shutdown capability."""
    return {
        # Technical parameters (based on 300 kmol H2/hr + 100 kmol CO2/hr feed)
        "P_100": 50.0,   # Power consumption at 100% [MW] - electrolysis + compressors only
        "M_100": 2.56,   # Methanol production at 100% [ton/hr] - assuming 80% conversion efficiency
        "C_100": 100.0,  # CO2 consumption at 100% [kmol/hr] - matches your feed rate
        
        # Shutdown state parameters
        "P_shutdown": 0.0,  # No power consumption during shutdown [MW]
        "M_shutdown": 0.0,  # No production during shutdown [ton/hr]
        "C_shutdown": 0.0,  # No CO2 consumption during shutdown [ton/hr]
        
        # Startup transition parameters (6 hours: 0% → 100%)
        "startup_duration": 6,  # Hours needed for startup
        "P_startup": 25.0,      # Average power during startup [MW] (50% of full load)
        "M_startup": 0.0,       # No production during startup [ton/hr]
        "C_startup": 0.0,       # No CO2 consumption during startup [ton/hr]
        "OPEX_startup": 150.0,  # Higher OPEX during startup [€/hour]
        
        # Shutdown transition parameters (6 hours: 100% → 0%)
        "shutdown_duration": 6,  # Hours needed for shutdown
        "P_shutdown_trans": 10.0,  # Reduced power during shutdown transition [MW] (20% of full load)
        "M_shutdown_trans": 0.0,   # No production during shutdown transition [ton/hr]
        "C_shutdown_trans": 0.0,   # No CO2 consumption during shutdown transition [ton/hr]
        "OPEX_shutdown": 120.0,    # OPEX during shutdown transition [€/hour]
        
        # Economic parameters (2019 market conditions with green methanol premium)
        "Price_Methanol": 800.0,      # €/ton (green methanol premium pricing)
        "Price_CO2": 1.078,           # €/kmol (24.5 €/ton ÷ 44.01 kg/kmol × 1000 kg/ton)
        "Annualized_CAPEX": 5.01e6,   # €/year (based on 22,426 ton/year capacity)
        "OPEX_Fixed": 1.35e6,         # €/year (4% of CAPEX)
        "OPEX_Variable": 85.0,        # €/hour at 100% operation
        "OPEX_shutdown_state": 25.0,  # €/hour during shutdown state (minimal maintenance)
        
        # Operational constraints
        "T_stab": 4  # Minimum stabilization hours after reaching operational state
    }


def load_data():
    """Load 2019 electricity price data."""
    data_file = 'electricity_data/elspot_prices_2019.xlsx'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading 2019 electricity data from {data_file}...")
    df = pd.read_excel(data_file)
    
    # Use SE3 (Swedish zone 3) prices, skip first row (metadata)
    price_series = df['SE3'].iloc[1:]
    prices = price_series.dropna().tolist()
    prices = [float(p) for p in prices if isinstance(p, (int, float)) and not pd.isna(p) and p > 0]
    
    print(f"  elspot_prices_2019.xlsx: {len(prices)} prices from column 'SE3'")
    print(f"Total loaded: {len(prices)} hours of 2019 electricity price data")
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
    Build the Pyomo MILP model for e-methanol plant optimization.
    
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
    
    # Operating state variables (mutually exclusive)
    model.x_run = pyo.Var(model.T, domain=pyo.Binary, 
                         doc="Running at 100% load")
    
    model.x_shutdown = pyo.Var(model.T, domain=pyo.Binary, 
                              doc="Complete shutdown (0% load)")
    
    # Transition state variables
    model.x_startup = pyo.Var(model.T, domain=pyo.Binary, 
                             doc="In startup transition")
    
    model.x_shutdown_trans = pyo.Var(model.T, domain=pyo.Binary, 
                                    doc="In shutdown transition")
    
    # Transition event indicators (start of transition sequences)
    model.y_start_startup = pyo.Var(model.T, domain=pyo.Binary, 
                                   doc="Start of startup sequence")
    
    model.y_start_shutdown = pyo.Var(model.T, domain=pyo.Binary, 
                                    doc="Start of shutdown sequence")
    
    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================
    
    def objective_rule(model):
        """
        Maximize annual profit = Revenue - Costs
        
        States:
        - Running (100%): Full production and consumption
        - Shutdown: No production/consumption, minimal OPEX
        - Startup transition (6h): Power consumption, no production, higher OPEX
        - Shutdown transition (6h): Reduced power, no production, moderate OPEX
        """
        
        # Revenue from methanol production (only during running state)
        methanol_revenue = sum(
            model.x_run[t] * params["M_100"] * params["Price_Methanol"]
            for t in model.T
        )
        
        # Electricity costs
        electricity_cost = sum(
            model.price_elec[t] * (
                model.x_run[t] * params["P_100"] +
                model.x_startup[t] * params["P_startup"] +
                model.x_shutdown_trans[t] * params["P_shutdown_trans"] +
                model.x_shutdown[t] * params["P_shutdown"]
            )
            for t in model.T
        )
        
        # CO2 costs (only during running state)
        co2_cost = sum(
            model.x_run[t] * params["C_100"] * params["Price_CO2"]
            for t in model.T
        )
        
        # Variable OPEX
        variable_opex = sum(
            model.x_run[t] * params["OPEX_Variable"] +
            model.x_startup[t] * params["OPEX_startup"] +
            model.x_shutdown_trans[t] * params["OPEX_shutdown"] +
            model.x_shutdown[t] * params["OPEX_shutdown_state"]
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
        )
        
        return annual_profit
    
    # Set objective to maximize profit
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # =====================================================
    # CONSTRAINTS
    # =====================================================
    
    def state_exclusivity_rule(model, t):
        """
        Exactly one state must be active at any time:
        running OR shutdown OR startup transition OR shutdown transition
        """
        return (model.x_run[t] + model.x_shutdown[t] + 
                model.x_startup[t] + model.x_shutdown_trans[t]) == 1
    
    model.state_exclusivity = pyo.Constraint(model.T, rule=state_exclusivity_rule)
    
    def startup_sequence_rule(model, t):
        """
        Startup sequence logic: When startup begins, it must continue for 6 hours
        """
        startup_duration = params["startup_duration"]
        
        if t == 0:
            # Initial state: assume plant starts shutdown
            return model.x_shutdown[t] == 1
        
        # If startup begins at time t, next (startup_duration-1) hours must be startup
        if t + startup_duration - 1 < len(model.T):
            return (startup_duration - 1) * model.y_start_startup[t] <= sum(
                model.x_startup[tau] for tau in range(t + 1, t + startup_duration)
            )
        else:
            return pyo.Constraint.Skip
    
    model.startup_sequence = pyo.Constraint(model.T, rule=startup_sequence_rule)
    
    def shutdown_sequence_rule(model, t):
        """
        Shutdown sequence logic: When shutdown begins, it must continue for 6 hours
        """
        shutdown_duration = params["shutdown_duration"]
        
        # If shutdown transition begins at time t, next (shutdown_duration-1) hours must be shutdown transition
        if t + shutdown_duration - 1 < len(model.T):
            return (shutdown_duration - 1) * model.y_start_shutdown[t] <= sum(
                model.x_shutdown_trans[tau] for tau in range(t + 1, t + shutdown_duration)
            )
        else:
            return pyo.Constraint.Skip
    
    model.shutdown_sequence = pyo.Constraint(model.T, rule=shutdown_sequence_rule)
    
    def startup_trigger_rule(model, t):
        """
        Startup can only begin from shutdown state
        """
        if t == 0:
            return model.y_start_startup[t] == 0  # Can't start transition at t=0
        else:
            return model.y_start_startup[t] <= model.x_shutdown[t-1]
    
    model.startup_trigger = pyo.Constraint(model.T, rule=startup_trigger_rule)
    
    def shutdown_trigger_rule(model, t):
        """
        Shutdown transition can only begin from running state
        """
        if t == 0:
            return model.y_start_shutdown[t] == 0  # Can't start transition at t=0
        else:
            return model.y_start_shutdown[t] <= model.x_run[t-1]
    
    model.shutdown_trigger = pyo.Constraint(model.T, rule=shutdown_trigger_rule)
    
    def startup_completion_rule(model, t):
        """
        After startup sequence completes, plant must be in running state
        """
        startup_duration = params["startup_duration"]
        
        if t >= startup_duration:
            # If startup started at t-startup_duration, then at t we should be running
            return model.x_run[t] >= model.y_start_startup[t - startup_duration]
        else:
            return pyo.Constraint.Skip
    
    model.startup_completion = pyo.Constraint(model.T, rule=startup_completion_rule)
    
    def shutdown_completion_rule(model, t):
        """
        After shutdown sequence completes, plant must be in shutdown state
        """
        shutdown_duration = params["shutdown_duration"]
        
        if t >= shutdown_duration:
            # If shutdown started at t-shutdown_duration, then at t we should be shutdown
            return model.x_shutdown[t] >= model.y_start_shutdown[t - shutdown_duration]
        else:
            return pyo.Constraint.Skip
    
    model.shutdown_completion = pyo.Constraint(model.T, rule=shutdown_completion_rule)
    
    def transition_start_logic_rule(model, t):
        """
        Link transition start events to transition states
        """
        return (model.x_startup[t] >= model.y_start_startup[t] +
                model.x_shutdown_trans[t] >= model.y_start_shutdown[t])
    
    model.transition_start_logic = pyo.Constraint(model.T, rule=transition_start_logic_rule)
    
    def stabilization_rule(model, t):
        """
        After reaching operational state (running or shutdown), 
        stay in that state for at least T_stab hours before allowing transitions
        """
        T_stab = params["T_stab"]
        
        # Only apply if there are enough hours remaining
        if t + T_stab < len(model.T):
            return sum(model.y_start_startup[tau] + model.y_start_shutdown[tau] 
                      for tau in range(t, t + T_stab + 1)) <= 1
        else:
            return sum(model.y_start_startup[tau] + model.y_start_shutdown[tau] 
                      for tau in range(t, len(model.T))) <= 1
    
    model.stabilization = pyo.Constraint(model.T, rule=stabilization_rule)
    
    return model
