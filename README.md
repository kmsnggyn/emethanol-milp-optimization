# E-Methanol Plant Optimization Model

A Mixed-Integer Linear Programming (MILP) model for optimal hourly operating schedule of a green e-methanol plant using Pyomo.

## Project Overview

This project implements a techno-economic optimization model to determine the optimal hourly operating schedule for a green e-methanol plant over one year (8760 hours). The model maximizes annual profit by deciding when to run at full load (100%) versus minimum turndown load (10%), based on volatile hourly electricity prices.

### Key Features

- **Perfect foresight MILP optimization** using Pyomo
- **Binary operating decisions** (100% vs 10% load)
- **Ramp penalty modeling** with production losses and energy penalties
- **Stabilization constraints** requiring minimum time in each state
- **Comprehensive economic modeling** including CAPEX, OPEX, and variable costs

## Project Structure

```
├── model.py              # Core Pyomo MILP model definition
├── main.py               # Main executable script
├── generate_prices.py    # Script to generate dummy electricity price data
├── test_model.py         # 24-hour test version for validation
├── check_solvers.py      # Utility to check available solvers
├── data/
│   └── dummy_prices.csv  # 8760 hours of realistic electricity prices
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Description

### Decision Variables
- `x[t]`: Binary variable (1 = 100% load, 0 = 10% load) for each hour t
- `y_up[t]`: Binary indicator for start of ramp-up events
- `y_down[t]`: Binary indicator for start of ramp-down events

### Objective Function
Maximize annual profit = Revenue - Costs
- **Revenue**: Methanol sales at current production rates
- **Costs**: Electricity, CO₂, variable OPEX, ramp penalties, fixed costs

### Constraints
1. **Ramp Logic**: Links operating states with ramp indicators
2. **Stabilization**: Plant must remain in state for minimum time after ramping

### Model Parameters (Dummy Values)

```python
params = {
    # Plant Technical Parameters (100 MW electrolyzer capacity)
    "P_100": 100.0,    # Power at 100% load [MW]
    "M_100": 8.5,      # Methanol production at 100% [ton/hr]
    "C_100": 6.2,      # CO2 consumption at 100% [ton/hr]
    
    "P_10": 15.0,      # Power at 10% load [MW]  
    "M_10": 0.85,      # Methanol production at 10% [ton/hr]
    "C_10": 0.62,      # CO2 consumption at 10% [ton/hr]

    # Ramp Penalties
    "Production_Loss_Up": 4.0,     # Methanol loss during ramp-up [ton]
    "Energy_Penalty_Up": 10.0,     # Extra energy during ramp-up [MWh]
    "Production_Loss_Down": 1.5,   # Methanol loss during ramp-down [ton]
    "Energy_Penalty_Down": 5.0,    # Extra energy during ramp-down [MWh]

    # Economic Parameters
    "Price_Methanol": 750.0,       # €/ton
    "Price_CO2": 50.0,             # €/ton
    "Annualized_CAPEX": 8.5e6,     # €/year
    "OPEX_Fixed": 2.5e6,           # €/year
    "OPEX_Variable": 150.0,        # €/hr

    # Operational Constraints
    "T_stab": 3  # Minimum hours in state after ramping
}
```

## Dependencies

- Python 3.8+
- Pyomo ≥6.6.0
- pandas ≥1.5.0  
- numpy ≥1.24.0
- Compatible MILP solver (GLPK, CBC, CPLEX, or Gurobi)

## Installation

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install a MILP solver** (required to solve the optimization):
   
   **Option 1 - GLPK (recommended for academic use)**:
   ```bash
   conda install -c conda-forge glpk
   ```
   
   **Option 2 - CBC**:
   ```bash
   conda install -c conda-forge coincbc  
   ```
   
   **Option 3 - Download GLPK for Windows**:
   Download from http://winglpk.sourceforge.net/

## Usage

1. **Generate dummy price data**:
   ```bash
   python generate_prices.py
   ```

2. **Run the optimization**:
   ```bash
   python main.py
   ```

3. **Test with 24-hour model**:
   ```bash
   python test_model.py
   ```

## Expected Output

When properly configured with a solver, the model outputs:
- Total annual profit
- Operational metrics (capacity factor, hours at each load)
- Production metrics (total methanol output)
- Ramp events statistics  
- Energy consumption breakdown
- Weighted average electricity price

## Model Validation

The model has been designed with the following validation features:
- **Logical constraints** ensure physically feasible solutions
- **Stabilization constraints** prevent unrealistic rapid switching
- **Economic penalties** for ramping reflect real operational costs
- **Test version** with 24 hours for quick validation

## For Your Thesis

This model provides the core framework for your Master's thesis. To adapt it:

1. **Replace dummy parameters** with your Aspen simulation data
2. **Adjust time horizon** if needed (currently 8760 hours)
3. **Modify constraints** based on your specific plant requirements
4. **Add sensitivity analysis** for key parameters
5. **Include uncertainty modeling** if desired

## Solver Requirements

This is a large-scale MILP with:
- 26,280 binary variables (3 × 8760 hours)
- ~26,280 constraints
- Solution time: Minutes to hours depending on solver and hardware

Commercial solvers (CPLEX, Gurobi) typically solve faster than open-source ones (GLPK, CBC).

## Author

Master's thesis project for green e-methanol plant optimization using Pyomo and MILP modeling.
