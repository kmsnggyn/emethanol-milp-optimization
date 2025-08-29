# E-Methanol Plant Optimization System

A comprehensive Mixed-Integer Linear Programming (MILP) optimization system for industrial-scale green e-methanol production, implementing quaternary operational states with intelligent shutdown capability and adaptive Model Predictive Control (MPC).

## System Architecture

### Core Components

**model.py** - MILP optimization engine
- Quaternary state system: running, shutdown, startup, shutdown_trans
- Industrial-scale parameters: 300 kmol/hr H2 + 100 kmol/hr CO2 feed
- Economic model: €800/ton methanol, 50 MW power consumption
- Transition constraints: 6-hour startup/shutdown sequences

**mpc.py** - Adaptive MPC controller
- PlantState management with transition tracking
- AdaptivePatternForecaster with real-time learning
- Rolling 24-hour optimization horizon
- Quaternary state transition logic

**run_optimization.py** - Analysis and demonstration suite
- Comprehensive economic analysis
- State distribution analysis
- Visualization generation
- Performance metrics calculation

## Mathematical Formulation

### State Variables
```
x_run[t] ∈ {0,1}         - Running at full capacity
x_shutdown[t] ∈ {0,1}    - Complete shutdown
x_startup[t] ∈ {0,1}     - Startup transition
x_shutdown_trans[t] ∈ {0,1} - Shutdown transition
```

### Operational Constraints
- State exclusivity: Σ states = 1 ∀t
- Transition sequences: 6-hour startup/shutdown
- Stabilization periods: 4-hour post-transition
- Power consumption: 50 MW (running), 15 MW (transitions), 2 MW (shutdown)

### Economic Objective
```
max Σ[t] (Revenue[t] - Electricity[t] - CO2[t] - OPEX[t])

Where:
Revenue[t] = Production[t] × €800/ton
Electricity[t] = Power[t] × Price_elec[t]
CO2[t] = CO2_consumption[t] × €50/ton
```

## Process Parameters

### Feed Specifications
- Hydrogen: 300 kmol/hr (renewable electrolytic)
- Carbon Dioxide: 100 kmol/hr (captured CO2)
- Methanol Production: 2.56 ton/hr (100% capacity)

### Economic Parameters
- Methanol Price: €800/ton
- Power Consumption: 50 MW at full capacity
- Annual Fixed Costs: €6.4M (CAPEX + Fixed OPEX)
- Breakeven Electricity: €22.6/MWh
- Plant Capacity: 22,426 ton/year

### Transition Dynamics
- Startup Duration: 6 hours (15 MW power)
- Shutdown Duration: 6 hours (8 MW power)
- Stabilization Period: 4 hours post-transition
- Shutdown Power: 2 MW (auxiliaries only)

## Adaptive Forecasting System

### Pattern Learning Components
- Hourly patterns: 24-hour price profiles
- Daily patterns: Day-of-week effects
- Weekly trends: Longer-term movements
- Exponential smoothing: α = 0.1

### Forecast Generation
```python
forecast_price = base_hourly × daily_factor + trend_adjustment + noise
```

### Real-time Adaptation
- Continuous pattern updates from observed data
- Minimum 48-hour data requirement for stable patterns
- Volatility-based uncertainty modeling

## Usage

### Basic Optimization
```python
from model import build_model, solve_model, load_data
from mpc import ForecastingMPC

# Load electricity price data
prices, params = load_data()

# Run MPC simulation
mpc = ForecastingMPC()
results = mpc.run_mpc_simulation(prices, params)
```

### Analysis and Visualization
```python
python run_optimization.py
```

## System Performance

### Operational Characteristics
- Capacity Factor: Variable (market-dependent)
- Production Efficiency: 2.56 ton/hour when running
- Transition Frequency: Optimized based on price volatility
- Power Efficiency: 19.5 MWh/ton methanol

### Economic Performance
- Break-even Operation: Electricity prices < €22.6/MWh
- Annual Fixed Costs: €6.4M
- Variable Costs: €665/ton (excluding electricity)
- Revenue Potential: €18M/year at full capacity

## Dependencies

```
pyomo>=6.0
gurobipy>=10.0  # or highs for open-source solver
pandas>=1.5
numpy>=1.20
matplotlib>=3.5
openpyxl>=3.0   # for Excel data loading
```

## Technical Implementation

### MILP Solver Integration
- Primary: Gurobi (commercial)
- Alternative: HiGHS (open-source)
- Constraint matrix: Sparse formulation for efficiency
- Solution time: < 1 second per MPC iteration

### Data Management
- Excel-based price data loading
- Pandas DataFrame integration
- Real-time parameter updates
- Pickle-based state persistence

### Performance Optimization
- Vectorized constraint generation
- Minimal model rebuilding
- Efficient state tracking
- Memory-optimized data structures

## Integration Points

### Aspen Plus Connectivity
Ready for integration with detailed process models:
- State variable export for dynamic simulation
- Heat integration calculations
- Detailed mass/energy balances
- Equipment sizing validation

### Industrial Control Systems
MPC framework designed for:
- DCS integration via OPC protocols
- Real-time price data feeds
- Automated decision implementation
- Safety system interlocks

## Mathematical Validation

### Constraint Verification
- State exclusivity enforcement
- Transition sequence validation
- Mass/energy balance closure
- Economic model consistency

### Solution Quality
- Optimal gap tolerance: 0.1%
- Feasibility verification at each timestep
- Constraint violation monitoring
- Economic objective validation

## Development Notes

This system represents a complete industrial optimization framework with:
- Realistic process parameters based on 300 kmol/hr H2 feed
- Proper transition dynamics and costs
- Adaptive learning capabilities
- Full shutdown operational flexibility
- Integration-ready architecture for detailed process simulation