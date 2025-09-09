# E-Methanol Plant Optimization System

A Mixed-Integer Linear Programming (MILP) optimization system for industrial-scale green e-methanol production with binary operational states and perfect forecast analysis.

## System Architecture

### Core Components

**model.py** - MILP optimization engine
- Binary state system: 100% load or 10% load operation
- Industrial-scale parameters: 604.8 kg/hr H2 + 4,401 kg/hr CO2 feed
- Economic model: €800/ton methanol, 32.4 MW power consumption
- Mass-based material flows and costs

**run_optimization.py** - Perfect forecast analysis
- Complete electricity price knowledge optimization
- Economic analysis and visualization
- Performance metrics calculation
- Breakeven analysis

## Mathematical Formulation

### State Variables
```
x_100[t] ∈ {0,1}    - Operating at 100% capacity
x_10[t] ∈ {0,1}     - Operating at 10% capacity
y_ramp_up[t] ∈ {0,1}   - Ramping up from 10% to 100%
y_ramp_down[t] ∈ {0,1} - Ramping down from 100% to 10%
```

### Operational Constraints
- State exclusivity: x_100[t] + x_10[t] = 1 ∀t
- Ramping logic: y_ramp_up[t] ≤ x_10[t-1] × x_100[t]
- Stabilization periods: 4-hour post-ramp stabilization
- Power consumption: 32.4 MW (100%), 3.34 MW (10%)

### Economic Objective
```
max Σ[t] (Revenue[t] - Electricity[t] - CO2[t] - OPEX[t] - Fixed_Costs)

Where:
Revenue[t] = Production[t] × €0.8/kg
Electricity[t] = Power[t] × Price_elec[t]
CO2[t] = CO2_consumption[t] × €0.05/kg
```

## Process Parameters

### Feed Specifications (Mass-Based)
- Hydrogen: 604.8 kg/hr (300 kmol/hr)
- Carbon Dioxide: 4,401 kg/hr (100 kmol/hr) at 100% load
- Methanol Production: 2,689 kg/hr (2.689 ton/hr) at 100% load

### Power Consumption
- Electrolysis: 31.4 MW (100%), 3.14 MW (10%)
- Methanol Plant: 1.0 MW (100%), 0.2 MW (10%)
- Total: 32.4 MW (100%), 3.34 MW (10%)

### Economic Parameters
- Methanol Price: €0.8/kg (€800/ton)
- CO2 Price: €0.05/kg (€50/ton)
- Annual Fixed Costs: €10.26M (CAPEX + Fixed OPEX)
- 100% Load Breakeven: €9.42/MWh
- 10% Load Breakeven: €14.07/MWh

### Ramping Dynamics
- Production Loss: 100% (no production during ramps)
- Energy Penalty: 50% additional power during ramps
- Stabilization Period: 4 hours post-ramp

## Perfect Forecast Analysis

### Optimization Approach
- Complete knowledge of all electricity prices
- No forecasting uncertainty
- Theoretical maximum profit scenario
- Optimal load selection based on price thresholds

### Key Results (2023 Data)
- Total Profit: €-5.65M (theoretical maximum)
- Capacity Factor: 65.2%
- Operation: 61.3% at 100% load, 38.7% at 10% load
- No ramping events (perfect foresight)

## Usage

### Basic Optimization
```python
from model import build_model, solve_model, load_data, get_parameters

# Load electricity price data
data = load_data()
prices = data['price']
params = get_parameters()

# Build and solve model
model = build_model(prices, params)
termination_condition = solve_model(model)
```

### Run Complete Analysis
```python
python run_optimization.py
```

## System Performance

### Operational Characteristics
- Capacity Factor: 65.2% (perfect forecast scenario)
- Production Efficiency: 2.689 ton/hour at 100% load
- Power Efficiency: 12.0 MWh/ton methanol (100% load)
- Turndown Capability: 10% minimum load

### Economic Performance
- Break-even Operation: Electricity prices < €9.42/MWh (100% load)
- Break-even Operation: Electricity prices < €14.07/MWh (10% load)
- Annual Fixed Costs: €10.26M
- Variable Costs: €305/hour (100% load), €47/hour (10% load)
- Revenue Potential: €18.8M/year at full capacity

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
- Solution time: < 2 seconds for full year optimization

### Data Management
- Excel-based price data loading (2023 Nord Pool data)
- Pandas DataFrame integration
- Mass-based material flow calculations
- Real-time parameter updates

### Performance Optimization
- Vectorized constraint generation
- Efficient state tracking
- Memory-optimized data structures
- Dynamic time horizon based on data length

## Market Data Integration

### 2023-2024 Electrolysis Technology
- Specific Energy: 52 kWh/kg H2 (77% LHV efficiency)
- CAPEX: €1,666-2,250/kW (European systems)
- Fixed O&M: €43/kW/year
- Stack Replacement: Every 7.5 years (40% of CAPEX)

### Electricity Price Data
- Source: Nord Pool SE3 zone (Sweden)
- Period: 2023 (8,285 hours available)
- Price Range: €0.01 - €332.00/MWh
- Average Price: €54.80/MWh

## Key Insights

### Economic Viability
- Plant operates at 100% load when electricity < €9.42/MWh
- Plant operates at 10% load when electricity < €14.07/MWh
- Perfect forecast enables optimal load selection
- Binary operation provides operational flexibility

### Technology Performance
- 35.8% reduction in electrolysis power vs. older estimates
- Mass-based flows improve model clarity
- Realistic CAPEX and OPEX based on market data
- Proper electricity cost calculation (hourly variable)

## Development Notes

This system represents a complete industrial optimization framework with:
- Realistic process parameters based on 2023-2024 market data
- Mass-based material flows for clarity
- Perfect forecast optimization for theoretical maximum
- Integration-ready architecture for detailed process simulation