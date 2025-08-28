# E-methanol Plant MILP Optimization

Mixed-Integer Linear Programming model for techno-economic optimization of green e-methanol plant operations.

## Overview

This optimization model determines the optimal hourly operating schedule for an e-methanol production plant over one year (8760 hours). The model chooses between high-capacity (100%) and low-capacity (10%) operation modes to maximize annual profit considering electricity prices, ramp costs, and operational constraints.

## Model Features

- Binary decision variables for hourly operating mode (100% vs 10% capacity)
- Ramp logic with explicit modeling of startup/shutdown events and penalties
- Stabilization constraints requiring minimum duration in each operational state
- Economic optimization maximizing annual profit
- Perfect foresight optimization using complete price information

## Requirements

- Python 3.8+
- Pyomo >= 6.6.0
- Gurobi >= 12.0.0 (academic license) OR HiGHS >= 1.11.0
- pandas, numpy, matplotlib (see requirements.txt)

## Usage

```bash
# Clone repository
git clone https://github.com/kmsnggyn/emethanol-milp-optimization.git
cd emethanol-milp-optimization

# Install dependencies
pip install -r requirements.txt

# Run optimization
python main.py
```

## File Structure

- `main.py` - Main optimization script
- `model.py` - Pyomo MILP model definition  
- `generate_prices.py` - Creates dummy electricity price data
- `quick_analysis.py` - Break-even analysis for understanding decision logic
- `data/dummy_prices.csv` - 8760 hours of electricity price data
- `data/parameter_template.csv` - Template for updating model parameters

## Model Parameters

Key parameters include plant technical specifications (power consumption, production rates), economic factors (methanol price, costs), and operational constraints (minimum state duration). Default values are provided for testing; replace with actual plant data for real applications.

## Results

Typical optimization results:
- Capacity factor: ~81-83%
- Annual ramp events: ~300
- Solve time: <0.3 seconds
- Optimal economic decisions balancing electricity costs and ramp penalties

This model is designed for academic research in renewable energy systems optimization and process industry scheduling.
