# E-methanol Plant MILP Optimization

A Mixed-Integer Linear Programming (MILP) model for techno-economic optimization of green e-methanol plant operations with hourly scheduling and ramp penalties.

## 🎯 Project Overview

This repository contains a comprehensive optimization model for determining optimal operating schedules of an e-methanol production plant over an 8760-hour (1 year) time horizon. The model decides between high-capacity (100%) and low-capacity (10%) operation modes to maximize annual profit while considering electricity prices, ramp costs, and operational constraints.

## 🧮 Model Features

- **Binary Decision Variables**: Hour-by-hour operating mode decisions (100% vs 10% capacity)
- **Ramp Logic**: Explicit modeling of ramp-up and ramp-down events with associated penalties
- **Stabilization Constraints**: Minimum duration requirements for operational states
- **Economic Optimization**: Maximizes annual profit considering electricity costs, methanol revenue, and ramp penalties
- **Perfect Foresight**: Uses complete price information for global optimization

## 📊 Visualization Capabilities

The project includes comprehensive plotting functionality:

### Generated Plots
- **Operational Overview**: Time series showing electricity prices and capacity decisions
- **Operational Statistics**: Distribution analysis, daily patterns, and ramp frequency
- **Economic Analysis**: Revenue, costs, and profitability metrics
- **Summary Dashboard**: Comprehensive overview with key performance indicators
- **Monthly Performance**: Seasonal patterns and performance trends

### Key Metrics Tracked
- Capacity factor and operational patterns
- Ramp event frequency and timing
- Economic performance (revenue, costs, profit)
- Price-response behavior
- Seasonal and daily operational trends

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.13+
Pyomo >= 6.6.0
Gurobi >= 12.0.0 (with academic license) OR HiGHS >= 1.11.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/kmsnggyn/emethanol-milp-optimization.git
cd emethanol-milp-optimization

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Model

#### Basic Optimization
```bash
python main.py
```

#### Optimization with Visualization
```bash
python main_with_plots.py
```

#### Standalone Visualization (with sample data)
```bash
python visualize.py
```

## 📁 Project Structure

```
├── main.py                     # Core optimization script
├── main_with_plots.py         # Optimization with integrated plotting
├── model.py                   # Pyomo MILP model definition
├── visualize.py               # Comprehensive plotting module
├── generate_prices.py         # Electricity price data generation
├── quick_analysis.py          # Break-even analysis tools
├── update_parameters.py       # Parameter management utilities
├── check_solvers.py          # Solver availability checker
├── test_model.py             # Model validation tests
├── requirements.txt          # Python dependencies
├── data/
│   ├── dummy_prices.csv      # Sample electricity price data
│   └── parameter_template.csv # Excel-compatible parameter template
└── plots/                    # Generated visualization outputs
    ├── operational_overview_*.png
    ├── operational_statistics.png
    ├── economic_analysis.png
    ├── summary_dashboard.png
    └── monthly_performance_summary.png
```

## 🔧 Model Parameters

The model uses configurable parameters for:

### Plant Technical Parameters
- Power consumption at 100% and 10% capacity
- Methanol production rates
- CO2 consumption rates

### Ramp Penalties
- Production losses during ramp events
- Energy penalties for transitions
- Duration constraints

### Economic Parameters
- Methanol selling price
- CO2 costs
- Fixed and variable OPEX
- Annualized CAPEX

### Operational Constraints
- Minimum state duration (stabilization time)
- Capacity limits

## 📈 Sample Results

The model typically achieves:
- **Capacity Factor**: ~81-83% (high capacity operation)
- **Ramp Events**: ~300 per year (0.8 per day average)
- **Solve Time**: <0.3 seconds for 8760-hour optimization
- **Decision Quality**: Economically optimal with perfect foresight

## 🎓 Academic Use

This model is designed for academic research in:
- Renewable energy systems optimization
- Process industry scheduling
- Techno-economic analysis
- Mixed-integer programming applications
- Energy storage and demand response

## 🛠️ Customization

### Using Real Data
1. Replace dummy prices in `data/dummy_prices.csv` with actual electricity market data
2. Update plant parameters in `main.py` or use `parameter_template.csv`
3. Run `update_parameters.py` to automatically update parameters from Excel

### Extending the Model
- Modify `model.py` to add constraints or objectives
- Extend `visualize.py` for custom analysis plots
- Add new parameters through the template system

## 📊 Visualization Examples

The plotting system generates publication-ready figures including:

- **Time Series Analysis**: Price vs operational decisions over various timeframes
- **Statistical Distributions**: Capacity factor distributions and operational patterns  
- **Economic Breakdown**: Cost and revenue analysis with profitability metrics
- **Performance Dashboards**: Comprehensive overview with key operational metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model enhancements
- Additional visualization features
- Performance improvements
- Documentation updates

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Pyomo](http://www.pyomo.org/) optimization modeling language
- Solved using [Gurobi](https://www.gurobi.com/) and [HiGHS](https://highs.dev/) solvers
- Visualizations powered by [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)

---

*This project is part of academic research in renewable energy systems optimization and techno-economic analysis of e-methanol production.*
