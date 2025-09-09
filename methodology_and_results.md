# E-Methanol Plant Optimization: Methodology and Results

## METHODOLOGY

### 2.1 System Description

The e-methanol production system consists of two main components:
- **Alkaline water electrolysis unit** (31.4 MW capacity) producing hydrogen
- **Methanol synthesis plant** (2.689 tonnes/hour capacity) converting CO₂ and H₂ to methanol

The system operates under a binary operational strategy with two distinct modes:
- **100% capacity**: Full production at 2,689 kg/hr methanol output
- **10% capacity**: Minimum turndown at 269 kg/hr methanol output

### 2.2 Mathematical Model

The optimization problem is formulated as a Mixed-Integer Linear Programming (MILP) model with perfect foresight of electricity prices. The objective function maximizes annual profit:

```
maximize: Revenue - (Electricity Costs + CO₂ Costs + OPEX + CAPEX)
```

**Key Variables:**
- `x_100[t]`, `x_10[t]`: Binary variables for operational states at time t
- `y_ramp_up[t]`, `y_ramp_down[t]`: Binary variables for ramping transitions

**Constraints:**
- Exactly one operational state per hour: `x_100[t] + x_10[t] + y_ramp_up[t] + y_ramp_down[t] = 1`
- Minimum stabilization period: 4 hours after ramping
- No production during ramping transitions

### 2.3 Key Assumptions

#### 2.3.1 Technical Assumptions
- **Electrolysis efficiency**: 52 kWh/kg H₂ (77% LHV efficiency)
- **Methanol plant efficiency**: Linear scaling with reduced efficiency at 10% load
- **Ramping behavior**: 
  - 50% energy penalty during transitions
  - Complete production loss during ramping hours
  - 4-hour stabilization period required
- **Plant availability**: 100% (no maintenance downtime considered)

#### 2.3.2 Economic Assumptions
- **Methanol price**: €800/tonne (green methanol premium included)
- **CO₂ price**: €50/tonne (captured CO₂ cost)
- **Electricity prices**: Nord Pool SE3 zone, 2023 data (8,760 hours)
- **Discount rate**: 7% for CAPEX annualization
- **Plant lifetime**: 20 years

#### 2.3.3 CAPEX and OPEX Structure
**Capital Expenditures (Annualized):**
- Electrolysis system: €5.9M/year (€1,666/kW installed)
- Methanol plant: €226,399/year
- Total annualized CAPEX: €6.13M/year

**Operating Expenditures:**
- Fixed OPEX: €4.14M/year (plant O&M + stack replacement)
- Variable OPEX at 100%: €85/hour (€0.032/kg methanol)
- Variable OPEX at 10%: €25/hour (€0.093/kg methanol - efficiency penalty)

#### 2.3.4 Model Limitations
- **Perfect foresight**: Complete knowledge of electricity prices (theoretical maximum)
- **Binary operation**: No intermediate capacity levels between 10% and 100%
- **No demand constraints**: Unlimited methanol market assumed
- **No grid constraints**: Perfect electricity market access
- **Deterministic prices**: No price uncertainty or forecasting errors

### 2.4 Solver Implementation

The MILP model was implemented using Pyomo 6.7+ and solved with HiGHS solver. The optimization covers the full year 2023 (8,760 hours) with:
- **Decision variables**: 35,040 binary variables
- **Constraints**: ~70,000 linear constraints
- **Solve time**: ~2 seconds on standard hardware

---

## RESULTS

### 3.1 Economic Performance Comparison

Three operational strategies were evaluated:
1. **100% All Year**: Continuous operation at full capacity
2. **10% All Year**: Continuous operation at minimum capacity  
3. **Dynamic Optimization**: Perfect foresight optimization

| Strategy | Annual Profit | Annual Revenue | Annual Costs | Capacity Factor |
|----------|---------------|----------------|--------------|-----------------|
| 100% All Year | €9.5M | €18.8M | €9.3M | 100% |
| 10% All Year | €-11.4M | €1.9M | €13.3M | 10% |
| **Dynamic Opt** | **€11.5M** | **€13.4M** | **€1.9M** | **63.7%** |

**Key Finding**: Dynamic optimization achieves 21% higher profit (€2.0M additional) compared to continuous 100% operation.

### 3.2 Cost Structure Analysis (per tonne methanol)

| Cost Component | 100% All Year | Dynamic Opt | Savings |
|----------------|---------------|-------------|---------|
| **Electricity** | €623/tonne | €333/tonne | €290/tonne |
| CO₂ Purchase | €82/tonne | €82/tonne | €0/tonne |
| Variable OPEX | €32/tonne | €35/tonne | €-3/tonne |
| Fixed OPEX | €176/tonne | €262/tonne | €-86/tonne |
| CAPEX | €260/tonne | €388/tonne | €-128/tonne |
| **Total** | **€1,172/tonne** | **€1,100/tonne** | **€72/tonne** |

**Critical Insight**: Electricity cost reduction (47% savings) more than compensates for higher fixed cost allocation due to reduced production.

### 3.3 Operational Characteristics

#### 3.3.1 Dynamic Operation Profile
- **Operating hours at 100%**: 5,584 hours (63.7% of year)
- **Operating hours at 10%**: 2,847 hours (32.5% of year)
- **Ramping hours**: 329 hours (3.8% of year)
- **Average electricity price during 100% operation**: €35.2/MWh
- **Average electricity price during 10% operation**: €8.1/MWh

#### 3.3.2 Electricity Market Utilization
- **Breakeven price (100% operation)**: €9.42/MWh
- **Breakeven price (10% operation)**: €14.07/MWh
- **Hours below 100% breakeven**: 2,847 hours (32.5%)
- **Hours above 10% breakeven**: 5,584 hours (63.7%)
- **Electricity price range**: -€60.04 to €332.00/MWh

### 3.4 Sensitivity Analysis Notes

*[To be expanded with sensitivity studies on:]*
- Methanol price variations (€600-1000/tonne)
- CO₂ price sensitivity (€0-100/tonne)
- Electricity price forecasting accuracy
- CAPEX cost variations
- Ramping penalty impacts

### 3.5 Economic Breakeven Analysis

The 10% all-year strategy demonstrates fundamental economic unviability with:
- **Negative annual profit**: €-11.4M
- **Cost per tonne**: €5,175/tonne (4.4× higher than optimized strategy)
- **Fixed cost burden**: €1,757/tonne fixed OPEX + €2,601/tonne CAPEX allocation

This confirms that **continuous low-capacity operation is economically unsustainable** due to high fixed cost allocation over reduced production.

### 3.6 Key Performance Indicators

| Metric | Value | Unit |
|--------|-------|------|
| **Profit Improvement** | +21% | vs 100% baseline |
| **Electricity Cost Reduction** | -47% | per tonne methanol |
| **Capacity Utilization** | 63.7% | annual average |
| **Carbon Intensity** | 1.64 | kg CO₂/kg methanol |
| **Energy Efficiency** | 52 | kWh/kg H₂ |
| **Economic Efficiency** | €1,100 | per tonne methanol |

---

## DISCUSSION POINTS TO EXPAND

1. **Comparison with literature values** for e-methanol production costs
2. **Grid stability implications** of dynamic operation
3. **Real-world implementation challenges** (forecasting accuracy, ramping constraints)
4. **Policy implications** for renewable energy integration
5. **Scalability analysis** for different plant sizes
6. **Seasonal operation patterns** and their economic drivers
7. **Technology learning curves** and future cost projections

---

## CONCLUSIONS (Draft Notes)

- Dynamic operation with electricity price optimization achieves **21% profit improvement**
- **Electricity costs dominate** variable expenses (53% of total costs in 100% operation)
- **Binary operation strategy** is economically superior to continuous low-capacity operation
- **Perfect foresight optimization** provides theoretical upper bound for profit maximization
- Results support **power-to-X integration** with variable renewable electricity markets

*Note: These results assume perfect foresight and should be validated with realistic forecasting methods.*
