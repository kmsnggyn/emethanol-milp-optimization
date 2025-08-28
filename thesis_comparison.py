"""
Final thesis comparison: Three scenarios for e-methanol plant optimization

This script provides the final comparison for your Master's thesis between:
1. Steady-state operations (always 100% or always 10%)
2. Perfect foresight optimization (full year)
3. Model Predictive Control with 24-hour rolling horizon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import load_data, build_model, solve_model
from improved_mpc import simple_mpc_strategy
import pyomo.environ as pyo


def run_thesis_comparison():
    """Run the complete thesis comparison with all three scenarios."""
    
    # Load data
    prices, params = load_data()
    if prices is None:
        print("Failed to load data!")
        return
    
    print("="*80)
    print("MASTER'S THESIS: E-METHANOL PLANT OPTIMIZATION COMPARISON")
    print("="*80)
    print(f"Study period: 8760 hours (full year)")
    print(f"Electricity price range: €{min(prices):.2f} - €{max(prices):.2f}/MWh")
    
    # Calculate break-even for reference
    delta_revenue = (params["M_100"] - params["M_10"]) * params["Price_Methanol"]
    delta_power = params["P_100"] - params["P_10"]
    delta_co2 = (params["C_100"] - params["C_10"]) * params["Price_CO2"]
    breakeven_price = (delta_revenue - delta_co2) / delta_power
    print(f"Economic break-even point: €{breakeven_price:.2f}/MWh")
    
    hours_below_breakeven = sum(1 for p in prices if p < breakeven_price)
    print(f"Hours below break-even: {hours_below_breakeven} ({100*hours_below_breakeven/len(prices):.1f}%)")
    
    results = {}
    
    # ==========================================
    # SCENARIO 1: STEADY-STATE OPERATIONS
    # ==========================================
    print("\n" + "="*50)
    print("SCENARIO 1: STEADY-STATE OPERATIONS")
    print("="*50)
    
    # Always 100%
    revenue_100 = len(prices) * params["M_100"] * params["Price_Methanol"]
    elec_cost_100 = sum(params["P_100"] * price for price in prices)
    co2_cost_100 = len(prices) * params["C_100"] * params["Price_CO2"]
    var_opex_100 = len(prices) * params["OPEX_Variable"]
    fixed_costs = params["Annualized_CAPEX"] + params["OPEX_Fixed"]
    profit_100 = revenue_100 - elec_cost_100 - co2_cost_100 - var_opex_100 - fixed_costs
    
    results['steady_100'] = {
        'name': 'Always 100%',
        'capacity_factor': 100.0,
        'ramp_events': 0,
        'annual_profit': profit_100,
        'description': 'Continuous operation at full capacity (baseline)'
    }
    
    print(f"Always 100%: CF=100.0%, Profit=€{profit_100:,.0f}, Ramps=0")
    print(f"Note: 10% steady-state operation excluded (economically irrational)")
    
    # ==========================================
    # SCENARIO 2: PERFECT FORESIGHT
    # ==========================================
    print("\n" + "="*50)
    print("SCENARIO 2: PERFECT FORESIGHT OPTIMIZATION")
    print("="*50)
    print("Running MILP optimization with full year price forecast...")
    
    model = build_model(prices, params)
    status = solve_model(model)
    
    if str(status) == 'optimal':
        pf_decisions = [int(pyo.value(model.x[t])) for t in range(len(prices))]
        pf_capacity_values = [10 if d == 0 else 100 for d in pf_decisions]
        pf_capacity_factor = np.mean(pf_capacity_values)
        pf_ramps = sum(1 for i in range(1, len(pf_decisions)) if pf_decisions[i] != pf_decisions[i-1])
        pf_profit = pyo.value(model.objective)
        
        results['perfect_foresight'] = {
            'name': 'Perfect Foresight',
            'capacity_factor': pf_capacity_factor,
            'ramp_events': pf_ramps,
            'annual_profit': pf_profit,
            'decisions': pf_decisions,
            'description': 'MILP optimization with perfect price forecast'
        }
        
        print(f"Perfect Foresight: CF={pf_capacity_factor:.1f}%, Profit=€{pf_profit:,.0f}, Ramps={pf_ramps}")
    else:
        print(f"Perfect foresight optimization failed: {status}")
        results['perfect_foresight'] = None
    
    # ==========================================
    # SCENARIO 3: MODEL PREDICTIVE CONTROL
    # ==========================================
    print("\n" + "="*50)
    print("SCENARIO 3: MODEL PREDICTIVE CONTROL (MPC)")
    print("="*50)
    print("Running MPC with 24-hour rolling forecast horizon...")
    
    mpc_result = simple_mpc_strategy(prices, params, forecast_horizon=24)
    
    results['mpc_24h'] = {
        'name': 'MPC (24h horizon)',
        'capacity_factor': mpc_result['capacity_factor'],
        'ramp_events': mpc_result['ramp_events'],
        'annual_profit': mpc_result['annual_profit'],
        'decisions': mpc_result['decisions'],
        'description': 'Model Predictive Control with realistic forecast limitation'
    }
    
    print(f"MPC (24h): CF={mpc_result['capacity_factor']:.1f}%, " + 
          f"Profit=€{mpc_result['annual_profit']:,.0f}, Ramps={mpc_result['ramp_events']}")
    
    # ==========================================
    # SUMMARY AND ANALYSIS
    # ==========================================
    print("\n" + "="*80)
    print("THESIS RESULTS SUMMARY")
    print("="*80)
    
    # Create summary table
    summary_scenarios = ['steady_100', 'perfect_foresight', 'mpc_24h']
    
    print(f"{'Scenario':<25} {'Capacity Factor':<15} {'Ramp Events':<12} {'Annual Profit':<15} {'Profit Rank':<12}")
    print("-" * 85)
    
    # Sort by profit for ranking
    profit_ranking = []
    for key in summary_scenarios:
        if results[key] is not None:
            profit_ranking.append((key, results[key]['annual_profit']))
    
    profit_ranking.sort(key=lambda x: x[1], reverse=True)
    profit_ranks = {scenario: rank+1 for rank, (scenario, _) in enumerate(profit_ranking)}
    
    for key in summary_scenarios:
        if results[key] is not None:
            r = results[key]
            rank = profit_ranks.get(key, 'N/A')
            print(f"{r['name']:<25} {r['capacity_factor']:<14.1f}% {r['ramp_events']:<12} " + 
                  f"€{r['annual_profit']:>12,.0f} {rank:<12}")
    
    # Key insights
    print("\n" + "="*50)
    print("KEY INSIGHTS FOR THESIS")
    print("="*50)
    
    if results['perfect_foresight'] and results['mpc_24h']:
        pf = results['perfect_foresight']
        mpc = results['mpc_24h']
        
        profit_gap = pf['annual_profit'] - mpc['annual_profit']
        cf_gap = pf['capacity_factor'] - mpc['capacity_factor']
        ramp_gap = pf['ramp_events'] - mpc['ramp_events']
        
        print(f"1. Value of Perfect Information:")
        print(f"   - Perfect foresight achieves €{profit_gap:,.0f} higher profit than MPC")
        print(f"   - Equivalent to €{profit_gap/8760:.0f} per operating hour")
        
        print(f"\n2. Operational Impact of Forecast Horizon:")
        print(f"   - Perfect foresight: {pf['capacity_factor']:.1f}% capacity factor")
        print(f"   - MPC (24h horizon): {mpc['capacity_factor']:.1f}% capacity factor")
        print(f"   - Forecast limitation reduces capacity utilization by {cf_gap:.1f} percentage points")
        
        print(f"\n3. Operational Flexibility:")
        print(f"   - Perfect foresight: {pf['ramp_events']} ramp events")
        print(f"   - MPC (24h horizon): {mpc['ramp_events']} ramp events")
        print(f"   - Limited forecast {'' if ramp_gap >= 0 else 'increases' if ramp_gap < -10 else 'slightly affects'} ramping frequency")
        
        print(f"\n3. Economic Insight:")
        print(f"   - Always 100% baseline: €{results['steady_100']['annual_profit']:,.0f}")
        print(f"   - Perfect foresight optimization adds: €{pf['annual_profit'] - results['steady_100']['annual_profit']:,.0f}")
        print(f"   - MPC (24h) vs baseline: €{mpc['annual_profit'] - results['steady_100']['annual_profit']:,.0f}")
        
        print(f"\n4. Practical Trade-offs:")
        best_profit = max(r['annual_profit'] for r in results.values() if r is not None)
        print(f"   - Best achievable profit: €{best_profit:,.0f}")
        
        if best_profit > 0:
            print(f"   ✓ Profitable operations possible under optimal conditions")
        else:
            print(f"   ⚠ All operational strategies show negative profitability")
            print(f"     Economic viability requires: lower CAPEX, higher methanol prices, or cheaper electricity")
    
    # Create thesis plots
    create_thesis_plots(results, prices, breakeven_price)
    
    return results


def create_thesis_plots(results, prices, breakeven_price):
    """Create publication-quality plots for thesis."""
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })
    
    # Main comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E-Methanol Plant Optimization: Thesis Results Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Capacity Factor Comparison
    ax1 = axes[0, 0]
    scenarios = []
    cfs = []
    colors = []
    
    for key in ['steady_100', 'perfect_foresight', 'mpc_24h']:
        if results[key] is not None:
            scenarios.append(results[key]['name'])
            cfs.append(results[key]['capacity_factor'])
            colors.append(['blue', 'green', 'orange'][len(scenarios)-1])
    
    bars1 = ax1.bar(scenarios, cfs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Capacity Factor (%)')
    ax1.set_title('Plant Utilization by Scenario')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cf in zip(bars1, cfs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cf:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Annual Profit Comparison
    ax2 = axes[0, 1]
    profits = [results[key]['annual_profit']/1e6 for key in ['steady_100', 'perfect_foresight', 'mpc_24h'] 
               if results[key] is not None]
    
    bars2 = ax2.bar(scenarios, profits, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Annual Profit (Million €)')
    ax2.set_title('Economic Performance by Scenario')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels and color coding
    for bar, profit in zip(bars2, profits):
        height = bar.get_height()
        color = 'green' if profit > 0 else 'red'
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                f'€{profit:.1f}M', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', color=color)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Plot 3: Operational Flexibility (Ramp Events)
    ax3 = axes[1, 0]
    ramps = [results[key]['ramp_events'] for key in ['steady_100', 'perfect_foresight', 'mpc_24h'] 
             if results[key] is not None]
    
    bars3 = ax3.bar(scenarios, ramps, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Ramp Events per Year')
    ax3.set_title('Operational Flexibility by Scenario')
    ax3.grid(True, alpha=0.3)
    
    for bar, ramp in zip(bars3, ramps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{ramp}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Time series comparison (first 168 hours = 1 week)
    ax4 = axes[1, 1]
    hours = range(168)
    
    # Price background
    ax4_twin = ax4.twinx()
    ax4_twin.plot(hours, prices[:168], color='lightblue', alpha=0.6, linewidth=1, label='Electricity Price')
    ax4_twin.axhline(y=breakeven_price, color='gray', linestyle=':', alpha=0.8, label=f'Break-even: €{breakeven_price:.0f}/MWh')
    ax4_twin.set_ylabel('Electricity Price (€/MWh)', color='blue')
    ax4_twin.tick_params(axis='y', labelcolor='blue')
    
    # Operating decisions
    if results['perfect_foresight'] and 'decisions' in results['perfect_foresight']:
        pf_capacity = [10 if d == 0 else 100 for d in results['perfect_foresight']['decisions'][:168]]
        ax4.plot(hours, pf_capacity, 'g-', linewidth=2, label='Perfect Foresight', alpha=0.8)
    
    if results['mpc_24h'] and 'decisions' in results['mpc_24h']:
        mpc_capacity = [10 if d == 0 else 100 for d in results['mpc_24h']['decisions'][:168]]
        ax4.plot(hours, mpc_capacity, 'r--', linewidth=2, label='MPC (24h)', alpha=0.8)
    
    ax4.set_xlabel('Hour of Year')
    ax4.set_ylabel('Plant Capacity (%)', color='red')
    ax4.set_title('First Week: Operation Comparison')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/thesis_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nThesis comparison plot saved to plots/thesis_comparison.png")
    
    # Additional detailed time series plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    hours_full = range(len(prices))
    plt.plot(hours_full, prices, color='lightblue', alpha=0.6, linewidth=0.5, label='Electricity Price')
    plt.axhline(y=breakeven_price, color='red', linestyle='--', linewidth=2, 
                label=f'Break-even: €{breakeven_price:.0f}/MWh')
    plt.ylabel('Electricity Price (€/MWh)')
    plt.title('Annual Electricity Prices and Plant Operations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    if results['perfect_foresight'] and 'decisions' in results['perfect_foresight']:
        pf_capacity_full = [10 if d == 0 else 100 for d in results['perfect_foresight']['decisions']]
        plt.plot(hours_full, pf_capacity_full, 'g-', alpha=0.7, linewidth=1, label='Perfect Foresight')
    
    if results['mpc_24h'] and 'decisions' in results['mpc_24h']:
        mpc_capacity_full = [10 if d == 0 else 100 for d in results['mpc_24h']['decisions']]
        plt.plot(hours_full, mpc_capacity_full, 'r-', alpha=0.7, linewidth=1, label='MPC (24h horizon)')
    
    plt.xlabel('Hour of Year')
    plt.ylabel('Plant Capacity (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/thesis_annual_operations.png', dpi=300, bbox_inches='tight')
    print(f"Annual operations plot saved to plots/thesis_annual_operations.png")


if __name__ == "__main__":
    results = run_thesis_comparison()
