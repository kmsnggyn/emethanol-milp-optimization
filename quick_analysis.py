"""
Quick analysis of the ramp decision logic
"""

import pandas as pd
import numpy as np

def quick_break_even_analysis():
    """Quick calculation of break-even prices for ramping decisions"""
    
    # Parameters from main.py
    params = {
        "P_100": 100.0, "M_100": 8.5, "P_10": 15.0, "M_10": 0.85,
        "Production_Loss_Down": 1.5, "Energy_Penalty_Down": 5.0,
        "Production_Loss_Up": 4.0, "Energy_Penalty_Up": 10.0,
        "Price_Methanol": 750.0, "T_stab": 3
    }
    
    # Load price data to get average
    try:
        price_df = pd.read_csv('data/dummy_prices.csv')
        avg_price = price_df['price_eur_per_mwh'].mean()
    except:
        avg_price = 50.0  # fallback
    
    print("RAMP DECISION LOGIC ANALYSIS")
    print("="*40)
    
    # Key metrics
    power_savings = params["P_100"] - params["P_10"]  # 85 MW
    production_loss_per_hour = params["M_100"] - params["M_10"]  # 7.65 ton/hr
    production_loss_cost_per_hour = production_loss_per_hour * params["Price_Methanol"]
    
    print(f"Power savings when ramping down: {power_savings} MW")
    print(f"Production loss when at 10%: {production_loss_per_hour:.2f} ton/hr")
    print(f"Production loss cost: €{production_loss_cost_per_hour:,.0f}/hr")
    print(f"Average electricity price: €{avg_price:.1f}/MWh")
    
    # Calculate ramp penalties
    ramp_down_penalty = (params["Production_Loss_Down"] * params["Price_Methanol"] + 
                        params["Energy_Penalty_Down"] * avg_price)
    ramp_up_penalty = (params["Production_Loss_Up"] * params["Price_Methanol"] + 
                      params["Energy_Penalty_Up"] * avg_price)
    total_ramp_penalty = ramp_down_penalty + ramp_up_penalty
    
    print(f"\nRAMP PENALTIES:")
    print(f"Ramp-down penalty: €{ramp_down_penalty:,.0f}")
    print(f"Ramp-up penalty: €{ramp_up_penalty:,.0f}")
    print(f"Total ramp penalty: €{total_ramp_penalty:,.0f}")
    
    # Break-even analysis for minimum duration (T_stab = 3 hours)
    min_duration = params["T_stab"]
    total_production_loss = min_duration * production_loss_cost_per_hour
    total_cost = total_production_loss + total_ramp_penalty
    
    # Break-even: power_savings * duration * price = total_cost
    break_even_price = total_cost / (power_savings * min_duration)
    
    print(f"\nBREAK-EVEN ANALYSIS (minimum {min_duration}h duration):")
    print(f"Total production loss cost: €{total_production_loss:,.0f}")
    print(f"Total ramp penalties: €{total_ramp_penalty:,.0f}")
    print(f"Total cost of ramping: €{total_cost:,.0f}")
    print(f"Power savings per hour: {power_savings} MW × price")
    print(f"Break-even price: €{break_even_price:.1f}/MWh")
    
    print(f"\n🎯 DECISION RULE:")
    print(f"The model will ramp down when electricity price > €{break_even_price:.1f}/MWh")
    print(f"(assuming the plant stays at 10% for at least {min_duration} hours)")
    
    # Show sensitivity to duration
    print(f"\nSENSITIVITY TO DURATION:")
    for hours in [3, 6, 12, 24]:
        total_prod_loss = hours * production_loss_cost_per_hour
        total_cost_duration = total_prod_loss + total_ramp_penalty
        be_price = total_cost_duration / (power_savings * hours)
        print(f"{hours:2d} hours at 10%: Break-even = €{be_price:.1f}/MWh")

if __name__ == "__main__":
    quick_break_even_analysis()
