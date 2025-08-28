"""
Script to update model parameters from Excel spreadsheet.

This script reads your real parameter values from the parameter_template.csv file
and automatically updates the main.py file with your Aspen simulation data.
"""

import pandas as pd
import re


def read_parameters_from_csv(csv_file="data/parameter_template.csv"):
    """
    Read parameter values from the CSV template.
    
    Returns:
        dict: Dictionary of parameter values
    """
    
    try:
        df = pd.read_csv(csv_file)
        params = {}
        
        print("Reading parameters from spreadsheet...")
        print("=" * 50)
        
        for _, row in df.iterrows():
            var_code = row['Variable Code']
            your_value = row['Your Value']
            current_dummy = row['Current Dummy Value']
            
            # Use your value if provided, otherwise keep dummy value
            if pd.notna(your_value) and str(your_value).strip() != "":
                params[var_code] = float(your_value)
                print(f"✓ {var_code}: {your_value} {row['Unit']} (UPDATED)")
            else:
                params[var_code] = float(current_dummy)
                print(f"- {var_code}: {current_dummy} {row['Unit']} (dummy value)")
        
        print("=" * 50)
        return params
        
    except FileNotFoundError:
        print("Error: parameter_template.csv not found!")
        print("Please make sure the CSV file exists in the data/ folder.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def update_main_py(new_params):
    """
    Update the main.py file with new parameter values.
    
    Args:
        new_params (dict): Dictionary of parameter values
    """
    
    try:
        # Read the current main.py file
        with open('main.py', 'r') as file:
            content = file.read()
        
        # Create the new parameters section
        params_section = '    # Model parameters (updated from Aspen simulation data)\n'
        params_section += '    params = {\n'
        params_section += '        # == PLANT TECHNICAL PARAMETERS ==\n'
        params_section += '        # Based on Aspen Plus steady-state simulations\n'
        params_section += f'        "P_100": {new_params["P_100"]},  # Power consumption at 100% load [MW]\n'
        params_section += f'        "M_100": {new_params["M_100"]},    # Methanol production at 100% load [ton/hr]\n'
        params_section += f'        "C_100": {new_params["C_100"]},    # CO2 consumption at 100% load [ton/hr]\n'
        params_section += '        \n'
        params_section += f'        "P_10": {new_params["P_10"]},     # Power consumption at 10% load [MW]\n'
        params_section += f'        "M_10": {new_params["M_10"]},     # Methanol production at 10% load [ton/hr]\n'
        params_section += f'        "C_10": {new_params["C_10"]},     # CO2 consumption at 10% load [ton/hr]\n'
        params_section += '\n'
        params_section += '        # == DYNAMIC RAMP PENALTIES ==\n'
        params_section += '        # These represent the total deviation from steady-state over the ramp event duration\n'
        params_section += '        # Ramp-Up Event (10% → 100%)\n'
        params_section += f'        "Production_Loss_Up": {new_params["Production_Loss_Up"]},   # Total tons of methanol NOT produced vs. staying at 100%\n'
        params_section += f'        "Energy_Penalty_Up": {new_params["Energy_Penalty_Up"]},    # Extra MWh consumed vs. staying at 100%\n'
        params_section += '        \n'
        params_section += '        # Ramp-Down Event (100% → 10%)\n'
        params_section += f'        "Production_Loss_Down": {new_params["Production_Loss_Down"]}, # Total tons of methanol NOT produced vs. staying at 10%\n'
        params_section += f'        "Energy_Penalty_Down": {new_params["Energy_Penalty_Down"]},   # Extra MWh consumed vs. staying at 10%\n'
        params_section += '\n'
        params_section += '        # == ECONOMIC PARAMETERS ==\n'
        params_section += f'        "Price_Methanol": {new_params["Price_Methanol"]},     # €/ton\n'
        params_section += f'        "Price_CO2": {new_params["Price_CO2"]},           # €/ton\n'
        params_section += f'        "Annualized_CAPEX": {new_params["Annualized_CAPEX"]},   # €/year\n'
        params_section += f'        "OPEX_Fixed": {new_params["OPEX_Fixed"]},       # €/year (staff, maintenance, etc.)\n'
        params_section += f'        "OPEX_Variable": {new_params["OPEX_Variable"]},      # Additional variable costs per hour of operation [€/hr]\n'
        params_section += '\n'
        params_section += '        # == OPERATIONAL CONSTRAINTS ==\n'
        params_section += f'        "T_stab": {int(new_params["T_stab"])}  # Minimum hours plant must stay in a state after ramping\n'
        params_section += '    }'
        
        # Find and replace the existing params section
        pattern = r'    # Model parameters.*?params = \{.*?\n    \}'
        new_content = re.sub(pattern, params_section, content, flags=re.DOTALL)
        
        # Write the updated content back to main.py
        with open('main.py', 'w') as file:
            file.write(new_content)
        
        print("\n✓ Successfully updated main.py with new parameters!")
        print("You can now run 'python main.py' with your real data.")
        
    except Exception as e:
        print(f"Error updating main.py: {e}")


def main():
    """
    Main function to update parameters from spreadsheet.
    """
    
    print("E-Methanol Parameter Update Tool")
    print("=" * 40)
    print("This tool reads your parameter values from parameter_template.csv")
    print("and updates the main.py file automatically.")
    print()
    
    # Read parameters from CSV
    params = read_parameters_from_csv()
    
    if params is None:
        return
    
    # Ask for confirmation
    response = input("\nDo you want to update main.py with these values? (y/n): ")
    if response.lower() in ['y', 'yes']:
        update_main_py(params)
    else:
        print("Update cancelled.")


if __name__ == "__main__":
    main()
