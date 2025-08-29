"""
True MILP-based Model Predictive Control for E-Methanol Plant

This implements proper MPC that:
1. Uses forecasted prices for next 24 hours
2. Solves the full MILP optimization problem each hour
3. Takes only the first hour decision and repeats
4. No analytical shortcuts - let the solver find optimal strategy
"""

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import pickle
from model import build_model, solve_model, load_data


class PlantState:
    """Track the current state of the plant for MPC implementation with shutdown capability."""
    
    def __init__(self):
        # Plant states: 'shutdown', 'startup', 'running', 'shutdown_trans'
        self.current_state = 'shutdown'  # Start in shutdown
        self.transition_remaining = 0    # Hours remaining in current transition
        self.stabilization_remaining = 0 # Hours remaining in stabilization after transition
        
    def update(self, new_state, params):
        """Update plant state after a decision using parameters from model."""
        startup_duration = params.get('startup_duration', 6)
        shutdown_duration = params.get('shutdown_duration', 6) 
        stabilization_hours = params.get('T_stab', 4)
        
        # Handle transition logic
        if new_state != self.current_state:
            # State change requested
            if self.current_state == 'shutdown' and new_state == 'running':
                # Start startup sequence
                self.current_state = 'startup'
                self.transition_remaining = startup_duration - 1  # -1 because we're entering first hour
                self.stabilization_remaining = 0
                
            elif self.current_state == 'running' and new_state == 'shutdown':
                # Start shutdown sequence  
                self.current_state = 'shutdown_trans'
                self.transition_remaining = shutdown_duration - 1
                self.stabilization_remaining = 0
                
        else:
            # No state change requested, continue current process
            if self.transition_remaining > 0:
                # Still in transition
                self.transition_remaining -= 1
                
                # Check if transition is complete
                if self.transition_remaining == 0:
                    if self.current_state == 'startup':
                        self.current_state = 'running'
                        self.stabilization_remaining = stabilization_hours
                    elif self.current_state == 'shutdown_trans':
                        self.current_state = 'shutdown'
                        self.stabilization_remaining = stabilization_hours
                        
            elif self.stabilization_remaining > 0:
                # In stabilization period
                self.stabilization_remaining -= 1
    
    def can_start_transition(self):
        """Check if plant can start a new transition."""
        return (self.transition_remaining == 0 and 
                self.stabilization_remaining == 0 and
                self.current_state in ['shutdown', 'running'])


class AdaptivePatternForecaster:
    """
    Adaptive pattern-based forecaster that learns from observed data in real-time.
    
    Updates forecasting patterns as new data becomes available during MPC simulation.
    Uses decomposition into hourly, daily, weekly, and trend components.
    """
    
    def __init__(self):
        self.observed_prices = []
        self.observed_timestamps = []
        
        # Pattern components (updated as data comes in)
        self.hourly_patterns = np.zeros(24)  # Average price by hour of day
        self.daily_patterns = np.zeros(7)    # Average price by day of week
        self.weekly_trends = []              # Weekly average trends
        self.seasonal_component = 0.0        # Long-term seasonal trend
        
        # Learning parameters
        self.min_data_points = 48            # Need at least 2 days of data
        self.smoothing_factor = 0.1          # For exponential smoothing
        
    def update_with_observation(self, timestamp_hour, actual_price):
        """
        Update forecasting patterns with newly observed price data.
        
        Args:
            timestamp_hour (int): Hour index (0-8759)
            actual_price (float): Observed electricity price [€/MWh]
        """
        self.observed_prices.append(actual_price)
        self.observed_timestamps.append(timestamp_hour)
        
        # Only update patterns if we have sufficient data
        if len(self.observed_prices) >= self.min_data_points:
            self._update_patterns()
    
    def _update_patterns(self):
        """Update all forecasting patterns based on observed data."""
        
        # Convert to arrays for easier processing
        prices = np.array(self.observed_prices)
        timestamps = np.array(self.observed_timestamps)
        
        # Update hourly patterns (hour-of-day effects)
        for hour in range(24):
            hour_mask = (timestamps % 24) == hour
            if np.any(hour_mask):
                hour_prices = prices[hour_mask]
                # Exponential smoothing with recent data
                if len(hour_prices) > 0:
                    recent_avg = np.mean(hour_prices[-min(7, len(hour_prices)):])  # Last week
                    self.hourly_patterns[hour] = (
                        self.smoothing_factor * recent_avg + 
                        (1 - self.smoothing_factor) * self.hourly_patterns[hour]
                    )
        
        # Update daily patterns (day-of-week effects)
        if len(prices) >= 24:  # At least one day
            for day in range(7):
                # Find prices for this day of week
                day_hours = []
                for i, ts in enumerate(timestamps):
                    # Approximate day of week (assuming hour 0 is Monday)
                    approx_day = (ts // 24) % 7
                    if approx_day == day:
                        day_hours.append(prices[i])
                
                if day_hours:
                    recent_avg = np.mean(day_hours[-min(24, len(day_hours)):])
                    self.daily_patterns[day] = (
                        self.smoothing_factor * recent_avg + 
                        (1 - self.smoothing_factor) * self.daily_patterns[day]
                    )
        
        # Update weekly trends (capture longer-term movements)
        if len(prices) >= 168:  # At least one week
            weekly_avgs = []
            for week_start in range(0, len(prices) - 167, 168):
                week_data = prices[week_start:week_start + 168]
                weekly_avgs.append(np.mean(week_data))
            
            self.weekly_trends = weekly_avgs
            
            # Update seasonal component (long-term trend)
            if len(weekly_avgs) >= 2:
                # Simple linear trend
                recent_weeks = weekly_avgs[-min(4, len(weekly_avgs)):]
                self.seasonal_component = np.mean(np.diff(recent_weeks))
    
    def forecast(self, current_hour, horizon_hours=24):
        """
        Generate forecast based on learned patterns and recent observations.
        
        Args:
            current_hour (int): Current hour index (0-8759)
            horizon_hours (int): Number of hours to forecast ahead
            
        Returns:
            list: Forecasted prices for next horizon_hours
        """
        
        if len(self.observed_prices) < self.min_data_points:
            # Fallback to simple pattern if insufficient data
            return self._simple_fallback_forecast(current_hour, horizon_hours)
        
        forecast_prices = []
        
        for h in range(horizon_hours):
            future_hour = current_hour + h
            
            # Base price from hourly pattern
            hour_of_day = future_hour % 24
            base_price = self.hourly_patterns[hour_of_day]
            
            # If base price is 0 (no data yet), use recent average
            if base_price == 0:
                base_price = np.mean(self.observed_prices[-min(24, len(self.observed_prices)):])
            
            # Apply daily pattern adjustment
            day_of_week = (future_hour // 24) % 7
            daily_factor = 1.0
            if self.daily_patterns[day_of_week] > 0:
                overall_avg = np.mean(self.observed_prices[-min(168, len(self.observed_prices)):])
                if overall_avg > 0:
                    daily_factor = self.daily_patterns[day_of_week] / overall_avg
            
            # Apply weekly trend
            weeks_ahead = h // 168
            trend_adjustment = weeks_ahead * self.seasonal_component
            
            # Combine components
            forecast_price = base_price * daily_factor + trend_adjustment
            
            # Add realistic uncertainty based on recent forecast errors
            if len(self.observed_prices) >= 48:
                recent_volatility = np.std(self.observed_prices[-48:])  # Last 2 days volatility
                noise = np.random.normal(0, recent_volatility * 0.3)  # 30% of recent volatility
                forecast_price += noise
            
            # Keep price within reasonable bounds
            forecast_price = max(10.0, min(150.0, forecast_price))
            forecast_prices.append(forecast_price)
        
        return forecast_prices
    
    def _simple_fallback_forecast(self, current_hour, horizon_hours):
        """Fallback forecast when insufficient data is available."""
        forecast_prices = []
        
        # Use simple daily pattern
        for h in range(horizon_hours):
            future_hour = current_hour + h
            hour_of_day = future_hour % 24
            
            # Simple pattern
            if 6 <= hour_of_day <= 18:  # Day
                base_price = 50.0
            else:  # Night
                base_price = 40.0
            
            # Add some noise
            forecast_price = base_price + np.random.normal(0, 5.0)
            forecast_price = max(10.0, min(100.0, forecast_price))
            forecast_prices.append(forecast_price)
        
        return forecast_prices


class ForecastingMPC:
    """
    True MILP-based MPC using adaptive pattern-based price forecasting.
    
    This solves the full optimization problem each hour using forecasted prices,
    implements the first hour decision, then re-optimizes with updated forecasts.
    
    The forecaster learns and adapts from observed price data during simulation.
    """
    
    def __init__(self):
        """Initialize with adaptive forecaster."""
        self.adaptive_forecaster = AdaptivePatternForecaster()
        print("Initialized adaptive pattern-based forecaster")
        print("  Will learn from observed data during MPC simulation")
    
    def get_forecast(self, current_hour, actual_prices_so_far, horizon_hours=24):
        """
        Get price forecast for the next horizon_hours using adaptive learning.
        
        Args:
            current_hour (int): Current hour index (0-8759)
            actual_prices_so_far (list): Actual prices observed from hour 0 to current_hour
            horizon_hours (int): Forecast horizon
            
        Returns:
            list: Forecasted prices for next horizon_hours
        """
        
        # Update forecaster with all observed data up to current hour
        # (In practice, this would be more efficient with incremental updates)
        for hour_idx, price in enumerate(actual_prices_so_far):
            if hour_idx < len(self.adaptive_forecaster.observed_prices):
                continue  # Already have this data
            self.adaptive_forecaster.update_with_observation(hour_idx, price)
        
        # Generate forecast using learned patterns
        forecast_prices = self.adaptive_forecaster.forecast(current_hour, horizon_hours)
        
        return forecast_prices
    
    def build_mpc_model(self, forecast_prices, plant_state, params):
        """
        Build MILP model for MPC horizon using forecasted prices and shutdown capability.
        
        Args:
            forecast_prices (list): Forecasted prices for horizon
            plant_state (PlantState): Current plant state
            params (dict): Model parameters
            
        Returns:
            pyo.ConcreteModel: Configured MILP model for MPC horizon
        """
        
        horizon = len(forecast_prices)
        
        # Initialize the model
        model = pyo.ConcreteModel()
        
        # Time horizon: 0 to horizon-1
        model.T = pyo.RangeSet(0, horizon-1)
        
        # Electricity prices for forecast horizon
        model.price_elec = pyo.Param(model.T, initialize=dict(enumerate(forecast_prices)), 
                                    doc="Forecasted electricity price [€/MWh]")
        
        # Decision variables for states (mutually exclusive)
        model.x_run = pyo.Var(model.T, domain=pyo.Binary, doc="Running at 100% load")
        model.x_shutdown = pyo.Var(model.T, domain=pyo.Binary, doc="Complete shutdown")
        model.x_startup = pyo.Var(model.T, domain=pyo.Binary, doc="In startup transition")
        model.x_shutdown_trans = pyo.Var(model.T, domain=pyo.Binary, doc="In shutdown transition")
        
        # Transition start indicators
        model.y_start_startup = pyo.Var(model.T, domain=pyo.Binary, doc="Start startup sequence")
        model.y_start_shutdown = pyo.Var(model.T, domain=pyo.Binary, doc="Start shutdown sequence")
        
        # State exclusivity constraint
        def state_exclusivity_rule(model, t):
            return (model.x_run[t] + model.x_shutdown[t] + 
                    model.x_startup[t] + model.x_shutdown_trans[t]) == 1
        
        model.state_exclusivity = pyo.Constraint(model.T, rule=state_exclusivity_rule)
        
        # Initial state constraint
        def initial_state_rule(model, t):
            if t == 0:
                # Set initial state based on current plant state
                if plant_state.current_state == 'running':
                    return model.x_run[t] == 1
                elif plant_state.current_state == 'shutdown':
                    return model.x_shutdown[t] == 1
                elif plant_state.current_state == 'startup':
                    return model.x_startup[t] == 1
                elif plant_state.current_state == 'shutdown_trans':
                    return model.x_shutdown_trans[t] == 1
            else:
                return pyo.Constraint.Skip
        
        model.initial_state = pyo.Constraint(model.T, rule=initial_state_rule)
        
        # Transition logic (simplified for MPC horizon)
        def startup_transition_logic_rule(model, t):
            if t == 0:
                # Can only start transitions if plant allows it
                if not plant_state.can_start_transition():
                    return model.y_start_startup[t] == 0
                else:
                    return pyo.Constraint.Skip
            else:
                # Startup can only begin from shutdown
                return model.y_start_startup[t] <= model.x_shutdown[t-1]
        
        def shutdown_transition_logic_rule(model, t):
            if t == 0:
                # Can only start transitions if plant allows it
                if not plant_state.can_start_transition():
                    return model.y_start_shutdown[t] == 0
                else:
                    return pyo.Constraint.Skip
            else:
                # Shutdown can only begin from running
                return model.y_start_shutdown[t] <= model.x_run[t-1]
        
        model.startup_transition_logic = pyo.Constraint(model.T, rule=startup_transition_logic_rule)
        model.shutdown_transition_logic = pyo.Constraint(model.T, rule=shutdown_transition_logic_rule)
        
        # Objective function: maximize profit over horizon
        def objective_rule(model):
            # Methanol revenue (only during running state)
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
            
            # Total profit over horizon
            horizon_profit = (
                methanol_revenue 
                - electricity_cost 
                - co2_cost 
                - variable_opex
            )
            
            return horizon_profit
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        return model
    
    def run_mpc_simulation(self, actual_prices, params, verbose=True):
        """
        Run MPC simulation solving MILP at each time step.
        
        Args:
            actual_prices (list): Actual price time series (for comparison)
            params (dict): Model parameters
            verbose (bool): Print progress
            
        Returns:
            dict: Simulation results
        """
        
        total_hours = len(actual_prices)
        plant_state = PlantState()
        
        # Results tracking
        decisions = []  # Will store current state at each hour
        profits = []
        transition_events = []
        
        # Progress bar setup
        total_weeks = (total_hours + 167) // 168  # Round up for partial weeks
        
        if verbose:
            print(f"Running True MILP-based MPC simulation with shutdown capability...")
            print(f"Total hours: {total_hours}")
            print(f"MPC horizon: 24 hours")
            print(f"Solving {total_hours} MILP problems...")
        
        for hour in range(total_hours):
            # Progress bar - update every 168 hours (1 week)
            if verbose and hour % 168 == 0 and hour > 0:
                completed_weeks = hour // 168
                remaining_weeks = total_weeks - completed_weeks
                progress_bar = '█' * completed_weeks + '.' * remaining_weeks
                if hour == 168:  # First update, no \r needed
                    print(f"Progress: [{progress_bar}] {completed_weeks}/{total_weeks} weeks", end='', flush=True)
                else:  # Subsequent updates, overwrite previous line
                    print(f"\rProgress: [{progress_bar}] {completed_weeks}/{total_weeks} weeks", end='', flush=True)
            
            # Get forecast for next 24 hours using adaptive learning
            horizon_hours = min(24, total_hours - hour)
            
            # Pass observed prices up to current hour for learning
            actual_prices_so_far = actual_prices[:hour] if hour > 0 else []
            forecast_prices = self.get_forecast(hour, actual_prices_so_far, horizon_hours)
            
            # Build MPC model for this horizon
            mpc_model = self.build_mpc_model(forecast_prices, plant_state, params)
            
            # Solve the MPC model
            status = solve_model(mpc_model)
            
            if status not in ['optimal', 'feasible']:
                print(f"Warning: Solver status at hour {hour}: {status}")
                # Keep current state if solver fails
                current_state = plant_state.current_state
            else:
                # Extract first hour decision from solved model
                if pyo.value(mpc_model.x_run[0]) == 1:
                    current_state = 'running'
                elif pyo.value(mpc_model.x_shutdown[0]) == 1:
                    current_state = 'shutdown'
                elif pyo.value(mpc_model.x_startup[0]) == 1:
                    current_state = 'startup'
                elif pyo.value(mpc_model.x_shutdown_trans[0]) == 1:
                    current_state = 'shutdown_trans'
                else:
                    current_state = plant_state.current_state  # Fallback
            
            # Calculate profit for this hour based on current state
            transition_event = 0
            if current_state != plant_state.current_state and plant_state.can_start_transition():
                transition_event = 1
            
            # Calculate hourly profit based on state
            if current_state == 'running':
                profit_hour = (params["M_100"] * params["Price_Methanol"] -
                              params["P_100"] * actual_prices[hour] -
                              params["C_100"] * params["Price_CO2"] -
                              params["OPEX_Variable"])
            elif current_state == 'shutdown':
                profit_hour = (params["M_shutdown"] * params["Price_Methanol"] -
                              params["P_shutdown"] * actual_prices[hour] -
                              params["C_shutdown"] * params["Price_CO2"] -
                              params["OPEX_shutdown_state"])
            elif current_state == 'startup':
                profit_hour = (params["M_startup"] * params["Price_Methanol"] -
                              params["P_startup"] * actual_prices[hour] -
                              params["C_startup"] * params["Price_CO2"] -
                              params["OPEX_startup"])
            elif current_state == 'shutdown_trans':
                profit_hour = (params["M_shutdown_trans"] * params["Price_Methanol"] -
                              params["P_shutdown_trans"] * actual_prices[hour] -
                              params["C_shutdown_trans"] * params["Price_CO2"] -
                              params["OPEX_shutdown"])
            else:
                profit_hour = 0  # Fallback
            
            # Store results
            decisions.append(current_state)
            profits.append(profit_hour)
            transition_events.append(transition_event)
            
            # Update plant state 
            plant_state.update(current_state, params)
        
        # Complete progress bar
        if verbose:
            progress_bar = '█' * total_weeks
            print(f"\rProgress: [{progress_bar}] {total_weeks}/{total_weeks} weeks - Complete!")
        
        # Calculate total metrics
        total_profit = sum(profits)
        total_profit -= params["Annualized_CAPEX"] + params["OPEX_Fixed"]
        
        # Calculate production (only during running state)
        total_production = sum(params["M_100"] if state == 'running' else 0 
                             for state in decisions)
        
        total_transitions = sum(transition_events)
        
        # Capacity factor based on time in running state
        running_hours = sum(1 for state in decisions if state == 'running')
        capacity_factor = running_hours / len(decisions)
        
        results = {
            'strategy': 'True MILP-based MPC with Shutdown',
            'decisions': decisions,
            'profits': profits,
            'total_profit': total_profit,
            'total_production': total_production,
            'total_ramps': total_transitions,
            'capacity_factor': capacity_factor,
            'avg_profit_per_hour': total_profit / total_hours
        }
        
        if verbose:
            print(f"\nMPC Simulation Complete!")
            print(f"  Total Profit: €{total_profit:,.0f}")
            print(f"  Production: {total_production:,.0f} tons")
            print(f"  Transition Events: {total_transitions}")
            print(f"  Capacity Factor: {capacity_factor:.1%}")
            print(f"  Running Hours: {running_hours}/{total_hours}")
            print(f"  Shutdown Hours: {sum(1 for s in decisions if s == 'shutdown')}")
        
        return results


if __name__ == "__main__":
    # Test the true MILP-based MPC
    
    # Load data
    prices, params = load_data()
    if prices is None:
        # Use dummy data for testing
        prices = [50.0] * 168  # One week of constant prices
        params = {
            "P_100": 100.0, "M_100": 8.5, "C_100": 6.2,
            "P_10": 15.0, "M_10": 0.85, "C_10": 0.62,
            "Price_Methanol": 750.0, "Price_CO2": 50.0,
            "Production_Loss_Up": 4.0, "Energy_Penalty_Up": 10.0,
            "Production_Loss_Down": 1.5, "Energy_Penalty_Down": 5.0,
            "OPEX_Variable": 150.0, "T_stab": 3,
            "Annualized_CAPEX": 8.5e6, "OPEX_Fixed": 2.5e6
        }
    
    # Initialize and run MPC
    mpc = ForecastingMPC()
    
    # Test with first 168 hours (one week)
    test_prices = prices[:168]
    results = mpc.run_mpc_simulation(test_prices, params)
    
    print(f"\nResults: {results['strategy']}")
    print(f"Total Profit: €{results['total_profit']:,.0f}")
    print(f"Capacity Factor: {results['capacity_factor']:.1%}")
