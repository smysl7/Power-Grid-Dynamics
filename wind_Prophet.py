import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from itertools import product

# Load the wind power data
file_path = "Total_Daily_Wind_Power.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Rename columns for Prophet compatibility
df = df.rename(columns={'Date': 'ds', 'P(MW)': 'y'})

# Define parameter grid for optimization
param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],  # Flexibility of trend
    'changepoint_range': [0.8, 0.9],  # Portion of data for trend change detection
    'seasonality_prior_scale': [1, 5, 10],  # Strength of seasonal effect
    'seasonality_mode': ['additive', 'multiplicative'],  # Seasonal modeling approach
    'yearly_seasonality': [True, False],  # Test both enabling and disabling yearly seasonality
    'weekly_seasonality': [True, False]  # Test both enabling and disabling weekly seasonality
}

# Create all parameter combinations
all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

best_mae = float('inf')
best_params = None

# Train and evaluate models for each parameter combination
for params in all_params:
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        changepoint_range=params['changepoint_range'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        stan_backend="CMDSTANPY"
    )
    
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Evaluate MAE on training data
    y_true = df['y']
    y_pred = forecast['yhat'][:len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    
    # Update best parameters if current MAE is lower
    if mae < best_mae:
        best_mae = mae
        best_params = params

# Output best parameters and corresponding MAE
print("Best Parameters:", best_params)
print("Best MAE:", best_mae)
