import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.forecaster import StanBackendEnum
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = "Total_Daily_Wind_Power.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

df = df.rename(columns={'P(MW)': 'y'})
df['ds'] = df.index

auto_model = pm.auto_arima(df['y'], 
                           seasonal=True,          
                           m=7,                   
                           stepwise=True,         
                           trace=True)         

best_order = auto_model.order         # (p, d, q)
best_seasonal_order = auto_model.seasonal_order  # (P, D, Q, s)

sarima_model = SARIMAX(df['y'], order=best_order, seasonal_order=best_seasonal_order)
sarima_results = sarima_model.fit()

df['SARIMA_Pred'] = sarima_results.predict(start=df.index[0], end=df.index[-1])

prophet_model = Prophet(
    changepoint_prior_scale=0.5,
    changepoint_range=0.90, 
    seasonality_prior_scale=10, 
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)

prophet_model.fit(df)

future = prophet_model.make_future_dataframe(periods=0)
forecast = prophet_model.predict(future)


df['Prophet_Pred'] = forecast['yhat'].values

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['y'], label='Actual', color='black', linestyle='dashed')
plt.plot(df.index, df['SARIMA_Pred'], label='SARIMA Prediction', color='blue')
plt.plot(df.index, df['Prophet_Pred'], label='Prophet Prediction', color='red')
plt.xlabel('Date')
plt.ylabel('Wind Power (MW)')
plt.legend()
plt.show()

mae_sarima = mean_absolute_error(df['y'], df['SARIMA_Pred'])
rmse_sarima = np.sqrt(mean_squared_error(df['y'], df['SARIMA_Pred']))
r2_sarima = r2_score(df['y'], df['SARIMA_Pred'])  
bic_sarima = sarima_results.bic

mae_prophet = mean_absolute_error(df['y'], df['Prophet_Pred'])
rmse_prophet = np.sqrt(mean_squared_error(df['y'], df['Prophet_Pred']))
r2_prophet = r2_score(df['y'], df['Prophet_Pred'])  
bic_prophet = None  

comparison_df = pd.DataFrame({
    'Model': ['SARIMA', 'Prophet'],
    'MAE': [mae_sarima, mae_prophet],
    'RMSE': [rmse_sarima, rmse_prophet],
    'R2': [r2_sarima, r2_prophet],
    'BIC': [bic_sarima, bic_prophet]
})

print("Model Performance Comparison:")
print(comparison_df)
