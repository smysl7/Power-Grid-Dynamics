import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 读取数据
df = pd.read_csv("Total_Daily_Wind_Power.csv")
df['ds'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'P(MW)': 'y'})

# 定义不同的超参数值
changepoint_prior_values = [0.01, 0.1, 0.5, 1.0]

plt.figure(figsize=(12, 6))

for cp in changepoint_prior_values:
    model = Prophet(changepoint_prior_scale=cp, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    plt.plot(forecast['ds'], forecast['yhat'], label=f'cp_prior={cp}')

plt.scatter(df['ds'], df['y'], color='black', label="Actual Data", alpha=0.5)
plt.xlabel("Date")
plt.ylabel("Predicted Wind Power (MW)")
plt.title("Sensitivity Analysis of Changepoint Prior Scale")
plt.legend()
plt.grid(True)
plt.show()
