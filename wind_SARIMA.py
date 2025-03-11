import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

file_path = "Total_Daily_Wind_Power.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])  
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['P(MW)'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Power (MW)')
plt.grid(True)
plt.show()

# ADF
adf_test = adfuller(df['P(MW)'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print(f"Critical Values: {adf_test[4]}")

# Plot ACF and PACF with grid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(df['P(MW)'], ax=axes[0], lags=40)
axes[0].set_title("Autocorrelation Function (ACF)")
axes[0].grid(True) 

plot_pacf(df['P(MW)'], ax=axes[1], lags=40, method="ywm")
axes[1].set_title("Partial Autocorrelation Function (PACF)")
axes[1].grid(True) 

plt.show()
