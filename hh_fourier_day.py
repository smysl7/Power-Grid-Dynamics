import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = r"D:\hungrywatermelon\OneDrive - The University of Nottingham Ningbo China\桌面\Small LCL Data\LCL-June2015v2_1.csv"
df = pd.read_csv(file_path, encoding="utf-8")

df.columns = df.columns.str.strip()
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')


df.rename(columns={'KWH/hh (per half hour)': 'KWH'}, inplace=True)
df['KWH'] = pd.to_numeric(df['KWH'], errors='coerce')
df.dropna(subset=['KWH'], inplace=True)

df['HourOfDay'] = df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0

daily_pattern = df.groupby('HourOfDay')['KWH'].mean().reset_index()


T = 24           
N = 4            
t = daily_pattern['HourOfDay'].values


X = np.column_stack([np.ones_like(t)] +
                    [np.cos(2 * np.pi * n * t / T) for n in range(1, N+1)] +
                    [np.sin(2 * np.pi * n * t / T) for n in range(1, N+1)])
y = daily_pattern['KWH'].values

coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print("Fourier Series Coefficients:")
print(coeffs)

y_pred = X.dot(coeffs)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nError Metrics on Average Daily Pattern:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")


expression = f"P_k(t) = {coeffs[0]:.4f}"
for n in range(1, N+1):
    expression += f" + ({coeffs[n]:.4f}) cos(2π*{n}*t/24)"
    expression += f" + ({coeffs[N+n]:.4f}) sin(2π*{n}*t/24)"
print("\nFourier Series Model Expression:")
print(expression)


t_fine = np.linspace(0, 24, 240) 
X_fine = np.column_stack([np.ones_like(t_fine)] +
                         [np.cos(2 * np.pi * n * t_fine / T) for n in range(1, N+1)] +
                         [np.sin(2 * np.pi * n * t_fine / T) for n in range(1, N+1)])
y_fit = X_fine.dot(coeffs)

plt.figure(figsize=(10, 6))
plt.scatter(t, y, color='forestgreen', s=20, alpha=0.7, label="Actual Power Consumption", edgecolors='k')
plt.plot(t_fine, y_fit, color='darkorange', linewidth=2.5, label=f"Prediction", antialiased=True)
plt.xlabel("Hour of Day")
plt.ylabel("Average Power Consumption (KWH)")
plt.title("Daily Trend--Fourier Series Model Fit")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()