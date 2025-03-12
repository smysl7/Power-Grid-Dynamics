import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.rcParams['font.family'] = 'Arial'


plt.style.use('seaborn-whitegrid')


file_path = r"D:\hungrywatermelon\OneDrive - The University of Nottingham Ningbo China\桌面\Small LCL Data\LCL-June2015v2_1.csv"
df = pd.read_csv(file_path, encoding="utf-8")
df.columns = df.columns.str.strip()  
print("Column names:", df.columns)

df = df[df['LCLid'].between("MAC000037", "MAC000068")]


df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

df.rename(columns={'KWH/hh (per half hour)': 'KWH'}, inplace=True)

df['KWH'] = pd.to_numeric(df['KWH'], errors='coerce').astype('float32')

df = df.dropna(subset=['KWH'])

family_ids = df['LCLid'].unique()
train_families = random.sample(list(family_ids), 16)
test_families = [fam for fam in family_ids if fam not in train_families]

train_data = df[df['LCLid'].isin(train_families)].copy()
train_data['Hour'] = train_data['DateTime'].dt.hour
train_avg = train_data.groupby("Hour")["KWH"].mean().reset_index()

x_train = train_avg["Hour"].values  
y_train = train_avg["KWH"].values


poly_coeffs = np.polyfit(x_train, y_train, 5)
poly_equation = "P_k(t) = " + " + ".join([f"{coef:.6f} * t^{5-i}" for i, coef in enumerate(poly_coeffs)])
print("Polynomial fit expression:")
print(poly_equation)

def polynomial(t, coeffs):
    return np.polyval(coeffs, t)

test_data = df[df['LCLid'].isin(test_families)].copy()
test_data['Hour'] = test_data['DateTime'].dt.hour
test_avg = test_data.groupby("Hour")["KWH"].mean().reset_index()

x_test = test_avg["Hour"].values
y_test = test_avg["KWH"].values

x_dense = np.linspace(0, 24, 200)
y_dense = polynomial(x_dense, poly_coeffs)

mae = mean_absolute_error(y_test, polynomial(x_test, poly_coeffs))
mse = mean_squared_error(y_test, polynomial(x_test, poly_coeffs))
r2 = r2_score(y_test, polynomial(x_test, poly_coeffs))
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(10, 6), dpi=120)

plt.scatter(x_test, y_test, color='forestgreen', s=20, alpha=0.7, 
            label="Actual Power Consumption", edgecolors='k')

plt.plot(x_dense, y_dense, color='darkorange', linewidth=2.5, 
         label="Prediction", antialiased=True)

plt.xlabel("Hour of Day", fontsize=14, fontweight='bold')
plt.ylabel("Average Power Consumption (kWh)", fontsize=14, fontweight='bold')

plt.title("Daily Trend Polynomial Fit (Household Usage)", fontsize=16, fontweight='bold')

plt.legend(fontsize=12, loc="upper right", frameon=True)


plt.grid(True, linestyle="--", alpha=0.7)

plt.xticks(np.arange(0, 25, 2), fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

plt.show()
