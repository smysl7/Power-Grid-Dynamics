import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = r"D:\hungrywatermelon\OneDrive - The University of Nottingham Ningbo China\桌面\Small LCL Data\LCL-June2015v2_1.csv"
df = pd.read_csv(file_path, encoding="utf-8")

df.columns = df.columns.str.strip()

df = df[df['LCLid'].between("MAC000037", "MAC000068")]

df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

df.rename(columns={'KWH/hh (per half hour)': 'KWH'}, inplace=True)

df['KWH'] = pd.to_numeric(df['KWH'], errors='coerce')
df.dropna(subset=['KWH'], inplace=True)


family_ids = df['LCLid'].unique()
train_families = random.sample(list(family_ids), 16)  
test_families = [fam for fam in family_ids if fam not in train_families]  

df_train = df[df["LCLid"].isin(train_families)]
train_avg = df_train.groupby("DateTime")["KWH"].mean().reset_index()


p = 12  

y_train = train_avg["KWH"].values
nobs = len(y_train)
p = min(p, nobs - 1)  

ar_model = AutoReg(y_train, lags=p).fit()

ar_coeffs = ar_model.params

ar_equation = f"P_k(t) = {ar_coeffs[0]:.6f} + " + " + ".join([
    f"({ar_coeffs[i+1]:.6f}) * P_k(t-{i+1})" for i in range(p)
])
print("\n📌 Higher-Order AR Model Expression (p={p}):")
print(ar_equation)


results = []

for test_household_id in test_families:
    df_test = df[df["LCLid"] == test_household_id]
    test_avg = df_test.groupby("DateTime")["KWH"].mean().reset_index()
    
    y_test = test_avg["KWH"].values
    
    
    if len(y_test) <= p:
        print(f" Not enough data points for {test_household_id}, skipping")
        continue
    
    y_pred = ar_model.predict(start=p, end=len(y_test) - 1)
    
    mae = mean_absolute_error(y_test[p:], y_pred)
    mse = mean_squared_error(y_test[p:], y_pred)
    r2 = r2_score(y_test[p:], y_pred)
    
    results.append({
        "Household": test_household_id,
        "MAE": mae,
        "MSE": mse,
        "R²": r2
    })
    
    
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(y_test[p:])), y_test[p:], color='red', s=10, label="Actual Power Consumption")
    plt.plot(y_pred, color='blue', linewidth=2, linestyle="dashed", label="AR Model Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Power Consumption (kWh)")
    plt.title(f"AR Model Prediction (p={p}) for {test_household_id}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

results_df = pd.DataFrame(results)

print("\n📊 AR Model Performance on Test Households:")
print(results_df)

plt.figure(figsize=(10, 5))
plt.bar(results_df["Household"], results_df["R²"], color="blue", alpha=0.7)
plt.axhline(y=0, color='black', linestyle="--")  
plt.xlabel("Household ID")
plt.ylabel("R² Score")
plt.title(f"AR Model Performance (p={p}) on 16 Test Households")
plt.xticks(rotation=90)  
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()