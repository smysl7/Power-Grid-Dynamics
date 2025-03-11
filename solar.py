# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:01:25 2025

@author: 27240
"""
#%%Data description
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_hour = pd.read_csv("solar_hour.csv")
df_month = pd.read_csv("solar_month.csv")

df_hour.rename(columns={'Row Labels': 'Hour'}, inplace=True)
df_month.rename(columns={'Row Labels': 'Month'}, inplace=True)

df_hour_long = df_hour.melt(id_vars=['Hour'], var_name='Location', value_name='Solar Energy')
df_month_long = df_month.melt(id_vars=['Month'], var_name='Location', value_name='Solar Energy')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(ax=axes[0], x='Month', y='Solar Energy', data=df_month_long, color="lightblue")
axes[0].set_title("(A) Solar Energy by Month")
axes[0].set_xlabel("Months")
axes[0].set_ylabel("Solar Energy (W/m² or kWh/m²)")

sns.boxplot(ax=axes[1], x='Hour', y='Solar Energy', data=df_hour_long, color="lightblue")
axes[1].set_title("(B) Solar Energy by Hour")
axes[1].set_xlabel("Hours")
axes[1].set_ylabel("Solar Energy (W/m² or kWh/m²)")

plt.tight_layout()
plt.show()
#%%Find optimal p and q
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/27240/Desktop/project 3/solar_pq.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])  

df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
df = df.dropna(subset=["Hour"])
df["Hour"] = df["Hour"].astype(int)
df = df[df["Hour"].between(0, 23)]

df.sort_values(["Date","Hour"], inplace=True)

target_col = "SolarEnergy"
def find_best_pq(series, max_p=3, max_q=3):
    best_score, best_cfg = float("inf"), None
    for p,q in itertools.product(range(max_p+1), range(max_q+1)):
        try:
            model = ARIMA(series, order=(p,0,q))
            result = model.fit()
            bic = result.bic
            if bic < best_score:
                best_score, best_cfg = bic, (p, q)
        except:
            continue
    return best_cfg

results = {}
for hour, group_df in df.groupby("Hour"):
    group_df = group_df.sort_values("Date").set_index("Date")
    
    series = group_df[target_col].dropna()
    if len(series) < 5:
        results[hour] = (None, None)
    else:
        results[hour] = find_best_pq(series, max_p=3, max_q=3)

df_results = pd.DataFrame(
    {"Hour": h, "p": (cfg or (None,None))[0], "q": (cfg or (None,None))[1]} 
    for h,cfg in results.items()
).sort_values("Hour")

print(df_results)


#%%Comparison
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("C:/Users/27240/Desktop/project 3/solar_hour.csv")
df.rename(columns={"Row Labels": "hour"}, inplace=True)
df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(method="ffill")

locations = ["Easthill Road", "Elm Crescent", "Forest Road", "Maple Drive East", "YMCA", "Grand Total"]

#Stochastic Model
def stochastic_model(df, actual_cols, T=24, dt=1, mu=0.02, sigma=0.1, A=100):
    df_stochastic = df.copy(deep=True)
    df_stochastic["time_index"] = np.arange(len(df))
    
    for col in actual_cols:
        P_pred = np.zeros(len(df))
        P_pred[0] = df[col].iloc[0]
        W = np.random.normal(0, np.sqrt(dt), len(df))
        
        for n in range(1, len(df)):
            periodic_term = A * np.sin(2 * np.pi * df_stochastic["time_index"].iloc[n] / T)
            P_pred[n] = P_pred[n-1] + mu * dt + periodic_term * dt + sigma * W[n]
        
        df_stochastic[f"{col}_predicted"] = P_pred

    df_stochastic.fillna(df.mean(), inplace=True)  
    return df_stochastic

#Persistence Model
def persistence_model(df, actual_cols, window_size=3):
    df_persistence = df.copy(deep=True)
    for col in actual_cols:
        df_persistence[f"{col}_predicted"] = df[col].rolling(window=window_size, min_periods=1).mean().fillna(df[col].mean())
    return df_persistence

#ARMA Model
def auto_select_arma(df, actual_col):
    best_score, best_cfg = float("inf"), None
    best_model = None

    for p, q in itertools.product(range(3), range(3)):  
    
        try:
            model = ARIMA(df[actual_col], order=(p, 0, q))
            model_fit = model.fit()
            score = model_fit.bic  

            if score < best_score:
                best_score, best_cfg = score, (p, q)
                best_model = model_fit
        except:
            continue 

    return best_model, best_cfg

def arma_model(df, actual_cols):
    df_arma = df.copy(deep=True)
    for col in actual_cols:
        best_arma_model, best_cfg = auto_select_arma(df.copy(), col)
        df_arma[f"{col}_predicted"] = best_arma_model.predict(start=1, end=len(df), dynamic=False).fillna(df[col].mean())
    
    return df_arma

def calculate_errors(df, actual_cols):
    mae_values, rmse_values = [], []
    for col in actual_cols:
        if f"{col}_predicted" in df.columns:
            df_clean = df[[col, f"{col}_predicted"]].dropna()
            mae_values.extend(np.abs(df_clean[col] - df_clean[f"{col}_predicted"]).values)
            rmse_values.extend((df_clean[col] - df_clean[f"{col}_predicted"])**2)
    
    return mae_values, np.sqrt(np.mean(rmse_values))  

df_persistence = persistence_model(df, locations)
df_stochastic = stochastic_model(df, locations)
df_arma = arma_model(df, locations)

mae_persistence, rmse_persistence = calculate_errors(df_persistence, locations)
mae_stochastic, rmse_stochastic = calculate_errors(df_stochastic, locations)
mae_arma, rmse_arma = calculate_errors(df_arma, locations)


mae_values = [np.mean(mae_stochastic), np.mean(mae_persistence), np.mean(mae_arma)]
rmse_values = [rmse_stochastic, rmse_persistence, rmse_arma]
models = ["Stochastic", "Persistence", "ARMA"]

error_df = pd.DataFrame({"Model": models, "MAE": mae_values, "RMSE": rmse_values})

  
mae_df = pd.DataFrame({
    "Model": ["Stochastic"] * len(mae_stochastic) + 
             ["Persistence"] * len(mae_persistence) + 
             ["ARMA"] * len(mae_arma),
    "MAE": np.concatenate([mae_stochastic, mae_persistence, mae_arma])
})


plt.figure(figsize=(10, 6), dpi=150)


error_df_melted = error_df.melt(id_vars="Model", var_name="Metric", value_name="Error Value")

ax = sns.barplot(
    x="Model", y="Error Value", hue="Metric", data=error_df_melted, 
    palette=["darkorange", "royalblue"], edgecolor="black"
)


for p in ax.patches:
    height_adjustment = -200 if p.get_height() > 3000 else -50  
    ax.annotate(f"{p.get_height():.1f}", 
                (p.get_x() + p.get_width() / 2., p.get_height() + height_adjustment),  
                ha='center', va='top', fontsize=12, fontweight="bold", color="black", xytext=(0, -5), 
                textcoords="offset points")

plt.xlabel("Model", fontsize=14, fontweight="bold")
plt.ylabel("Error Value", fontsize=14, fontweight="bold")
plt.xticks(rotation=0, fontsize=12, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Metric", fontsize=12)

plt.show()

mae_df = pd.DataFrame({
    "Model": ["Stochastic"] * len(mae_stochastic)+ 
             ["Persistence"] * len(mae_persistence) + 
             ["ARMA"] * len(mae_arma),
    "MAE": np.concatenate([mae_stochastic,mae_persistence , mae_arma])
})
plt.figure(figsize=(10, 6), dpi=150) 


sns.boxplot(x="Model", y="MAE", data=mae_df, palette=["blue", "orange", "green"], linewidth=2.5)


plt.xlabel("Model", fontsize=14)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()



def compute_bic(sigma2, num_params, num_samples):
    return num_samples * np.log(sigma2) + num_params * np.log(num_samples)

bic_values = []
for name, df_model in zip(["Stochastic", "Persistence", "ARMA"], [df_stochastic,df_persistence , df_arma]):
    residuals = []
    for col in locations:
        if f"{col}_predicted" in df_model.columns:
            residuals.append((df_model[col] - df_model[f"{col}_predicted"]).dropna().values)

    if residuals:
        residuals = np.concatenate(residuals)
        sigma2 = np.var(residuals)
    else:
        sigma2 = 1e-6

    num_params = {"Stochastic": 1, "Persistence": 2, "ARMA": 3}[name]
    bic = compute_bic(sigma2, num_params, len(residuals))
    bic_values.append((name, bic))


bic_df = pd.DataFrame(bic_values, columns=["Model", "BIC"])


plt.figure(figsize=(10,6), dpi=150)
ax = sns.barplot(x="Model", y="BIC", data=bic_df, palette=["blue", "green", "red"], edgecolor="black")


for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=12, fontweight="bold", color="black", 
                xytext=(0,2), textcoords="offset points")  

plt.xlabel("Model", fontsize=14, fontweight="bold")
plt.ylabel("BIC", fontsize=14, fontweight="bold")

plt.show()

def plot_actual_vs_predicted_subplot(df_persistence, df_stochastic, df_arma, time_col, actual_cols, locations):
    num_locations = len(locations)
    cols=2
    fig, axes = plt.subplots(nrows=3, ncols=cols, figsize=(15,8), sharex=True)
    for i, location in enumerate(locations):
        row, col = divmod(i, cols)  
        ax = axes[row, col]
        ax.plot(df_persistence[time_col], df_persistence[location], label="Actual", color="black", linewidth=2)    
        ax.plot(df_stochastic[time_col], df_stochastic[f"{location}_predicted"], label="Model 1", linestyle="dashed", color="green")
        ax.plot(df_persistence[time_col], df_persistence[f"{location}_predicted"], label="Model 2", linestyle="dashed", color="yellow")
        ax.plot(df_arma[time_col], df_arma[f"{location}_predicted"], label="Model 3", linestyle="dashed", color="red")
        
        ax.set_ylabel("Solar Output")
        ax.set_title(f"{location}")
        ax.legend()
        ax.grid()

    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted_subplot(
    df_persistence, df_stochastic, df_arma,   
    "hour",
    locations,  
    locations  
)

error_persistence_all = []
error_stochastic_all = []
error_arma_all = []

for loc in locations:
    error_persistence = df[loc] - df_persistence[f"{loc}_predicted"]
    error_stochastic = df[loc] - df_stochastic[f"{loc}_predicted"]
    error_arma = df[loc] - df_arma[f"{loc}_predicted"]

    error_persistence_all.extend(error_persistence.dropna().values)
    error_stochastic_all.extend(error_stochastic.dropna().values)
    error_arma_all.extend(error_arma.dropna().values)

errors_sorted_persistence = np.sort(error_persistence_all)
errors_sorted_stochastic = np.sort(error_stochastic_all)
errors_sorted_arma = np.sort(error_arma_all)


cdf_persistence = np.linspace(0, 1, len(errors_sorted_persistence))
cdf_stochastic = np.linspace(0, 1, len(errors_sorted_stochastic))
cdf_arma = np.linspace(0, 1, len(errors_sorted_arma))

plt.figure(figsize=(8,6), dpi=150)


plt.fill_between(errors_sorted_persistence, cdf_persistence, alpha=0.3, hatch="///", color="blue", edgecolor="blue")
plt.fill_between(errors_sorted_stochastic, cdf_stochastic, alpha=0.3, hatch="\\\\", color="green", edgecolor="green")
plt.fill_between(errors_sorted_arma, cdf_arma, alpha=0.3, hatch="xx", color="red", edgecolor="red")


plt.plot(errors_sorted_persistence, cdf_persistence, color="blue", linewidth=2, linestyle="dashed", label="Persistence Model")
plt.plot(errors_sorted_stochastic, cdf_stochastic, color="green", linewidth=2, linestyle="solid", label="Stochastic Model")
plt.plot(errors_sorted_arma, cdf_arma, color="red", linewidth=2, linestyle="dotted", label="ARMA Model")


plt.xlabel("Forecasting Error (Actual - Predicted)", fontsize=14, fontweight="bold", style="italic")
plt.ylabel("CDF", fontsize=14, fontweight="bold", style="italic")


plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

plt.show()
#%%Grid function
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def generate_ar2_series(alpha0, alpha1, alpha2, sigma, T, dt):
    n_steps = int(T/dt)
    t_vals = np.linspace(0, T, n_steps+1)
    P = np.zeros(n_steps+1)
    if n_steps > 0:
        P[1] = 0.0

    for k in range(1, n_steps):
        e_k = np.random.normal(loc=0.0, scale=sigma)
        P[k+1] = alpha0 + alpha1*P[k] + alpha2*P[k-1] + e_k
    
    return t_vals, P

def create_ring_adjacency(n):
    A = np.zeros((n,n), dtype=float)
    for i in range(n):
        A[i, (i+1) % n] = 1
        A[(i+1) % n, i] = 1
    return A

def swing_equation_ode(t, y, n, A, kappa, gamma, interp_funcs):
    theta = y[:n]
    omega = y[n:]
    
    dtheta_dt = omega
    domega_dt = np.zeros(n)
    
    for k in range(n):
        Pk_t = interp_funcs[k](t)
        coupling_sum = 0.0
        for l in range(n):
            if A[k,l] != 0:
                coupling_sum += A[k,l]*np.sin(theta[k] - theta[l])
        domega_dt[k] = Pk_t - kappa*coupling_sum - gamma*omega[k]
    
    return np.concatenate([dtheta_dt, domega_dt])

def run_simulation(n, T_sim=10.0, dt=0.01, alpha0=0.0, alpha1=0.8, alpha2=-0.3, sigma=0.02):
    P_arr_list = []
    t_vals = None
    for k in range(n):
        t_tmp, P_tmp = generate_ar2_series(alpha0, alpha1, alpha2, sigma, T_sim, dt)
        if t_vals is None:
            t_vals = t_tmp  
        P_arr_list.append(P_tmp)
    for idx in range(len(t_vals)):
        mean_pt = np.mean([P_arr_list[k][idx] for k in range(n)])
        for k in range(n):
            P_arr_list[k][idx] -= mean_pt
    interp_funcs = []
    for k in range(n):
        f_interp = interp1d(t_vals, P_arr_list[k], kind='nearest', fill_value='extrapolate')
        interp_funcs.append(f_interp)
    
    A = create_ring_adjacency(n)
    kappa = 2.0
    gamma = 0.8
    
    theta_init = 0.01*np.random.randn(n)   
    omega_init = np.zeros(n)
    y0 = np.concatenate([theta_init, omega_init])
    
    sol = solve_ivp(
        lambda t,y: swing_equation_ode(t,y,n,A,kappa,gamma,interp_funcs),
        [0, T_sim],
        y0,
        t_eval=np.linspace(0,T_sim,500),
        method='RK45'
    )
    
    t_sol = sol.t
    theta_sol = sol.y[:n, :]   
    omega_sol = sol.y[n:, :]   
    
    
    for i in range(len(t_sol)):
        ref_angle = theta_sol[0, i]
        theta_sol[:, i] -= ref_angle
    
    freq_sol = 50.0 + omega_sol/(2.0*np.pi)
    
    plt.figure(figsize=(8, 5))
    for k in range(n):
        plt.plot(t_sol, theta_sol[k,:], label=f"$\\theta_{{{k+1}}}$")
    plt.title(f"Phase Angles({n} Nodes)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    for k in range(n):
        plt.plot(t_sol, freq_sol[k,:], label=f"Node {k+1}")
    plt.title(f"Frequency({n} Nodes)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_simulation(n=2, T_sim=5.0, dt=0.01)
    run_simulation(n=5, T_sim=10.0, dt=0.01)
    run_simulation(n=10, T_sim=60.0, dt=0.05)


