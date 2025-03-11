import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from prophet import Prophet

df = pd.read_csv("Total_Daily_Wind_Power.csv")

df.rename(columns={"Date": "ds", "P(MW)": "y"}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],  
    'changepoint_range': [0.8, 0.9, 0.95],  
    'seasonality_prior_scale': [1.0, 5.0, 10.0],  
    'seasonality_mode': ['additive', 'multiplicative'], 
    'seasonality_type': ['yearly_only', 'weekly_only']  
}

best_params = {
    'changepoint_prior_scale': 0.5,
    'changepoint_range': 0.9,
    'seasonality_prior_scale': 10,
    'seasonality_mode': 'multiplicative',
    'seasonality_type': 'yearly_only'
}

results = []

for cps in param_grid['changepoint_prior_scale']:
    for cpr in param_grid['changepoint_range']:
        for sps in param_grid['seasonality_prior_scale']:
            for smode in param_grid['seasonality_mode']:
                for stype in param_grid['seasonality_type']:
                    
                    prophet_model = Prophet(
                        changepoint_prior_scale=cps if cps != best_params['changepoint_prior_scale'] else best_params['changepoint_prior_scale'],
                        changepoint_range=cpr if cpr != best_params['changepoint_range'] else best_params['changepoint_range'],
                        seasonality_prior_scale=sps if sps != best_params['seasonality_prior_scale'] else best_params['seasonality_prior_scale'],
                        seasonality_mode=smode if smode != best_params['seasonality_mode'] else best_params['seasonality_mode'],
                        yearly_seasonality=True if stype == 'yearly_only' else False,
                        weekly_seasonality=True if stype == 'weekly_only' else False
                    )

                    prophet_model.fit(df)
                    future = prophet_model.make_future_dataframe(periods=0)
                    forecast = prophet_model.predict(future)
                    
                    rmse_val = np.sqrt(mean_squared_error(df['y'], forecast['yhat']))

                    results.append({
                        'changepoint_prior_scale': cps,
                        'changepoint_range': cpr,
                        'seasonality_prior_scale': sps,
                        'seasonality_mode': smode,
                        'seasonality_type': stype,
                        'RMSE': rmse_val
                    })

df_results = pd.DataFrame(results)

param_list_continuous = ['changepoint_prior_scale', 'changepoint_range', 'seasonality_prior_scale']

#Permutation MSE
def permutation_mse_param(param_name):
    grouped_mean = df_results.groupby(param_name)['RMSE'].mean()
    global_mean = df_results['RMSE'].mean()
    diff = (grouped_mean - global_mean).abs()
    return diff.mean()

perm_mse_vals = [permutation_mse_param(p) for p in param_list_continuous]

#Value-varying Sensitivity
def sensitivity_param(param_name):
    grouped_std = df_results.groupby(param_name)['RMSE'].std().fillna(0)
    return grouped_std.mean()

sens_vals = [sensitivity_param(p) for p in param_list_continuous]

#Pearson Correlation
def correlation_param(param_name):
    return df_results[param_name].corr(df_results['RMSE'])

corr_vals = [correlation_param(p) for p in param_list_continuous]


def plot_radar_chart(labels, values, title, color):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.08)
    plt.show()

labels_continuous = ["Changepoint prior scale", "Changepoint range", "Seasonality prior scale"]

plot_radar_chart(labels_continuous, perm_mse_vals, "Permutation MSE", "blue")
plot_radar_chart(labels_continuous, sens_vals, "Value Sensitivity", "green")
plot_radar_chart(labels_continuous, corr_vals, "Pearson Correlation", "red")

#Boxplot
plt.figure(figsize=(12,4))

# Seasonality mode
plt.subplot(1,2,1)
sns.boxplot(x="seasonality_mode", y="RMSE", data=df_results)
plt.title("RMSE Distribution for seasonality_mode")

# Yearly vs Weekly Seasonality
plt.subplot(1,2,2)
sns.boxplot(x="seasonality_type", y="RMSE", data=df_results)
plt.title("RMSE Distribution for yearly_seasonality vs. weekly_seasonality")

plt.tight_layout()
plt.show()
