import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import detrend


np.random.seed(42)

df = pd.read_csv("C:/Users/27240/Desktop/project 3/wind_predictions.csv")
time_series = df["ds"]
power_values = df["Prophet_Pred"]


power_values_detrended = detrend(power_values)

Pk_t = 2 * ((power_values_detrended - np.min(power_values_detrended)) /
            (np.max(power_values_detrended) - np.min(power_values_detrended))) - 1

Pk_t_smooth = gaussian_filter1d(Pk_t, sigma=5)

Pk_t_smooth -= np.mean(Pk_t_smooth)


gamma = 1  
kappa = 1500  


Pk_interp = interp1d(np.linspace(0, 10, len(Pk_t_smooth)), Pk_t_smooth,
                     kind='cubic', fill_value="extrapolate")


t_span = [0, 10]
t_eval = np.linspace(0, 10, 500)  


def swing_equation_5n_fixed(t, theta):
    P = Pk_interp(t)  
    
    theta_fixed = 0 
    thetas = np.concatenate(([theta_fixed], theta[:4]))  
    omegas = theta[4:]

    d_omegas = np.zeros(4)
    d_thetas = omegas.copy()

    d_omegas[0] = P - kappa * (np.sin(thetas[1] - thetas[0]) + np.sin(thetas[1] - thetas[2])) - gamma * omegas[0]
    for i in range(1, 3):
        d_omegas[i] = P - kappa * (np.sin(thetas[i+1] - thetas[i]) + np.sin(thetas[i+1] - thetas[i+2])) - gamma * omegas[i]
    d_omegas[3] = P - kappa * np.sin(thetas[4] - thetas[3]) - gamma * omegas[3]

    return np.concatenate((d_thetas, d_omegas))


y0_5 = np.random.uniform(-np.pi/2, np.pi/2, 8) 


sol_5nodes = solve_ivp(swing_equation_5n_fixed, t_span, y0_5, t_eval=t_eval, method='RK45')

theta_5nodes = (sol_5nodes.y[:4] + np.pi) % (2 * np.pi) - np.pi
omega_5nodes = sol_5nodes.y[4:]

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.plot(t_eval, theta_5nodes[i], label=f'Theta {i+2}')
plt.title('Phase Angle Evolution for n=5 with Normalized Prophet Pk(t)')
plt.xlabel('Time (Days)')
plt.ylabel('Phase Angle (θ)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.plot(t_eval, omega_5nodes[i], label=f'Omega {i+2}')
plt.title('Frequency Evolution for n=5 with Normalized Prophet Pk(t)')
plt.xlabel('Time (Days)')
plt.ylabel('Frequency (dθ/dt)')
plt.legend()
plt.show()

def swing_equation_10n_fixed(t, theta):
    P = Pk_interp(t)
    
    theta_fixed = 0
    thetas = np.concatenate(([theta_fixed], theta[:9])) 
    omegas = theta[9:]

    d_omegas = np.zeros(9)
    d_thetas = omegas.copy()

    d_omegas[0] = P - kappa * (np.sin(thetas[1] - thetas[0]) + np.sin(thetas[1] - thetas[2])) - gamma * omegas[0]
    for i in range(1, 8):
        d_omegas[i] = P - kappa * (np.sin(thetas[i+1] - thetas[i]) + np.sin(thetas[i+1] - thetas[i+2])) - gamma * omegas[i]
    d_omegas[8] = P - kappa * np.sin(thetas[9] - thetas[8]) - gamma * omegas[8]

    return np.concatenate((d_thetas, d_omegas))

y0_10 = np.random.uniform(-np.pi/2, np.pi/2, 18)  

sol_10nodes = solve_ivp(swing_equation_10n_fixed, t_span, y0_10, t_eval=t_eval, method='RK45')

theta_10nodes = (sol_10nodes.y[:9] + np.pi) % (2 * np.pi) - np.pi
omega_10nodes = sol_10nodes.y[9:]

plt.figure(figsize=(10, 5))
for i in range(9):
    plt.plot(t_eval, theta_10nodes[i], label=f'Theta {i+2}')
plt.title('Phase Angle Evolution for n=10 with Normalized Prophet Pk(t)')
plt.xlabel('Time (Days)')
plt.ylabel('Phase Angle (θ)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for i in range(9):
    plt.plot(t_eval, omega_10nodes[i], label=f'Omega {i+2}')
plt.title('Frequency Evolution for n=10 with Normalized Prophet Pk(t)')
plt.xlabel('Time (Days)')
plt.ylabel('Frequency (dθ/dt)')
plt.legend()
plt.show()
