#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:03:12 2025

@author: maisieamos
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.interpolate import interp1d

def swing_equation_2n(t, theta):
    #set gamma
    gamma = 1
    #set kappa
    kappa = 999.80
    #set Pk_t 
    #P =  0.000119*(t**4) - 0.00240*(t**3) + 0.021246*(t**2) - 0.065211*t + 0.203511
    P = (1.3394 * (10**-8))*(t**5) - (7.6842 * (10** -6))*(t**4)+(1.5999 * (10**-3))*(t**3)-(0.1437 * (t**2)) +4.82278*t - 6.0622
    
    theta_1, theta_2, freq_1, freq_2 = theta
    
    diff_freq_1 = P - (kappa*np.sin(theta_1 - theta_2)) - gamma*freq_1
    diff_freq_2 = P - (kappa*np.sin(theta_2 - theta_1)) - gamma*freq_2
    diff_theta_1 = freq_1
    diff_theta_2 = freq_2
    
    return diff_theta_1, diff_theta_2, diff_freq_1, diff_freq_2

sol_2nodes = solve_ivp(swing_equation_2n, [0, 10], [5, 4, 0 , 0], dense_output=True)
t = np.linspace(0, 10, 1000)
theta_2nodes = sol_2nodes.sol(t)
theta_2nodes = theta_2nodes[0:2]
plt.plot(t, theta_2nodes.T)
plt.xlabel('t')
plt.legend(['theta 1', 'theta 2'], shadow = True)
plt.show()

#%%
def finding_kappa(N):
    
    kappa = 999
    
    for n in np.arange(N):
        sol_kappa = solve_ivp(swing_equation_2n, [0,60], [5, 4, 0, 0], dense_output=True, args=(kappa, 1))
        t = np.linspace(0, 60, 10)
        theta_kappa = sol_kappa.sol(t)
        theta_1 = theta_kappa[0]
        theta_2 = theta_kappa[1]
        
        if abs(theta_1[-1] - theta_2[-1]) < (10**(-3)):
            break
        else:
            kappa = kappa + 0.01
    print(theta_1)
    print(theta_2)
            
    return kappa

#%%

def swing_equation_5n(t, theta):
    #set gamma
    gamma = 1
    #set kappa
    kappa = 999.80
    #set Pk_t 
    #P = 0.000119*(t**4) - 0.00240*(t**3) + 0.021246*(t**2) - 0.065211*t + 0.203511 
    #P = (1.3394 * (10**-8))*(t**5) - (7.6842 * (10** -6))*(t**4)+(1.5999 * (10**-3))*(t**3)-(0.1437 * (t**2)) +4.82278*t - 6.0622
    P = (1.466 * (10**-5))*(t**3)-(0.00381 * (t**2)) +0.0474*t - 32.2949
    
    theta_1, theta_2, theta_3, theta_4, theta_5, freq_1, freq_2, freq_3, freq_4, freq_5 = theta
    
    diff_freq_1 = P - kappa*np.sin(theta_1 - theta_2) - gamma*freq_1
    diff_freq_2 = P - kappa*(np.sin(theta_2 - theta_1) + np.sin(theta_2 -theta_3)) - gamma*freq_2
    diff_freq_3 = P - kappa*(np.sin(theta_3 - theta_2) + np.sin(theta_3 - theta_4)) - gamma*freq_3
    diff_freq_4 = P - kappa*(np.sin(theta_4 - theta_3) + np.sin(theta_4 - theta_5)) - gamma*freq_4
    diff_freq_5 = P - kappa*np.sin(theta_5 - theta_4) - gamma*freq_5
    diff_theta_1 = freq_1
    diff_theta_2 = freq_2
    diff_theta_3 = freq_3
    diff_theta_4 = freq_4
    diff_theta_5 = freq_5
    
    return diff_theta_1, diff_theta_2, diff_theta_3, diff_theta_4, diff_theta_5, diff_freq_1, diff_freq_2, diff_freq_3, diff_freq_4, diff_freq_5

y0 = np.zeros(5)
for i in np.arange(5):
    y0[i] = np.random.uniform(-np.pi, np.pi)

sol_5nodes = solve_ivp(swing_equation_5n, [0, np.pi], [y0[0], y0[1], y0[2], y0[3], y0[4], 0, 0, 0, 0 , 0], dense_output=True)
t = np.linspace(0, np.pi, 1000)
theta_5nodes = sol_5nodes.sol(t)
phaseangle_5nodes = theta_5nodes[0:5]
omega_5nodes = theta_5nodes[5:10]

plt.plot(t, phaseangle_5nodes.T)
plt.title('Phase Angles with 5 Nodes')
plt.xlabel('t')
plt.legend(['theta 1', 'theta 2', 'theta 3', 'theta 4', 'theta 5'])
plt.savefig('n=5 angles wind.png',dpi = 240)
plt.show()

plt.plot(t, omega_5nodes.T)
plt.title('Frequency Values with 5 Nodes')
plt.xlabel('t')
plt.legend(['omega 1', 'omega 2', 'omega 3', 'omega 4', 'omega 5'])
plt.savefig('n=5 freq wind.png',dpi = 240)
plt.show()

#%%

def swing_equation_10n(t, theta):
    #set gamma
    gamma = 1
    #set kappa
    kappa = 999.80
    #set Pk_t 
    P = 0.000119*(t**4) - 0.00240*(t**3) + 0.021246*(t**2) - 0.065211*t + 0.203511 
    
    
    theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, freq_1, freq_2, freq_3, freq_4, freq_5,  freq_6, freq_7, freq_8, freq_9, freq_10 = theta
    
    diff_freq_1 = P - kappa*np.sin(theta_1 - theta_2) - gamma*freq_1
    diff_freq_2 = P - kappa*(np.sin(theta_2 - theta_1) + np.sin(theta_2 -theta_3)) - gamma*freq_2
    diff_freq_3 = P - kappa*(np.sin(theta_3 - theta_2) + np.sin(theta_3 - theta_4)) - gamma*freq_3
    diff_freq_4 = P - kappa*(np.sin(theta_4 - theta_3) + np.sin(theta_4 - theta_5)) - gamma*freq_4
    diff_freq_5 = P - kappa*(np.sin(theta_5 - theta_4) + np.sin(theta_5 - theta_6)) - gamma*freq_5
    diff_freq_6 = P - kappa*(np.sin(theta_6 - theta_5) + np.sin(theta_6 - theta_7)) - gamma*freq_6
    diff_freq_7 = P - kappa*(np.sin(theta_7 - theta_6) + np.sin(theta_7 -theta_8)) - gamma*freq_7
    diff_freq_8 = P - kappa*(np.sin(theta_8 - theta_7) + np.sin(theta_8 - theta_9)) - gamma*freq_8
    diff_freq_9 = P - kappa*(np.sin(theta_9 - theta_8) + np.sin(theta_9 - theta_10)) - gamma*freq_9
    diff_freq_10 = P - kappa*np.sin(theta_10 - theta_9) - gamma*freq_10
    diff_theta_1 = freq_1
    diff_theta_2 = freq_2
    diff_theta_3 = freq_3
    diff_theta_4 = freq_4
    diff_theta_5 = freq_5
    diff_theta_6 = freq_6
    diff_theta_7 = freq_7
    diff_theta_8 = freq_8
    diff_theta_9 = freq_9
    diff_theta_10 = freq_10
    
    return diff_theta_1, diff_theta_2, diff_theta_3, diff_theta_4, diff_theta_5, diff_theta_6, diff_theta_7, diff_theta_8, diff_theta_9, diff_theta_10, diff_freq_1, diff_freq_2, diff_freq_3, diff_freq_4, diff_freq_5, diff_freq_6, diff_freq_7, diff_freq_8, diff_freq_9, diff_freq_10

y0 = np.zeros(10)
for i in np.arange(10):
    y0[i] = np.random.uniform(-np.pi, np.pi)

sol_10nodes = solve_ivp(swing_equation_10n, [0, np.pi], [y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], y0[6], y0[7], y0[8], y0[9],  0, 0, 0, 0 , 0, 0, 0, 0, 0 , 0], dense_output=True, method = 'RK45')
t = np.linspace(0, np.pi, 1000)
theta_10nodes = sol_10nodes.sol(t)
phaseangles_10nodes = theta_10nodes[0:10]
omega_10nodes = theta_10nodes[10:21]

plt.plot(t, phaseangles_10nodes.T)
plt.xlabel('t')
plt.legend(['theta 1', 'theta 2', 'theta 3', 'theta 4', 'theta 5', 'theta 6', 'theta 7', 'theta 8', 'theta 9', 'theta 10'])
plt.title('Phase Angles with 10 Nodes')
plt.savefig('n=10 angles wind.png', dpi = 240)
plt.show()

plt.plot(t, omega_10nodes.T)
plt.xlabel('t')
plt.legend(['omega 1', 'omega 2', 'omega 3', 'omega 4', 'omega 5', 'omega 6', 'omega 7', 'omega 8', 'omega 9', 'omega 10'])

plt.title('Frequency Values with 10 Nodes')
plt.savefig('n=10 freq wind.png', dpi = 240)
plt.show()