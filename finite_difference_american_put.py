#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:51:21 2018

@author: MichaelLin
"""

# FRE6233 Homework 8
# Written by Po-Hsuan (Michael) Lin
# Date: April 15, 2018

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

r = 0.01
sigma = 0.2
K = 100
T = 1

x_max = 200 # x_max is the highest stock price value
M = 1000 # M will be the number of increments we desire
dx = x_max/M # x increment values 

i = np.linspace(1,M-1,M-1)
x_value = dx * i # Stock prices

# Set up initial payoff at the right hand bound and U to capture each iteration
u_initial = np.maximum(K - x_value,0)
uarray = [ ]

for N in list([100,200,400,800,1600]):
    dt = T/N # time increment values
    
    # Calculate a, b, c
    a = dt * ((-r * i)/2 - (sigma**2 * i**2)/2)
    b = dt * ((1/dt) + (sigma**2 * i**2) + r)
    c = dt * ((r * i)/2 - (sigma**2 * i**2)/2)
    
    # Calculate the weighting matrix, as denoted as M in lecture notes
    A = np.zeros((M-1,M-1))
    for j in range(1,M-2):
        A[j,j] = b[j]
        A[j,j-1] = a[j]
        A[j,j+1] = c[j]
    
    # To make it full rank, correction made following the help pdf
    A[0,0] = 1
    A[0,1] = 0
    A[M-2,M-2] = 1
    A[M-2,M-3] = 0
    # If purely tridiagonal
    # A[0,0] = b[0]
    # A[0,1] = c[0]
    # A[M-2,M-2] = a[M-2]
    # A[M-2,M-3] = b[M-2]
    
    u = np.maximum(K - x_value,0)
    # Iteration
    for t in range(N):
        # u[0] = u[0] - a[0] * K * np.exp(-r * (t+1) * dt
        # Linear algebra solver
        u = np.linalg.solve(A,u)
        u[0] = K * np.exp(-r * (N-t+1) * dt)
        u[M-2] = 0
        u = np.maximum(u,u_initial)
    uarray.append(u)
    
# This is only for comparison purpose
def black_scholes_european_put(t, S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S/K) + (r - 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    return (K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S * norm.cdf(-d1))

# Calculate conversion table and present some selected data
conversion_table = np.zeros((5, 6)) 
indices = list([399,449,499,549,599]) 
for j in range(5):
    for k in range(5):
        conversion_table[j][k] = uarray[k][indices[j]]
    conversion_table[j][k+1] = black_scholes_european_put(0, x_value[indices[j]], K, r, sigma, T)
data = pd.DataFrame(conversion_table)
data.columns = ['N = 100', 'N = 200', 'N = 400', 'N = 800', 'N = 1600', 'European']
data.index = ['x = 80', 'x = 90', 'x = 100', 'x = 110','x = 120']
print(data)

# Plot the approximated American put option price using finite difference
#plt.figure(1,figsize = (20,10))
#plt.plot(x_value,u)
#plt.xlabel('Stock Price')
#plt.ylabel('Option Price')
#plt.ylim(-10,100)
#plt.title('American Put Option Price: Finite Difference')
#plt.show()
#
## Plot the free boundary when N = 1600
#N = 1600
#boundary = np.zeros(N) 
#l = np.maximum(K - x_value,0)
#for n in range(N):
#    l[0] = K * np.exp(-r * (N-n+1) * dt)
#    l = np.linalg.solve(A, l)
#    # Found the specific boudary point
#    for k in range(len(u)): 
#        if l[k] > u_initial[k] and k != 0:
#            boundary[n] = x_value[k]
#            break
#    l = np.maximum(l, u_initial)
#plt.figure(2,figsize = (10,5))
#t = dt * np.linspace(1, N, N)
#plt.plot(t, boundary)
#plt.xlabel('Time in Year')
#plt.ylabel('Stock price')
#plt.title('American Put Option Free Boundary')
    
    
    
    
    
    
    
    
    
