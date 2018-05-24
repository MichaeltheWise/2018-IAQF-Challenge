#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:40:18 2018

@author: MichaelLin
"""

# Main.py
# FRE6381 Final Project
# Black Litterman Model 
# Written by: Po-Hsuan (Michael) Lin
# Date: May, 17, 2018
# This is the main code that calls on different functions from PortfolioFunctions.py
# and BLackLittermanPortfolioOptimizer.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import PortfolioFunctions as pf
import BlackLittermanPortfolioOptimizer as BL

# Instantiation
r = 0.01

# Extract data from data file of some financial stocks in 2016
data = pd.read_csv("Data.csv",header = 0, index_col = 0)
data.index = pd.to_datetime(data.index)

# Calculate the log return
data = np.log(data) - np.log(data.shift(1))
data = data.dropna()

# Calculate the covariance matrix of return of the stocks
sigmaR = data.cov()
fig = plt.figure(figsize = (20,10))
sns.heatmap(sigmaR,xticklabels = sigmaR.columns, yticklabels = sigmaR.columns)
ax = plt.gca()
ax.set_title("Financial Stock Covariance Heat Map")

# Store the data
covariance_matrix = sigmaR.as_matrix()
expected_return = (data.mean()).as_matrix()
# sigmaR.to_csv("sigmaR.csv")

## Markowitz Portfolio Optimizer ##
# Calculate tangency portfolio
W_T = pf.Tangency_Portfolio(expected_return,r,covariance_matrix)
Portfolio = pd.DataFrame(data = W_T * 100)
Portfolio.columns = ['Tangency Portfolio']
Portfolio.index = data.columns

# Calculate minimum variance portfolio with cash position
portfolio_expected_return = np.array([0.005,0.02,0.05,0.10])
portfolio_expected_return = np.reshape(portfolio_expected_return, (4,1))
W_min_cash, W_min = pf.Minimum_Variance_Portfolio(expected_return,r,covariance_matrix,portfolio_expected_return)
W_min_cash = W_min_cash * 100
W_min = np.transpose(W_min * 100)
Portfolio['MVP with mu = 0.5%'] = W_min[:,0]
Portfolio['MVP with mu = 2%'] = W_min[:,1]
Portfolio['MVP with mu = 5%'] = W_min[:,2]
Portfolio['MVP with mu = 10%'] = W_min[:,3]

# Calculate minimum variance portfolio without cash position
W_min_no_cash = pf.Minimum_Variance_No_Cash_Portfolio(expected_return,r,covariance_matrix)
W_min_no_cash = np.transpose(W_min_no_cash * 100)
Portfolio['MVP with no cash'] = W_min_no_cash

# Calculate maximum return portfolio with cash position
portfolio_expected_vol = np.array([0.005,0.02,0.05,0.10])
portfolio_expected_vol = np.reshape(portfolio_expected_vol, (4,1))
W_max_cash, W_max = pf.Maximum_Return_Portfolio(expected_return,r,covariance_matrix,portfolio_expected_vol)
W_max_cash = W_max_cash * 100
W_max = np.transpose(W_max * 100)
Portfolio['MRP with sigma = 0.5%'] = W_max[:,0]
Portfolio['MRP with sigma = 2%'] = W_max[:,1]
Portfolio['MRP with sigma = 5%'] = W_max[:,2]
Portfolio['MRP with sigma = 10%'] = W_max[:,3]

# Calculate return and std
tangency_return = pf.Portfolio_Return(expected_return,r,W_T*100)
tangency_sigma = pf.Portfolio_Std(covariance_matrix,W_T*100)
min_no_cash_return = pf.Portfolio_Return(expected_return,r,W_min_no_cash)
min_no_cash_sigma = pf.Portfolio_Std(covariance_matrix,W_min_no_cash)
min_return = np.zeros((4,1))
min_sigma = np.zeros((4,1))
max_return = np.zeros((4,1))
max_sigma = np.zeros((4,1))
for i in range(0,4):
    min_return[i] = pf.Portfolio_Return(expected_return,r,W_min[:,i])
    min_sigma[i] = pf.Portfolio_Std(covariance_matrix,W_min[:,i])
    max_return[i] = pf.Portfolio_Return(expected_return,r,W_max[:,i])
    max_sigma[i] = pf.Portfolio_Std(covariance_matrix,W_max[:,i])

## Black-Litterman Portfolio Optimizer ##
# Instantiation
delta = 2.5
tau = 0.05
# W_eq found using market capitalization
W_eq = np.array([[0.194597543,0.047809018,0.113142089,0.056973787,0.239977421,0.060538216,0.068339369,0.052660954,0.165961603]]).T
P = np.array([[0,0,0,0,0,1,0,0,0],[0,0,-1,1,0,0,0,0,0],[0.35,-0.35,0,0,0.65,0,0,0,-0.65]])
Q = np.array([[0.05],[0.02],[0.015]])
omega = np.zeros((3,3))
omega[0,0] = P[0,:] @ covariance_matrix @ np.transpose(P[0,:])
omega[1,1] = P[1,:] @ covariance_matrix @ np.transpose(P[1,:])
omega[2,2] = P[2,:] @ covariance_matrix @ np.transpose(P[2,:])

# Call the formula
expected_return, modified_expected_return, modified_sigmaR, modified_weight = BL.Black_Litterman_Model(W_eq,covariance_matrix,delta,tau,P,Q,omega)
modified_sigmaR = pd.DataFrame(data = modified_sigmaR)
modified_weight = modified_weight * 100
# modified_sigmaR.to_csv("sigmaR.csv")

# Visualization
W_eq = W_eq * 100
Weight_comparison_data = pd.DataFrame(data = W_eq)
Weight_comparison_data.columns = ['Original Weight']
Weight_comparison_data.index = data.columns
Weight_comparison_data['Modified Weight'] = modified_weight
fig = plt.figure()
Weight_comparison_data.plot(figsize = (20,10),kind = 'bar')
ax = plt.gca()
ax.set_title('Comparison Between Original Weight and Modified Weight')
ax.grid(True,linestyle = '--')
