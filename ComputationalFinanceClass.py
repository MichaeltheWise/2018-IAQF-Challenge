#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:23:49 2018

@author: MichaelLin
"""

# ComputationalFinanceClass.py
# Written by Po-Hsuan (Michael) Lin
# Date: April 07, 2018

# Create a class that contains functions that calculate european call/put option payoff

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean
import sys

class ComputationalFinance (object):
    
    def __init__(self, stock_price, strike_price):
        self.stock_price = stock_price
        self.strike_price = strike_price
        
    def European_Call_Option_Payoff(self):
        return np.maximum(self.stock_price - self.strike_price,0)
    
    def European_Put_Option_Payoff(self):
        return np.maximum(self.strike_price - self.stock_price,0)
    
    def Bull_Call_Spread(self,strike_price_1,strike_price_2): 
        if strike_price_1 > strike_price_2:
            sys.exit("Error! Please flip the strike prices so the first one is smaller!")
        else:
            payoff = np.maximum(self.stock_price - strike_price_1,0) - np.maximum(self.stock_price - strike_price_2,0)
            return payoff
        
    def Bull_Put_Spread(self,strike_price_1,strike_price_2): 
        if strike_price_1 < strike_price_2:
            sys.exit("Error! Please flip the strike prices so the first one is larger!")
        else:
            payoff = np.maximum(strike_price_1 - self.stock_price,0) - np.maximum(strike_price_2 - self.stock_price,0)
            return payoff
        
    def Bear_Call_Spread(self,strike_price_1,strike_price_2): 
        if strike_price_1 < strike_price_2:
            sys.exit("Error! Please flip the strike prices so the first one is larger!")
        else:
            payoff = np.maximum(self.stock_price - strike_price_1,0) - np.maximum(self.stock_price - strike_price_2,0)
            return payoff
     
    def Collar(self,strike_price_1,strike_price_2): 
        if strike_price_1 > strike_price_2:
            sys.exit("Error! Please flip the strike prices so the first one is smaller!")
        else:
            payoff = np.maximum(strike_price_1 - self.stock_price,0) - np.maximum(self.stock_price - strike_price_2,0)
            return payoff
        
    def Straddle(self): 
        return self.European_Call_Option_Payoff() + self.European_Put_Option_Payoff()
    
    def Strangle(self,strike_price_1,strike_price_2):
        if strike_price_1 == strike_price_2:
            sys.exit("Error! Please make sure the two strikes are different!")
        else:
            payoff = np.maximum(self.stock_price - strike_price_1,0) + np.maximum(strike_price_2 - self.stock_price,0)
            return payoff
        
    def Butterfly_Spread(self, strike_price_1, strike_price_2, strike_price_3):
        if (strike_price_1 < strike_price_2 and strike_price_2 < strike_price_3):
            z = (strike_price_3 - strike_price_2) / (strike_price_3 - strike_price_1)
            payoff = z * np.maximum(self.stock_price - strike_price_1,0) + (1-z) * np.maximum(self.stock_price - strike_price_3,0) - np.maximum(self.stock_price - strike_price_2,0)
            return payoff
        else: 
            sys.exit("Error! Please make sure strike price 1 < strike price 2 < strike price 3!")
        
    def Black_Scholes_European_Call(self,t,maturity_date,stock_price,strike_price,interest_rate,dividend_yield,volatility):
        d1 = (np.log(stock_price/strike_price) + (interest_rate - dividend_yield + (np.square(volatility)/2)) * (maturity_date - t)) / (volatility * np.sqrt(maturity_date - t))
        d2 = d1 - (volatility * np.sqrt(maturity_date - t))
        bs_european_call_price = stock_price * np.exp(-dividend_yield*(maturity_date -t)) * norm.cdf(d1) - strike_price * np.exp(-interest_rate*(maturity_date - t)) * norm.cdf(d2)
        bs_european_call_delta = np.exp(-dividend_yield*(maturity_date - t)) * norm.cdf(d1)
        bs_european_call_theta = (dividend_yield * stock_price * np.exp(-1 * dividend_yield * (maturity_date - t)) * norm.cdf(d1)) - ((volatility * np.exp(-1 * dividend_yield * (maturity_date - t)) * stock_price * norm.pdf(d1))/(2 * np.sqrt(maturity_date - t))) - (interest_rate * np.exp(-1 * interest_rate * (maturity_date - t)) * strike_price * norm.cdf(d2))
        bs_european_call_vega = np.sqrt(maturity_date - t) * np.exp(-dividend_yield*(maturity_date - t)) * stock_price * norm.pdf(d1)
        bs_european_call_gamma = (np.exp(-dividend_yield*(maturity_date - t)) * norm.pdf(d1)) / (stock_price * volatility * np.sqrt(maturity_date - t))
        bs_european_call_rho = (maturity_date - t) * np.exp(-interest_rate*(maturity_date - t)) * strike_price * norm.cdf(d2)
        return bs_european_call_price, bs_european_call_delta, bs_european_call_theta, \
                bs_european_call_vega, bs_european_call_gamma, bs_european_call_rho
    
    def Black_Scholes_European_Put(self,t,maturity_date,stock_price,strike_price,interest_rate,dividend_yield,volatility):
        d1 = (np.log(stock_price/strike_price) + (interest_rate + dividend_yield + (np.square(volatility)/2)) * (maturity_date - t)) / (volatility * np.sqrt(maturity_date - t))
        d2 = d1 - (volatility * np.sqrt(maturity_date - t))
        bs_european_put_price = strike_price * np.exp(-interest_rate*(maturity_date - t)) * norm.cdf(-d2) - stock_price * np.exp(-dividend_yield*(maturity_date -t)) * norm.cdf(-d1)
        bs_european_put_delta = -1 * np.exp(-dividend_yield*(maturity_date - t)) * norm.cdf(-d1)
        bs_european_put_theta = (-dividend_yield * stock_price * np.exp(-dividend_yield*(maturity_date - t)) * norm.cdf(-d1)) - ((volatility * np.exp(-dividend_yield*(maturity_date - t)) * stock_price * norm.pdf(d1))/(2 * np.sqrt(maturity_date - t))) + (interest_rate * np.exp(-interest_rate*(maturity_date - t)) * strike_price * norm.cdf(-d2))
        bs_european_put_vega = np.sqrt(maturity_date - t) * np.exp(-dividend_yield*(maturity_date - t)) * stock_price * norm.pdf(d1) 
        bs_european_put_gamma = (np.exp(-dividend_yield*(maturity_date - t)) * norm.pdf(d1)) / (stock_price * volatility * np.sqrt(maturity_date - t))
        bs_european_put_rho = -(maturity_date - t) * np.exp(-interest_rate*(maturity_date - t)) * strike_price * norm.cdf(-d2)                                                              
        return bs_european_put_price, bs_european_put_delta, bs_european_put_theta, \
                bs_european_put_vega, bs_european_put_gamma, bs_european_put_rho
    
    # For simplicity, interest_rate is denoted as r, volatility is denoted as sigma
    # strike_price is denoted as K, maturity_date is denoted as T   
    # x_max = Maximum price, dividend_yield is denoted as div
    # M is the number of price increments we desire
    # N is the number of time increments we desire
    def Black_Scholes_Explicit_FD_EO(self,r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition):
        # Price increment and time increment, assuming the lower bounds are both zero for simplicity
        dx = (x_max - x_min) / M
        dt = T / N
        # Test for stability
        if ((sigma**2 * x_max**2 * (dt/(dx**2))) > 0 and (sigma**2 * x_max**2 * (dt/(dx**2))) < 0.5):
            # Implement main code
            i = np.linspace(1,M+1,M+1)
            x_value = x_min + dx * i 
            x_value = x_value[1:M] 
            
            # Set up initial value
            if initial_condition == 'ic_call':
                # u_initial = np.maximum(x_value - K,0)
                u = np.maximum(x_value - K,0)
            elif initial_condition == 'ic_put':
                # u_initial = np.maximum(K - x_value,0)
                u = np.maximum(K - x_value,0)
                
            # Find weighting matrix
            bs_explicit_fd_eo_price = []
            a = dt * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
            b = dt * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r)
            c = dt * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
            A = np.zeros((M-1,M-1))
            if boundary_condition == 'dirichlet_bc':
                for j in range(1,M-2):
                    A[j,j] = b[j]
                    A[j,j-1] = a[j]
                    A[j,j+1] = c[j]
                A[0,0] = b[0]
                A[0,1] = c[0]
                A[M-2,M-2] = b[-1]
                A[M-2,M-3] = a[-1]
                for t in range(N):
                    u = A @ u
                    if initial_condition == 'ic_call':
                        u[0] = 0
                        u[-1] = u[-1] + c[-1] * (x_max - K)
                    elif initial_condition == 'ic_put':
                        u[0] = u[0] + a[0] * (K - x_max)
                        u[-1] = 0
                    bs_explicit_fd_eo_price.append(u)
            elif boundary_condition == 'neumann_bc':
                for j in range(1,M-2):
                    A[j,j] = b[j]
                    A[j,j-1] = a[j]
                    A[j,j+1] = c[j]
                A[0,0] = b[0] + 2 * a[0]
                A[0,1] = c[0] - a[0]
                A[M-2,M-2] = b[-1] + 2 * c[-1]
                A[M-2,M-3] = a[-1] - c[-1]
                for t in range(N):
                    u = A @ u
                    bs_explicit_fd_eo_price.append(u)  
            return bs_explicit_fd_eo_price
        else: 
            quit()
    
    # Implicit scheme    
    def Black_Scholes_Implicit_FD_EO(self,r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition):    
        # Price increment and time increment, assuming the lower bounds are both zero for simplicity
        dx = (x_max - x_min) / M
        dt = T / N
        # Implement main code
        i = np.linspace(1,M+1,M+1)
        x_value = x_min + dx * i 
        x_value = x_value[1:M] 
        
        # Set up initial value
        if initial_condition == 'ic_call':
            # u_initial = np.maximum(x_value - K,0)
            u = np.maximum(x_value - K,0)
        elif initial_condition == 'ic_put':
            # u_initial = np.maximum(K - x_value,0)
            u = np.maximum(K - x_value,0)
            
        # Find weighting matrix
        bs_implicit_fd_eo_price = []
        a = -1 * dt * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
        b = 2 - dt * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r)
        c = -1 * dt * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
        A = np.zeros((M-1,M-1))
        if boundary_condition == 'dirichlet_bc':
            for j in range(1,M-2):
                A[j,j] = b[j]
                A[j,j-1] = a[j]
                A[j,j+1] = c[j]
            A[0,0] = b[0]
            A[0,1] = c[0]
            A[M-2,M-2] = b[-1]
            A[M-2,M-3] = a[-1]
            for t in range(N):
                u = np.linalg.solve(A,u)
                if initial_condition == 'ic_call':
                    u[0] = 0
                    u[-1] = u[-1] - c[-1] * (x_max - K)
                elif initial_condition == 'ic_put':
                    u[0] = u[0] - a[0] * (K - x_max)
                    u[-1] = 0
                bs_implicit_fd_eo_price.append(u)
        elif boundary_condition == 'neumann_bc':
            for j in range(1,M-2):
                A[j,j] = b[j]
                A[j,j-1] = a[j]
                A[j,j+1] = c[j]
            A[0,0] = b[0] + 2 * a[0]
            A[0,1] = c[0] - a[0]
            A[M-2,M-2] = b[-1] + 2 * c[-1]
            A[M-2,M-3] = a[-1] - c[-1]
            for t in range(N):
                u = np.linalg.solve(A,u)
                bs_implicit_fd_eo_price.append(u)     
        return bs_implicit_fd_eo_price
    
    def Black_Scholes_CN_FD_EO(self,r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition):
        # Price increment and time increment, assuming the lower bounds are both zero for simplicity
        dx = (x_max - x_min) / M
        dt = T / N
        # Implement main code
        i = np.linspace(1,M+1,M+1)
        x_value = x_min + dx * i 
        x_value = x_value[1:M] 
        
        # Set up initial value
        if initial_condition == 'ic_call':
            # u_initial = np.maximum(x_value - K,0)
            u = np.maximum(x_value - K,0)
        elif initial_condition == 'ic_put':
            # u_initial = np.maximum(K - x_value,0)
            u = np.maximum(K - x_value,0)
        
        bs_CN_fd_eo_price = []
        a_implicit = -1 * dt * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
        b_implicit = (2 - dt * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r))
        c_implicit = -1 * dt * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
        a_explicit = dt * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
        b_explicit = dt * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r)
        c_explicit = dt * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
        A_implicit = np.zeros((M-1,M-1))
        A_explicit = np.zeros((M-1,M-1))
        if boundary_condition == 'dirichlet_bc':
            for j in range(1,M-2):
                A_implicit[j,j] = b_implicit[j] + 1
                A_implicit[j,j-1] = a_implicit[j]
                A_implicit[j,j+1] = c_implicit[j]
                A_explicit[j,j] = b_explicit[j] + 1
                A_explicit[j,j-1] = a_explicit[j]
                A_explicit[j,j+1] = c_explicit[j]
            A_implicit[0,0] = b_implicit[0] + 1
            A_implicit[0,1] = c_implicit[0]
            A_implicit[M-2,M-2] = b_implicit[-1] + 1
            A_implicit[M-2,M-3] = a_implicit[-1]
            A_explicit[0,0] = b_explicit[0] + 1
            A_explicit[0,1] = c_explicit[0]
            A_explicit[M-2,M-2] = b_explicit[-1] + 1
            A_explicit[M-2,M-3] = a_explicit[-1]
            for t in range(N):
                u = A_explicit @ u
                u = np.linalg.solve(A_implicit,u)
                if initial_condition == 'ic_call':
                    u[0] = 0
                    u[-1] = u[-1] + (c_explicit[-1] - c_implicit[-1])/2 * (x_max - K)
                elif initial_condition == 'ic_put':
                    u[0] = u[0] + (a_explicit[0] - a_implicit[0])/2 * (K - x_max)
                    u[-1] = 0
                bs_CN_fd_eo_price.append(u)   
        elif boundary_condition == 'neumann_bc':
            for j in range(1,M-2):
                A_implicit[j,j] = b_implicit[j] + 1
                A_implicit[j,j-1] = a_implicit[j]
                A_implicit[j,j+1] = c_implicit[j]
                A_explicit[j,j] = b_explicit[j] + 1 
                A_explicit[j,j-1] = a_explicit[j]
                A_explicit[j,j+1] = c_explicit[j]
            A_implicit[0,0] = b_implicit[0] + 2 * a_implicit[0] + 1
            A_implicit[0,1] = c_implicit[0] - a_implicit[0]
            A_implicit[M-2,M-2] = b_implicit[-1] + 2 * c_implicit[-1] + 1
            A_implicit[M-2,M-3] = a_implicit[-1] - c_implicit[-1]
            A_explicit[0,0] = b_explicit[0] + 2 * a_explicit[0] + 1
            A_explicit[0,1] = c_explicit[0] - a_explicit[0]
            A_explicit[M-2,M-2] = b_explicit[-1] + 2 * c_explicit[-1] + 1
            A_explicit[M-2,M-3] = a_explicit[-1] - c_explicit[-1]
            for t in range(N):
                u = A_explicit @ u
                u = np.linalg.solve(A_implicit,u)
                bs_CN_fd_eo_price.append(u)
        return bs_CN_fd_eo_price
    
    def Black_Scholes_Theta_Scheme_FD_EO(self,theta,r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition,flag):
        # Fastest way
        # The creation of a flag is to ensure that when pure theta scheme is required
        # as in the case of the last question for Question 6.3
        if theta == 0 and flag == 1: # Explicit Scheme
            bs_theta_scheme_fd_eo_price = self.Black_Scholes_Explicit_FD_EO(r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition)
        elif theta == 1 and flag == 1: # Implicit Scheme
            bs_theta_scheme_fd_eo_price = self.Black_Scholes_Implicit_FD_EO(r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition)
        elif theta == 1/2 and flag == 1: # Crank-Nicolson Scheme
            bs_theta_scheme_fd_eo_price = self.Black_Scholes_CN_FD_EO(r,sigma,div,x_min,x_max,K,T,M,N,initial_condition,boundary_condition)
        else:
            # Price increment and time increment, assuming the lower bounds are both zero for simplicity
            dx = (x_max - x_min) / M
            dt = T / N
            # Implement main code
            i = np.linspace(1,M+1,M+1)
            x_value = x_min + dx * i 
            x_value = x_value[1:M] 
            
            # Set up initial value
            if initial_condition == 'ic_call':
                # u_initial = np.maximum(x_value - K,0)
                u = np.maximum(x_value - K,0)
            elif initial_condition == 'ic_put':
                # u_initial = np.maximum(K - x_value,0)
                u = np.maximum(K - x_value,0)
                
            # We can construct a theta scheme matrix that can take in any kind of theta
            bs_theta_scheme_fd_eo_price = []
            a_implicit = -1 * dt * theta * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
            b_implicit = theta * (2 - dt * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r))
            c_implicit = -1 * dt * theta * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
            a_explicit = dt * (1-theta) * ((sigma**2 * (x_value/dx)**2)/2 - ((r - div) * (x_value/dx)/2))
            b_explicit = dt * (1-theta) * ((1/dt) - (sigma**2 * (x_value/dx)**2) - r)
            c_explicit = dt * (1-theta) * ((sigma**2 * (x_value/dx)**2)/2 + ((r - div) * (x_value/dx)/2))
            A_implicit = np.zeros((M-1,M-1))
            A_explicit = np.zeros((M-1,M-1))
            if boundary_condition == 'dirichlet_bc':
               for j in range(1,M-2):
                   A_implicit[j,j] = b_implicit[j]
                   A_implicit[j,j-1] = a_implicit[j]
                   A_implicit[j,j+1] = c_implicit[j]
                   A_explicit[j,j] = b_explicit[j]
                   A_explicit[j,j-1] = a_explicit[j]
                   A_explicit[j,j+1] = c_explicit[j]
               A_implicit[0,0] = b_implicit[0]
               A_implicit[0,1] = c_implicit[0]
               A_implicit[M-2,M-2] = b_implicit[-1]
               A_implicit[M-2,M-3] = a_implicit[-1]
               A_explicit[0,0] = b_explicit[0]
               A_explicit[0,1] = c_explicit[0]
               A_explicit[M-2,M-2] = b_explicit[-1]
               A_explicit[M-2,M-3] = a_explicit[-1]
               if theta == 0: 
                   for t in range(N):
                       u = A_explicit @ u
                       if initial_condition == 'ic_call':
                           u[0] = 0
                           u[-1] = u[-1] + ((1-theta) * c_explicit[-1] - theta * c_implicit[-1]) * (x_max - K)
                       elif initial_condition == 'ic_put':
                           u[0] = u[0] + ((1-theta) * a_explicit[0] - theta * a_explicit[0]) * (K - x_max)
                           u[-1] = 0
                       bs_theta_scheme_fd_eo_price.append(u)
               elif theta == 1:
                   for t in range(N):
                       u = np.linalg.solve(A_implicit,u)
                       if initial_condition == 'ic_call':
                           u[0] = 0
                           u[-1] = u[-1] + ((1-theta) * c_explicit[-1] - theta * c_implicit[-1]) * (x_max - K)
                       elif initial_condition == 'ic_put':
                           u[0] = u[0] + ((1-theta) * a_explicit[0] - theta * a_explicit[0]) * (K - x_max)
                           u[-1] = 0
                       bs_theta_scheme_fd_eo_price.append(u)
               else: 
                   for t in range(N):
                       u = A_explicit @ u
                       u = np.linalg.solve(A_implicit,u)
                       if initial_condition == 'ic_call':
                           u[0] = 0
                           u[-1] = u[-1] + ((1-theta) * c_explicit[-1] - theta * c_implicit[-1]) * (x_max - K)
                       elif initial_condition == 'ic_put':
                           u[0] = u[0] + ((1-theta) * a_explicit[0] - theta * a_implicit[0]) * (K - x_max)
                           u[-1] = 0
                       bs_theta_scheme_fd_eo_price.append(u)
                       
            elif boundary_condition == 'neumann_bc':
               for j in range(1,M-2):
                   A_implicit[j,j] = b_implicit[j]
                   A_implicit[j,j-1] = a_implicit[j]
                   A_implicit[j,j+1] = c_implicit[j]
                   A_explicit[j,j] = b_explicit[j]
                   A_explicit[j,j-1] = a_explicit[j]
                   A_explicit[j,j+1] = c_explicit[j]
               A_implicit[0,0] = b_implicit[0] + 2 * a_implicit[0]
               A_implicit[0,1] = c_implicit[0] - a_implicit[0]
               A_implicit[M-2,M-2] = b_implicit[-1] + 2 * c_implicit[-1]
               A_implicit[M-2,M-3] = a_implicit[-1] - c_implicit[-1]
               A_explicit[0,0] = b_explicit[0] + 2 * a_explicit[0]
               A_explicit[0,1] = c_explicit[0] - a_explicit[0]
               A_explicit[M-2,M-2] = b_explicit[-1] + 2 * c_explicit[-1]
               A_explicit[M-2,M-3] = a_explicit[-1] - c_explicit[-1]
               if theta == 0: 
                   for t in range(N):
                       u = A_explicit @ u
                       bs_theta_scheme_fd_eo_price.append(u)
               elif theta == 1:
                   for t in range(N):
                       u = np.linalg.solve(A_implicit,u)
                       bs_theta_scheme_fd_eo_price.append(u)
               else: 
                   for t in range(N):
                       u = A_explicit @ u
                       u = np.linalg.solve(A_implicit,u)
                       bs_theta_scheme_fd_eo_price.append(u)
            
        return bs_theta_scheme_fd_eo_price
    
    def Gaussian_RBF(self,e,x,y):
        r = (abs(x-y))
        phi_ga_rbf = np.exp(-e*r)**2
        phi_x_ga_rbf = (2 * e**2) * (x-y) * (np.exp(-e*r)**2)
        phi_xx_ga_rbf = (2 * e**2) * (np.exp(-e*r)**2) * ((2*e**2)*((x-y)**2) + 1)
        return phi_ga_rbf,phi_x_ga_rbf,phi_xx_ga_rbf

    def Multiquadric_RBF(self,e,x,y):
        r = (abs(x-y)) ** 2
        phi_mq_rbf = np.sqrt((e ** 2) + r)
        phi_x_mq_rbf = (x - y) / np.sqrt((e ** 2) + r)
        phi_xx_mq_rbf = (1/np.sqrt((e**2)+r))-(((x-y)**2)/(np.sqrt((e**2)+r)**3))
        return phi_mq_rbf,phi_x_mq_rbf,phi_xx_mq_rbf

    def Inverse_Multiquadric_RBF(self,e,x,y):
        r = (abs(x-y))**2
        phi_imq_rbf = 1/np.sqrt((e**2)+r)
        phi_x_imq_rbf = -1*((x-y)/((np.sqrt((e**2)+r))**3))
        phi_xx_imq_rbf = (2*((x-y)**2)-(e**2))/(np.sqrt((e**2)+r)**5)
        return phi_imq_rbf,phi_x_imq_rbf,phi_xx_imq_rbf 

    def Inverse_Quadratic_RBF(self,e,x,y):
        r = (abs(x-y))**2
        phi_iq_rbf = 1/((e**2)+r)
        phi_x_iq_rbf = -2*(x-y)/((x-y)**2 + e**2)**2
        phi_xx_iq_rbf = ((8*((x-y)**2))/((x-y)**2 + e**2)**3)- (2/(((x-y)**2) + (e**2))**2)
        return phi_iq_rbf,phi_x_iq_rbf,phi_xx_iq_rbf
    
    def Black_Scholes_Global_RBF_EO(self, r, sigma, x_min, x_max, M, N, T, epsilon, rbf_function, initial_condition, boundary_conditiion): 
        # Initialization
        dx = (x_max - x_min) / M
        dt = T / N
        i = np.linspace(1,M+1,M+1)
        s_value = x_min + dx * i 
        s_value = s_value[1:M]
        x_value = np.log(s_value)
        y_value = np.reshape(x_value,(M-1,1))
        
        # Determine which RBF function to use and assign L, Lx and Lxx matrices
        if rbf_function == 'ga_rbf':
            z = self.Gaussian_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'mq_rbf':
            z = self.Multiquadric_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'imq_rbf':
            z = self.Inverse_Multiquadric_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'iq_rbf':
            z = self.Inverse_Quadratic_RBF(epsilon,x_value,y_value)
        L = z[0]
        Lx = z[1]
        Lxx = z[2]
        
        # Calculate P
        P = np.linalg.solve(L,(np.subtract(np.add(np.multiply((0.5*(sigma**2)),Lxx),(np.multiply(r,L))),np.multiply((r - 0.5 * (sigma ** 2)),Lx))))

        # Initial payoff
        if initial_condition == 'ic_call':
            v = np.maximum(np.exp(x_value) - self.strike_price,0)
            v[0] = 0
            v[-1] = s_value[-1] - np.exp(-r * T) * self.strike_price
        elif initial_condition == 'ic_put':
            v = np.maximum(self.strike_price - np.exp(x_value),0)
            v[0] = np.exp(-r * T) * self.strike_price - s_value[-1]
            v[-1] = 0
                
        # Initial lambda, denote as l
        l = np.linalg.solve(L,v)
        
        bs_global_rbf_eo_price = []
        # Start of recursion
        for t in range(N):
            # Update lambda
            # l = np.linalg.inv(np.identity(np.size(P,1)) - 0.5 * dt * P) * (np.identity(np.size(P,1)) + 0.5 * dt * P) * l
            l_new = np.linalg.solve((np.identity(np.size(P,1)) - 0.5 * dt * P),((np.identity(np.size(P,1)) + 0.5 * dt * P) @ l))
            # Update v
            v = L @ l_new
            # Change v bounds
            if initial_condition == 'ic_call':
                v[0] = 0
                v[-1] = s_value[-1] - np.exp(-r * (T-(t * dt))) * self.strike_price
                bs_global_rbf_eo_price.append(v)
            elif initial_condition == 'ic_put':
                v[0] = np.exp(-r * (T-(t * dt))) * self.strike_price
                v[-1] = 0
                bs_global_rbf_eo_price.append(v)
            # Find the real lambda with the correct v
            l = np.linalg.inv(L) @ v
            
        return bs_global_rbf_eo_price
    
    def Black_Scholes_RBF_FD_EO(self, r, sigma, x_min, x_max, M, N, T, epsilon, rbf_function, initial_condition, boundary_conditiion):
        # Initialization
        dx = (x_max - x_min) / M
        dt = T / N
        i = np.linspace(1,M+1,M+1)
        x_value = x_min + dx * i 
        x_value = x_value[1:M]
        y_value = np.reshape(x_value,(M-1,1))
        
        # Calculate W
        if rbf_function == 'ga_rbf':
            z = self.Gaussian_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'mq_rbf':
            z = self.Multiquadric_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'imq_rbf':
            z = self.Inverse_Multiquadric_RBF(epsilon,x_value,y_value)
        elif rbf_function == 'iq_rbf':
            z = self.Inverse_Quadratic_RBF(epsilon,x_value,y_value)
        L = z[0]
        Lx = z[1]
        Lxx = z[2]
        
        W = np.zeros((M-1,M-1))
        for i in range(M-1):
            linear_operator = (r - 0.5 * (sigma ** 2)) * Lx[:,i] + (0.5 * (sigma ** 2)) * Lxx[:,i] - r * L[:,i]
            W[:,i] = np.linalg.solve(L,linear_operator)
        W = np.matrix.transpose(W)
        
        # Initialization of payoff
        if initial_condition == 'ic_call':
            v = np.maximum(x_value - self.strike_price,0)
            v[0] = 0
            v[-1] = x_value[-1] - np.exp(-r * T) * self.strike_price
        elif initial_condition == 'ic_put':
            v = np.maximum(self.strike_price - x_value,0)
            v[0] = np.exp(-r * T) * self.strike_price - x_value[-1]
            v[-1] = 0
        
        bs_rdf_fd_eo_price = []
        for t in range(N):
            v = np.linalg.solve((np.identity(np.size(W,1)) - 0.5 * dt * W),((np.identity(np.size(W,1)) + 0.5 * dt * W) @ v))
            if initial_condition == 'ic_call':
                v[0] = 0
                v[-1] = x_value[-1] - np.exp(-r * (T-(t * dt))) * self.strike_price
            elif initial_condition == 'ic_put':
                v[0] = np.exp(-r * (T-(t * dt))) * self.strike_price
                v[-1] = 0
            bs_rdf_fd_eo_price.append(v)
            
        return bs_rdf_fd_eo_price
    
    def Geometric_Brownian_Motion_Trajectory(self, pathnum, stepnum, T, r, sigma, initial_price):
        # Pathnum stores the number of paths we want to generate
        # T stores the terminal time while stepnum gives the number of steps we want
        dt = T / stepnum
        
        # Instantiation
        brownian_motion = np.random.normal(0,1,size = (pathnum,stepnum))
        S = np.zeros((pathnum,stepnum),dtype = float)
        S[:,0] = initial_price
        for i in range(1,stepnum):
            S[:,i] = S[:,i-1] * np.exp((r - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * brownian_motion[:,i])
        return S
    
    def Geometric_Brownian_Motion_Jump(self, pathnum, stepnum, T, r, div, sigma, lam_J, mu_J, sigma_J, initial_price):
        # lambda_J is used instead of lambda since lambda already defined in python
        # mu_J is jump mean parameter
        # sigma_J is jump volatility parameter
        dt = T / stepnum
        lam = lam_J * dt
        Z1 = np.random.normal(0,1,size = (pathnum,stepnum))
        N = np.random.poisson(lam = lam, size = (pathnum,stepnum))
        Z2 = np.random.normal(0,1,size = (pathnum,stepnum))
        M = np.zeros((pathnum,stepnum),dtype = float)
        X = np.zeros((pathnum,stepnum),dtype = float)
        S = np.zeros((pathnum,stepnum),dtype = float)
        
        # Original Method
        # Calculate M first
        for i in range(0,pathnum):
            for j in range(0,stepnum):
                if N[i,j] > 0:
                    M[i,j] = mu_J * N[i,j] + sigma_J * np.sqrt(N[i,j]) * Z2[i,j]
        
        # Calculate the path
        X[:,0] = np.log(initial_price)
        S[:,0] = initial_price
        for j in range(1,stepnum):
            X[:,j] = X[:,j-1] + ((r - div) - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * Z1[:,j] + M[:,j]
            S[:,j] = np.exp(X[:,j])
        return S
        
        ## Trying another method
        #kappa = np.exp(mu_J) - 1
        #for i in range(0,pathnum):
        #    S[:,0] = initial_price
        #    for j in range(0,stepnum):
        #        J = 0 
        #        if lam_J != 0:
        #            if N[i,j] > 0:
        #                for k in range(0,N[i,j]):
        #                    J = J + np.random.normal(mu_J - 0.5 * (sigma_J ** 2),sigma_J)
        #                M[i,j] = J
        #        S[i,j] = S[i,j-1] * np.exp((r - div - lam_J * kappa - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * Z1[i,j] + M[i,j])
        #return S
        
    def Arithmetic_Average_Price_Asian_Call(self, pathnum, stepnum, T, r, div, sigma, lam_J, mu_J, sigma_J, initial_price):
        # Using stock prices from Geometric Brownian Motion with Jumps
        geometric_brownian_motion_jump = self.Geometric_Brownian_Motion_Jump(pathnum,stepnum,T,r,div,sigma,lam_J,mu_J,sigma_J,initial_price)
        arithmean = np.zeros((pathnum,1))
        for i in range(0,pathnum):
            arithmean[i] = np.average(geometric_brownian_motion_jump[i,:])
        asian_call_payoff = np.exp(-r * T) * np.maximum(arithmean - self.strike_price,0)
        asian_call = np.average(asian_call_payoff)
        return asian_call, asian_call_payoff, geometric_brownian_motion_jump
        
    def Geometric_Average_Price_Asian_Call(self, T, r, geometric_brownian_motion_jump):
        # Use the same path generated in Arithmetic Average Price Asian Call
        pathnum = np.size(geometric_brownian_motion_jump,0)
        geomean = np.zeros((pathnum,1))
        for i in range(0,pathnum):
            geomean[i] = gmean(geometric_brownian_motion_jump[i,:])
        geo_asian_call_payoff = np.exp(-r * T) * np.maximum(geomean - self.strike_price,0)
        geo_asian_call = np.average(geo_asian_call_payoff)
        return geo_asian_call, geo_asian_call_payoff
    
    def BS_Geometric_Average_Price_Asian_Call(self, T, r, div, sigma, initial_price):
        sigma_tilda = sigma / np.sqrt(3)
        b = 0.5 * (r + div + (1/6) * (sigma ** 2))
        d1 = (np.log(initial_price/self.strike_price) + (r - b + 0.5 * (sigma_tilda ** 2)) * T) / (sigma_tilda * np.sqrt(T))
        d2 = d1 - sigma_tilda * np.sqrt(T)
        bs_geo_avg_price = initial_price * np.exp(-b * T) * norm.cdf(d1) - self.strike_price * np.exp(-r * T) * norm.cdf(d2)
        return bs_geo_avg_price
        
        # T_bar = (1/252) * np.sum(np.arange(1/252,1,1/252))
        # temp_sigma_sum = 0
        # for i in range(0,252):
        #    temp_sigma = (2 * i - 1) * (252 + 1 - i)/252
        #    temp_sigma_sum = temp_sigma_sum + temp_sigma
        # sigma_bar = ((sigma ** 2) / ((252 ** 2) * T_bar)) * temp_sigma_sum
        # delta = 0.5 * (sigma ** 2) - 0.5 * (sigma_bar ** 2)
        # d = (np.log(initial_price/self.strike_price) + (r - delta + 0.5 * (sigma_bar ** 2)) * T_bar) / sigma_bar * np.sqrt(T_bar)
        # bs_geo_avg_price = np.exp(-delta * T_bar - r * (1 - T_bar)) * initial_price * norm.cdf(d) - np.exp(-r) * self.strike_price * norm.cdf(d - sigma_bar * np.sqrt(T_bar))
        # return bs_geo_avg_price
    
    def Control_Variables_Arithmetic_Average_Asian_Call(self, asian_call_payoff, geo_asian_call_payoff, bs_geo_avg_price):
        # Calculate the optimal b using the function above
        data = np.append(asian_call_payoff,geo_asian_call_payoff,axis = 1)
        data = pd.DataFrame(data = data)
        optimal_b = (data.cov().as_matrix())[0,1] / (data.cov().as_matrix())[1,1]
        
        # Calculate variance reduction factor
        reduction_factor = 1 / (1 - ((data.corr().as_matrix())[0,1]) ** 2)
        
        control_variable = asian_call_payoff - optimal_b * (geo_asian_call_payoff - bs_geo_avg_price)
        control_variable_price = np.average(control_variable)
        return reduction_factor, control_variable, control_variable_price
        
        
        
        