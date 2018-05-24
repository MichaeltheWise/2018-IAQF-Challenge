#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:52:52 2018

@author: MichaelLin
"""

# FRE6233 Homework7 
# Author: Po-Hsuan (Michael) Lin
# Date: 04/07/2018

import numpy as np
import math

def binomial_option_pricing(s, k, u, d, r, n, call, amer):
    # s: initial stock price
    # k: initial strike price
    # u: up factor in binomial model
    # d: down factor in binomial model
    # r: risk-free rate
    # n: number of binomial step
    # call: if it's call, then it should be 1; if put, then it should be -1
    # amer: if it's American option, then it should be 1; if European, other number
    
    # Calculate risk neutral probability
    q = (1 + r - d) / (u - d)
    
    # Calculate node values
    node_val = np.zeros((n+1,n+1))
    node_val[0,0] = s
    for i in range(1,n+1):
        node_val[i,0] = node_val[i-1,0] * u
        for j in range(1,n+1):
            node_val[i,j] = node_val[i-1,j-1] * d
            
    # Check whether the nodes are constructed correctly
    # return node_val
    
    # Backward pricing
    option_val = np.zeros((n+1,n+1))
    for m in range(n+1):
        option_val[n,m] = max(call*(node_val[n,m]-k),0)
    
    # Check whether the terminal values are constructed correctly
    # return option_val
    
    for i in range(n-1,-1,-1):
        # Since it is American option, need to do a control flow
        for j in range(i+1):
            # Risk neutral pricing
            option_val[i,j] = (q * option_val[i+1,j] + (1-q) * option_val[i+1,j+1]) / (1 + r)
            if amer:
                option_val[i,j] = max(option_val[i,j],max(call*(node_val[i,j] - k),0))
            
    return node_val, option_val

# Test European Call Option
# node_tree, option_val_tree = binomial_option_pricing(100,90,2,1/2,1/2,2,1,0)
    
# Test European Put Option
# node_tree, option_val_tree = binomial_option_pricing(100,90,2,1/2,1/2,2,-1,0)
    
# Test American Call Option: should be the same as European call since no dividend
# node_tree, option_val_tree = binomial_option_pricing(100,90,2,1/2,1/2,2,1,1)
    
# Test American Put Option
# node_tree, option_val_tree = binomial_option_pricing(100,90,2,1/2,1/2,2,-1,1)

# Actual homework question
node_tree, option_val_tree = binomial_option_pricing(100,100,2,1/2,1/2,10,-1,1)
print(option_val_tree[0,0])       
            