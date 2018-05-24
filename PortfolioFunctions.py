#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:11:51 2018

@author: MichaelLin
"""
# PortfolioFunctions.py
# FRE6381 Final Project
# Black Litterman Model 
# Written by: Po-Hsuan (Michael) Lin
# Date: May, 17, 2018
# Function files: containing functions that calculate minimum variance portfoliio, 
# maximum return portfolio, tangency portfolio

## VARIABLES LIST ##
# Expected return: expected (equilibrium) return of assets; should be a matrix
# r: risk-free rate of assets
# weights: weights from the portfolios
# sigmaR: covariance matrix of the given assets' return in the portfolio; should be a matrix
# portfolio_return: the required return for a minimum variance portfolio; should be a number
# portfolio_sigma: the required std for a maximum return portfolio; should be a number

import numpy as np

def Portfolio_Return(expected_return, r, weights):
    mean_return = expected_return - r
    mean_return = np.transpose(mean_return)
    return (r + np.dot(mean_return,weights))

def Portfolio_Std(sigmaR, weights):
    return np.sqrt(np.transpose(weights) @ sigmaR @ weights)

def Tangency_Portfolio(expected_return, r, sigmaR):
    # In order for the formula to work, please ensure that expected_return and sigmaR are both matrices
    mean_return = expected_return - r
    mean_return = np.transpose(mean_return)
    H_matrix = np.linalg.solve(sigmaR, mean_return)
    one_matrix = np.ones(np.size(sigmaR,1))
    tangency_weight = (1 / (np.dot(np.transpose(one_matrix),H_matrix))) * H_matrix
    # tangency_weight should be a vector with the tangency weights for each assets
    return tangency_weight

def Minimum_Variance_Portfolio(expected_return, r, sigmaR, portfolio_return):
    # Call the tangency portfolio
    tangency_weight = Tangency_Portfolio(expected_return, r, sigmaR)
    # Calculate the cash portion of minimum variance portfolio
    mean_return = expected_return - r
    mean_return = np.transpose(mean_return)
    mv_cash = 1 - ((portfolio_return - r) / np.dot(mean_return,tangency_weight))
    mv_weight = (1 - mv_cash) * tangency_weight
    # mv_weight should be a vector while mv_cash is just a percentage
    return mv_cash, mv_weight
    
def Minimum_Variance_No_Cash_Portfolio(expected_return, r, sigmaR):
    # This portfolio has no cash investment
    one_matrix = np.ones(np.size(sigmaR,1))
    one_matrix = np.transpose(one_matrix)
    H_matrix = np.linalg.solve(sigmaR,one_matrix)
    mv_weight = (1 / (np.dot(np.transpose(one_matrix),H_matrix))) * H_matrix
    # mv_weight should be a vector
    return mv_weight

def Maximum_Return_Portfolio(expected_return, r, sigmaR, portfolio_sigma):
    # Call the tangency portfolio
    tangency_weight = Tangency_Portfolio(expected_return, r, sigmaR)
    # Calculate the cash portion of maximum return portfolio
    mean_return = expected_return - r
    mean_return = np.transpose(mean_return)
    H_matrix = np.linalg.solve(sigmaR, mean_return)
    one_matrix = np.ones(np.size(sigmaR,1))
    sign = np.sign(np.dot(np.transpose(one_matrix),H_matrix))
    G_matrix = np.transpose(tangency_weight) @ sigmaR @ tangency_weight
    mr_cash = 1 - ((portfolio_sigma * sign) / np.sqrt(G_matrix))
    mr_weight = (1 - mr_cash) * tangency_weight
    # mr_weight should be a vector while mr_cash is just a percentage
    return mr_cash, mr_weight

    
    
    
    
    
    
    
    
    
    