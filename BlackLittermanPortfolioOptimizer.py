#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:26:32 2018

@author: MichaelLin
"""

# BlackLittermanPortfolioOptimizer.py
# FRE6381 Final Project
# Black Litterman Model 
# Written by: Po-Hsuan (Michael) Lin
# Date: May, 17, 2018
# Function files: containing function that calculates Black-Litterman Models

# Black Litterman Model assumes market is in equilibrium and analysts have opinions 
# on specific assets relative to the accepted equilibrium, rather than expected returns of all assets

## VARIABLES LIST ##
# eq_weights: equilibrium weights
# sigmaR: covariance matrix of the given assets' return in the portfolio; should be a matrix
# delta: risk aversion parameter; see paper footnote
# tau: a scalar that measures the uncertainty of CAPM; see paper footnote
# P_matrix: probability of views happening; see paper
# Q_matrix: analyst's view matrix
# Omega_matrix: analyst's view uncertainty matrix; see paper

import numpy as np

def Black_Litterman_Model(eq_weights,sigmaR,delta,tau,P_matrix,Q_matrix,Omega_matrix):
    # First find the equilibrium expected return, assuming it is not given
    # According to the footnote, expected return = equilibrium risk premium and can be found using CAPM
    # Expected_return should be a vector
    expected_return = delta * np.dot(sigmaR, eq_weights)
    
    # From page nine of the paper (or page 35 from the journal)
    # We have the formula to calculate the new expected return with analyst's views incorporated
    # First calculate the view added term
    analyst_view_r_1 = np.dot(tau * np.dot(sigmaR,np.transpose(P_matrix)), np.linalg.inv(tau * np.dot(np.dot(P_matrix,sigmaR),np.transpose(P_matrix)) + Omega_matrix))
    analyst_view_r_2 = Q_matrix - np.dot(P_matrix,expected_return)
    view_term = np.dot(analyst_view_r_1,analyst_view_r_2)
    
    # Incorporate the added term into expected return
    modified_expected_return = expected_return + view_term
    
    # SigmaR can also be updated with the view
    analyst_view_s_1 = tau * sigmaR
    analyst_view_s_2 = np.dot(analyst_view_r_1, tau * np.dot(P_matrix, sigmaR))
    modified_sigmaR = sigmaR + analyst_view_s_1 - analyst_view_s_2
    
    # Find the optimal weight using modified_expected_return and modified_sigmaR
    modified_weight = np.dot(np.linalg.inv(delta * modified_sigmaR),modified_expected_return)
    return expected_return, modified_expected_return, modified_sigmaR, modified_weight
    
    
    
    
    