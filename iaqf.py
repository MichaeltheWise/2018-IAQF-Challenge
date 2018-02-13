#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:21:01 2018

@author: MichaelLin
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

## CLEANING THE DATA
SP500_tradedata = pd.read_csv('SnP500_10years_data.csv', header = 0, index_col = 0)
SP500_tradedata.index = pd.to_datetime(SP500_tradedata.index)
tradedata = pd.DataFrame()
tradedata['price'] = SP500_tradedata['Adj Close']

interest_rate_tradedata = pd.read_csv('USTREASURY-YIELD.csv', header = 0, index_col = 0)
interest_rate_tradedata.index = pd.to_datetime(interest_rate_tradedata.index)
interest_rate_tradedata.reindex(SP500_tradedata.index)

# The data we will be using
## DATA SETUP
data = pd.concat([tradedata,interest_rate_tradedata],axis = 1)
data = data.dropna()
data['60MA'] = data['price'].rolling(60).mean()
data['120MA'] = data['price'].rolling(120).mean()
data['pct_return'] = data['price'].pct_change()
data['volatility'] = data['pct_return'].rolling(90).std() * np.sqrt(255)
data = data.dropna()
data['volatility'].plot()

## PORTFOLIO 1
data['portfolio1_return'] = data['pct_return']
for i in range (0,len(data['60MA'])):
    if data['60MA'][i] > data['120MA'][i]:
        data['portfolio1_return'][i] = data['pct_return'][i]
    else:
        data['portfolio1_return'][i] = -(data['pct_return'][i])
data['cumsum_portfolio1_return'] = data['portfolio1_return'].cumsum()
ax = data['cumsum_portfolio1_return'].plot()

## PORTFOLIO 2
# First create option functions 
def calloption(spot_price,strike_price,maturity,volatility,interest_rate):
    # Write Black Scholes equation
    d1_term = (np.log(spot_price/strike_price) + (interest_rate + (np.square(volatility)/2)) * maturity) / (volatility * np.sqrt(maturity))
    d2_term = d1_term - (spot_price * np.sqrt(maturity))
    first_term = norm.cdf(d1_term) * spot_price
    second_term = norm.cdf(d2_term) * (strike_price * np.exp((-1) * interest_rate * maturity))
    return (first_term + second_term)

def putoption(spot_price,strike_price,maturity,volatility,interest_rate):
    # Write Black Scholes equation
    d1_term = (np.log(spot_price/strike_price) + (interest_rate + (np.square(volatility)/2)) * maturity) / (volatility * np.sqrt(maturity))
    d2_term = d1_term - (spot_price * np.sqrt(maturity))
    first_term = norm.cdf(-d2_term) * (strike_price * np.exp((-1) * interest_rate * maturity)) 
    second_term = norm.cdf(-d1_term) * spot_price
    return (first_term - second_term)

# Testing the output
# print (calloption(data['price'][0],data['price'][0],90,data['volatility'][0],data['3 MO'][0]/100))
# print (putoption(data['price'][0],data['price'][0],90,data['volatility'][0],data['3 MO'][0]/100))

data['portfolio2_return'] = data['pct_return']
for i in range (1,len(data['60MA'])):
    if i == len(data['60MA']):
        data['portfolio2_return'][i] = data['portfolio2_return'][i-1]
    elif (data['60MA'][i-1] > data['120MA'][i-1]) and i != len(data['60MA']):
        calloption_today = calloption(data['price'][i-1],data['price'][i-1],90,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        calloption_tmrw = calloption(data['price'][i],data['price'][i-1],89,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        data['portfolio2_return'][i-1] = (calloption_tmrw - calloption_today) / calloption_today
    elif (data['60MA'][i-1] < data['120MA'][i-1]) and i != len(data['60MA']):
        putoption_today = putoption(data['price'][i-1],data['price'][i-1],90,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        putoption_tmrw = putoption(data['price'][i],data['price'][i-1],89,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        data['portfolio2_return'][i-1] = (putoption_tmrw - putoption_today) / putoption_today
data['cumsum_portfolio2_return'] = data['portfolio2_return'].cumsum()
data['cumsum_portfolio2_return'].plot(ax = ax)

## PORTFOLIO 3
data['portfolio3_return'] = data['pct_return']
for i in range (1,len(data['60MA'])):
    if i == len(data['60MA']):
        data['portfolio3_return'][i] = data['portfolio3_return'][i-1]
        calloption_today = calloption(data['price'][i-1],data['price'][i-1],90,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        calloption_tmrw = calloption(data['price'][i],data['price'][i-1],89,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        putoption_today = putoption(data['price'][i-1],data['price'][i-1],90,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        putoption_tmrw = putoption(data['price'][i],data['price'][i-1],89,data['volatility'][i-1],(data['3 MO'][i-1])/100)
        straddle_today = calloption_today + putoption_today
        straddle_tmrw = calloption_tmrw + putoption_tmrw
        data['portfolio3_return'][i-1] = (straddle_tmrw - straddle_today) / straddle_today

data['cumsum_portfolio3_return'] = data['portfolio3_return'].cumsum()
data['cumsum_portfolio3_return'].plot(ax = ax)

# ADJUST COLUMNS FOR BETTER DATA OUTPUT
cols = data.columns.tolist()
cols = ['price','60MA','120MA','pct_return','portfolio1_return','cumsum_portfolio1_return','portfolio2_return','cumsum_portfolio2_return','portfolio3_return','cumsum_portfolio3_return','volatility','1 MO','3 MO','6 MO','1 YR', '2 YR', '3 YR', '5 YR', '7 YR', '10 YR', '20 YR', '30 YR']
data = data[cols]
# data.to_csv['iaqf_data_output.csv']




