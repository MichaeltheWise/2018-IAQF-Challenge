#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:32:31 2018

@author: MichaelLin
"""

import pandas as pd 
# import numpy as np

# Import csv file into python, so the data can be processed
tradedata = pd.read_csv('data.csv',header=0)
# print (tradedata)

# find all the different security in the data set and store in a vector that 
# can be cycle through
# print (tradedata.sym.unique())
securities = tradedata.sym.unique()
# print(len(securities))

# Instantiation
sum_t = [[] for x in range(len(securities))]
sum_o = [[] for x in range(len(securities))]
sum_a = [[] for x in range(len(securities))]
sum_u = [[] for x in range(len(securities))]
avg_price = [[] for x in range(len(securities))]
close_auction_price = []
close_auction_date = []

for i in range(0,len(securities)):
    # print (securities[i])
    finder = tradedata.loc[tradedata['sym']==securities[i]]
    sum_t[i] = finder['size'].sum()
    # print (sum)
    
    # Calculate the daily trading size through continuous trading
    continuous_trading_finder = finder.loc[finder['sale_cond']=='o']
    # print (continuous_trading_finder['size'])
    # break
    sum_o[i] = continuous_trading_finder['size'].sum()
    # print (sum_o)
    
    # Calculate the daily trading size through off-exchange trading
    off_exchange_finder = finder.loc[finder['sale_cond']=='a']
    sum_a[i] = off_exchange_finder['size'].sum()
    # print (sum_a)

    # Calculate the daily trading size through auction trading
    auction_trade_finder = finder.loc[finder['sale_cond']=='u']
    sum_u[i] = auction_trade_finder['size'].sum()
    # print (sum_u)
    
    avg_price[i] = finder['price'].mean()

    # Find the closing auction prices for each day
    close_auction_time_finder = finder.loc[finder['minute']=="20:00"]
    close_auction_finder = close_auction_time_finder.loc[
            close_auction_time_finder['sale_cond']=='u']
    df1 = pd.DataFrame(data = close_auction_finder['date'])
    df1[securities[i]] = close_auction_finder['price']
   
# print (securities)
d1 = {'Continuous trading volume':sum_o,'Off-exchange trading volume': sum_a,
      'Auction trading volume': sum_u,'Total trading volume': sum_t,
     'Average price': avg_price,'Securities':securities}
df = pd.DataFrame(data = d1)
df = df[['Securities','Average price','Total trading volume','Continuous trading volume',
         'Off-exchange trading volume','Auction trading volume']]
df.to_csv('question1.csv')

close_auction_corr = df1.corr()
print(df1)
# print (close_auction_corr)

# Given more time, I can complete all the questions. 
# Unfortunately I am stuck in handling panda dataframe append()

        