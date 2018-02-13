# Python Respository

The IAQF challenge consists of constructing 4 portfolios that will mimic a momentum strategy. Three files of this repository are dedicated to this challenge: iaqf.py, SnP500_10years_data.csv and USTREASURY-YIELD.csv. iaqf.py contains the first three portfolios. 

The first portfolio longs and shorts the underlying asset, which is S&P 500 index when 60 moving average and 120 moving average crosses path. The second portfolio follows the same condition but instead of S&P 500 Index, it longs call and put option related to the index. As for the last portfolio, everytime the moving average provides a signal, a straddle is purchased to mimic the strategy. 

A graph is constructed by the code, showing the cumulative returns of each portfolios with straddle portfolio showing the most returns in the long run while the option portfolio (portfolio 2) lags behind in return. 
