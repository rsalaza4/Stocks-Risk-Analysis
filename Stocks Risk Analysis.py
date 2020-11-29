# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf

# Define tickers symbols
tickerSymbols = ['MSFT', 'TSLA', 'AAPL', 'AMZN', 'GOOG']

# Get the data on these tickers
MSFT = yf.Ticker(tickerSymbols[0])
TSLA = yf.Ticker(tickerSymbols[1])
AAPL = yf.Ticker(tickerSymbols[2])
AMZN = yf.Ticker(tickerSymbols[3])
GOOG = yf.Ticker(tickerSymbols[4])

# Get the historical prices for the tickers
MSFT_df = MSFT.history(period='1d', start='2015-1-1', end='2020-1-1')
TSLA_df = TSLA.history(period='1d', start='2015-1-1', end='2020-1-1')
AAPL_df = AAPL.history(period='1d', start='2015-1-1', end='2020-1-1')
AMZN_df = AMZN.history(period='1d', start='2015-1-1', end='2020-1-1')
GOOG_df = GOOG.history(period='1d', start='2015-1-1', end='2020-1-1')

# Visualize one example
GOOG_df.head()

# Save the close columns of each stock into new variables
MSFT = MSFT_df['Close']
TSLA = TSLA_df['Close']
AAPL = AAPL_df['Close']
AMZN = AMZN_df['Close']
GOOG = GOOG_df['Close']

# Concatenate all stocks close columns into one data frame
stocks_df = pd.concat([MSFT, TSLA, AAPL, AMZN, GOOG], axis='columns', join='inner')

# Rename the data frame columns with their corresponding tickers symbols
stocks_df.columns = ['MSFT', 'TSLA', 'AAPL', 'AMZN', 'GOOG']

# Visualize the new data frame
stocks_df.head()

# Get daily percentage change
stocks_df = stocks_df.pct_change().dropna()

# Visualize new data frame
stocks_df.head()

# Plot daily percentage change
stocks_df.plot(figsize=(20, 10), title="Daily Returns");

# Calculate cumulative returns
cumulative_returns = (1 + stocks_df).cumprod()

# Plot cumulative returns
cumulative_returns.plot(figsize=(20, 10), title="Cumulative Returns");

# Box plot to visually show risk
stocks_df.plot.box(figsize=(20, 10), title="Portfolio Risk");

# Calculate standard deviation for each stock
stocks_df.std()

# Calculate annualized standard deviation (252 trading days)
stocks_df.std() * np.sqrt(252)

# Calculate and plot the rolling standard deviation for each stock using a 30 trading day window
stocks_df.rolling(window=30).std().plot(figsize=(20, 10), title="30 Day Rolling Standard Deviation");

# Calculate annualized Sharpe Ratios
sharpe_ratios = (stocks_df.mean() * 252) / (stocks_df.std() * np.sqrt(252))
sharpe_ratios = sharpe_ratios.sort_values(ascending=False)
sharpe_ratios

# Visualize the Sharpe ratios as a bar plot
sharpe_ratios.plot(figsize=(20, 10), kind="bar", title="Sharpe Ratios");
