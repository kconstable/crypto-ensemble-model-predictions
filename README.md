# An ensemble method to predict closing prices for bitcoin
In this project, I developed 1-day and 3-day ensemble models that consist of LSTM and Facebook prophet models that predict the price of bitcoin. 44 Features are used in the LSTM models consisting of bitcoin spot and futures prices, technical indicators, commodity prices,  market indexes, bitcoin news volume, bitcoin web searchers, and market sentiment towards bitcoin and crypto. A full summary of this project is included in the [repository](https://github.com/kconstable/crypto-ensemble-model-predictions/blob/main/Summary-LSTM-ensemble-bitcoin-price-predictions.pdf).

A simple trading strategy that takes advantage of price differentials between the long and short models was developed and back-tested on the testing data set from January 2021 to September 2021.  The model executed 6 trades during the period and resulted in a 50% return vs 20% for a buy-and-hold-strategy for the same period.  This is not investment advice, markets are complex and there's no reason to believe this performance could be repeated!


## Overview
An ensemble method was developed by combining a long-short-term-memory (LSTM) Model, a Facebook Prophet time-series forecasting model, and a naive model that always predicts tomorrow's closing price is equal to today's closing price.  The naive model was introduced to add more weight to the most current closing price. Two LSTM models were developed which take a different number of input days.  The short-LSTM model predicts tomorrow's price based on the previous 25-days of feature data, and the long-LSTM model predicts the next 3-days of prices based on the last 100-days of feature data.  The short model captures short-term trends while the long model does a better job at detecting longer-term patterns.

**Short-Model (1-day prediction)**
+ LSTM Short Model Weight (60%) 
+ Prophet Weight (20%)
+ Naive Weight (20%)

**Long-Model (3-day prediction)**
+ LSTM Long Model Weight (80%)
+ Prophet Model Weight (20%)

![image](https://user-images.githubusercontent.com/1649676/143141220-9c903950-8119-4cf2-b802-44d4cbc2e45e.png)



## Data Aquisition & Analysis
Data was aquired from six different sources consisting of public APIs and via web-scrapping. [Alpha Vantage](https://www.alphavantage.co/) was used to pull commodities, market indexes, and FX rates. Bitcoin spot and futures prices were sourced from [Gate.io](https://www.gate.io).  [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) publishes a crypto fear-and-greed index which was included in the LSTM models as as feature.  Google Trends was used to create a time-series of the volume of bitcoin related web searches.  The google news API provided the volume of Bitcoin related news articles, and [forbes.com](https://www.forbes.com) was used as a source of Bitcoin related articles for which sentiment was calculated. Bitcoin prices underwent rapid growth, from a low of ~4k to a high of ~63k during the period, resulting in a skewed/bi-model distribution of prices. The price data was log-transformed to make it stationary and features were max-min-scaled to convert each to a fixed scale.


![image](https://user-images.githubusercontent.com/1649676/137164160-713777d0-516d-4432-af37-1f3de06aa9bb.png)


## Feature Selection
Economic and technical indicators, commodity prices, market indexes, FX rates, market sentiment and Bitcoin news volume were considered for inclusion in the LTSM model.  Lagged versions of each indicator were evaluated to determine if values of up to 120 days prior to current had a stronger influence on current prices.  Recursive Feature Elimination (RFE) with a random forest regressor was used to select features by influence on closing prices. In total, 76 features were considered, and the RFE process selected the top 44 for inclusion in the model.
+ Bitcoin Spot Prices (open, high, low, close, volume) (5)
+ Bitcoin Futures Prices (open, high, low, close)(4)
+ Crypto Currency Prices (ETH, DOGE, LTC) (3)
+ Bitcoin Greed and Fear Index (1)
+ Sentiment of Bitcoin articles (full-text) from Forbes.com (Bollinger bands for the 10-day moving average score)(3)
+ Sentiment of Bitcoin articles (title of article, any source) from google news (Bollinger bands for the 10-day moving average score)(3)
+ Number of news article written about Bitcoin from google news (Bollinger Bands on total number of articles)(4)
+ Number of Bitcoin related search queries on google trends (Bollinger Bands on the number of searches)(4)
+ Market Indexes  (Energy Sector, The Nasdaq, Volatility index –lagged 110 days) (3)
+ Commodities (Natural gas prices, Oil prices lagged 110-days ) (2)
+ FX Rates (USD-EUR, USD-GBP) (2)
+ Technical Indicators on Bitcoin spot prices (RSI, Bollinger Bands, MACD, Stochastic Oscillator) (10)

![image](https://user-images.githubusercontent.com/1649676/143145639-281108ac-b9fa-40cd-8730-2359f8d1bf5e.png)




## LSTM Model
The LSTM models were built using Keras/Tensorflow, and the hyper-parameters were optimized using Keras tuner. The data was split into train (80%) and test (20%) sets and 10% of the training data was withheld as an out-of-sample validation set.  Early stopping was utilized to determine the number of epochs used in training.  

![image](https://user-images.githubusercontent.com/1649676/143146099-f69b1b40-e2af-45e4-99f8-45a8d7c9d9ad.png)

## Plotly/Dash Application
A simple interactive dashboard to navigate features and predictions is provided.  The dash files are contained in the dash folder
![image](https://user-images.githubusercontent.com/1649676/144723252-6db9a74b-0202-4fa1-8e77-c20690fc3449.png)


## File Structure
The data files in the repository are organized as follows:
B+ sentiment: web-scrapping, google-news, google-trends, sentiment calculation
+ data_aquisition: data collection, analysis, consolidation, 
+ feature_engineering: feature engineering, evaluation, and selection
+ models: data pre-processing, LSTM model training, optimization, ensemble construction, trading-back-tests
+ Data: Contains all consolidated data files
  +  BTC_market_data - all feature data
  +  BTC_market_data_shifted - all feature data plus shifted features
  +  BTC_market_data_final - final data file used for training with selected features
  +  BTC_news - full news articles from forbes with sentiment
  +  BTC_news_volume- news from all sources from google news
  +  BTC_news_counts - news from all sources from google news - with calculated metrics and sentiment
  +  BTC_short_backtest_ytd - backtest of the short LSTM model
  +  BTC_long_backtest_ytd - backtest of the long LSTM model
  +  BTC_prophet_backtest_ytd - backtest of the facebook prophet model
+ Dash: dash application files
  + main.py - main control flow
  + layout.py - controls the application layout
  + functions.py - data and plotting functions

