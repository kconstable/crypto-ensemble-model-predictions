from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import main


def create_ensemble_models(df, ensemble_dict):
    """
  Add a long/short ensemble models to the df according to the specs in ensemble_dict
  Params:
    df: Dataframe output from combine_models
    ensemble_dict: A dictionary of ensemble models and weights
  """
    # short ensemble model: sum(model prediction * weights)
    df['ensemble_short'] = df[ensemble_dict['short']['models']].mul(ensemble_dict['short']['weights']).sum(1)

    # long ensemble model: sum(model prediction * weights)
    # need to shift the 3rd days' prediction to the same rows as the 1-day prediction
    df['ensemble_long'] = df[ensemble_dict['long']['models']].mul(ensemble_dict['long']['weights']).sum(1)
    df['ensemble_long'] = df['ensemble_long'].shift(-2)

    return df


def combine_models(models, ensemble_dict=None, dropna=True):
    """
    Combine backtest data (lstm models, prophet, ensemble)
    params:
      model_dict: dictionary of model info, weights for ensemble
      models: short, medium, long, prophet, ensemble
      dropna: boolean - remove nan or not
    """
    # combine models
    for i, m in enumerate(models):
        # get model hist
        if m != 'ensemble' and m != 'naive':

            df_hist = pd.read_pickle(f'{main.PATH}{main.TICKER}_{m}_backtest_ytd.pickle')

            if i == 0:
                dff = df_hist.copy()
            else:
                df_hist.drop(columns=['close'], inplace=True)
                dff = dff.join(df_hist, how='inner')
    if dropna:
        dff.dropna(inplace=True)

    if 'naive' in models or 'ensemble' in models:
        # add the naive model
        dff['naive'] = dff.close.shift(1)
        dff['naive_error'] = dff['naive'] - dff['close']
        dff['naive_cum'] = dff['naive_error'].cumsum()

    if 'ensemble' in models:
        # create ensemble models
        dff = create_ensemble_models(dff, ensemble_dict)
        dff['ensemble_long_error'] = dff['ensemble_long'] - dff['close']
        dff['ensemble_long_cum'] = dff['ensemble_long_error'].cumsum()

        dff['ensemble_short_error'] = dff['ensemble_short'] - dff['close']
        dff['ensemble_short_cum'] = dff['ensemble_short_error'].cumsum()

    if dropna:
        dff.dropna(inplace=True)

    return dff


def create_trading_strategy(ensemble_dict, models, tolerance=0.01, max_loss=-15, cash=40000,
                            start_date='2021-01-27', end_date='2021-09-28'):
    """
    Backtest a trading strategy based on long/short ensemble predictions.
    Params:
      ensemble_dict: A dictionary that defines the long/short models and their weights
      model_dict: A dictionary that define model properties
      models: A list of models to included
      tolerance: A percent that determines the return difference between the long/short models
      cash: The initial investment that is allocated between coin/cash
    """
    # get the model backtest data
    df = combine_models(models, True)
    df = df.loc[start_date:end_date, :].copy()

    # create the long/short ensemble models
    df = create_ensemble_models(df, ensemble_dict)

    # Calculate the signal strength
    # return difference between the prediction in 3-days vs the next day
    df['signal_strength'] = (df['ensemble_long'] - df['ensemble_short']) / df['naive']

    # Add the strength
    # if the strength is less than the tolerance => hold
    # if the strength is negative => sell
    # if the strength is positive => buy
    # if the price is trending down, and at a loss of 15% or more, sell
    df['signal'] = df['signal_strength'].apply(lambda x: 'hold' if abs(x) < tolerance else ('sell' if x < 0 else 'buy'))

    # init columns
    df['coin'] = 0
    df['cash'] = 0
    df['transaction'] = 0
    df['action'] = ''
    df['profit_loss'] = 0
    df['return_since_buy'] = 0.0

    # Evaluate trade decisions at each time step, calculate profit/loss,holdings
    start_date = df.index.min()
    last_buy = start_date
    if max_loss is None:
        max_loss = -np.inf

    for idx, row in df.iterrows():

        # get the previous date
        prev_date = idx - timedelta(days=1)

        # initial buy
        if idx == start_date:
            df.at[idx, 'signal'] = 'buy'
            df.at[idx, 'coin'] = row.close
            df.at[idx, 'cash'] = cash - row.close
            df.at[idx, 'transaction'] = -row.close
            df.at[idx, 'action'] = 'buy'

        # buy- if coin isn't already owned
        elif row.signal == 'buy' and df.loc[prev_date, 'coin'] == 0:
            df.at[idx, 'cash'] = df.loc[prev_date, 'cash'] - row.close
            df.at[idx, 'transaction'] = -row.close
            df.at[idx, 'coin'] = row.close
            df.at[idx, 'action'] = 'buy'
            last_buy = idx

        # sell -if coin is currently owned if there is a sell signal
        # Or, if the return since purchase is less than the max_loss
        # this corrects for the scenario where the price is trending down for
        # both short /long models, but where the long model price is still higher than the short price
        # In this scenario the model will hold until the short price is higher than the long price
        elif df.loc[prev_date, 'coin'] > 0 and (
                row.signal == 'sell' or df.loc[prev_date, 'return_since_buy'] <= max_loss):
            df.at[idx, 'cash'] = df.loc[prev_date, 'cash'] + row.close
            df.at[idx, 'transaction'] = row.close
            df.at[idx, 'coin'] = 0
            df.at[idx, 'action'] = 'sell'
            df.at[idx, 'profit_loss'] = row.close - df.loc[prev_date, 'coin']

        # hold (hold cash or coin)-wait for the next signal
        else:
            df.at[idx, 'action'] = 'hold'
            df.at[idx, 'cash'] = df.loc[prev_date, 'cash']
            df.at[idx, 'coin'] = df.loc[prev_date, 'coin']

        # update the return since last purchase if coin is currently owned
        if df.loc[idx, 'coin'] > 0:
            return_since_buy = round((df.loc[idx, 'close'] / df.loc[last_buy, 'close'] - 1) * 100, 2)
            df.at[idx, 'return_since_buy'] = return_since_buy

    # add totals
    df['total'] = df.coin + df.cash

    # add invested amount
    df['invested'] = df.apply(lambda row: row.close if row.coin > 0 else 0, axis=1)

    return df[['close', 'ensemble_short', 'ensemble_long', 'signal_strength', 'signal', 'action', 'cash', 'coin',
               'transaction', 'total', 'profit_loss', 'invested', 'return_since_buy']]


def get_invested_dates(df):
    """
  Returns a dataframe of trades,dates,transaction amounts and profit/loss
  Params:
    df: The output from create_trading_strategy
  """
    # get dates with transactions
    df_trans = df[df.transaction != 0].copy()

    # move the index to the from_date field
    df_trans['from_date'] = df_trans.index

    # create the to_date (the next date in the df)
    df_trans['to_date'] = df_trans.from_date.shift(-1)

    # the row will have a null for to_date because of the
    # shifted rows. Replace that value with the latest date in the df
    idx = df_trans[df_trans.to_date.isnull()].index
    df_trans.loc[idx, 'to_date'] = df.index.max()

    return df_trans


def plot_trading_strategy(df, annotate=None):
    """
    Plot the trading strategy (close prices, long/short-ensemble models,
    invested periods, and total amount invested in coin/cash)
    Params:
      df: a dataframe of profit/loss. The output from create_trading_strategy
      annotate: A dictionary to define where to put the buy/sell annotations
                {date:buy-date,shift:y-axis offset}
    """
    # calculate profit/loss, return and the number of trades
    profit_loss = df.profit_loss.sum()
    return_strategy = round(profit_loss / df.loc[df.index.min(), 'total'] * 100, 1)
    return_hold = round((df.loc[df.index.max(), 'close'] / df.loc[df.index.min(), 'close'] - 1) * 100, 1)

    # number of trades
    no_trades = min(df[df.action == 'buy'].shape[0] - 1, df[df.action == 'sell'].shape[0])

    # get the dates of investment in coin
    df_trans = get_invested_dates(df)

    # create the plots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # total invested
    fig.add_trace(
        go.Scatter(
            name='Total Portfolio Value (Cash+Coin)',
            x=df.index,
            y=df.total,
            line=dict(color='darkgrey', width=4, dash='dot'),
        ), secondary_y=False
    )

    # add the investment periods as shapes
    for idx, row in df_trans.iterrows():
        if row.coin > 0:
            fig.add_shape(
                dict(
                    name='Invested',
                    type='rect',
                    fillcolor='skyblue',
                    x0=row.from_date,
                    x1=row.to_date,
                    y0=0,
                    y1=1,
                    opacity=0.1,
                ), secondary_y=False
            )

    # close price
    fig.add_trace(
        go.Scatter(
            name='Closing Price',
            x=df.index,
            y=df.close,
            line_color='#536872',
            fill='tozeroy',
        ), secondary_y=False
    )

    # short-ensemble-model
    fig.add_trace(
        go.Scatter(
            name='Ensemble Model (1day)',
            x=df.index,
            y=df.ensemble_short,
            line=dict(color='cornflowerblue', width=3)
        ), secondary_y=False
    )

    # long-ensemble-model
    fig.add_trace(
        go.Scatter(
            name='Ensemble Model (3day)',
            x=df.index,
            y=df.ensemble_long,
            line=dict(color='#ff7f0e', width=3)
        ), secondary_y=False
    )

    # add annotations
    txt = f'Profit/Loss:${profit_loss:,}|Strategy Return:{return_strategy}%|Buy & Hold Return:{return_hold}%|Trades:{no_trades}'

    # add subtitle
    fig.add_annotation(
        text=txt,
        xref="paper", yref="paper",
        x=-0.05, y=1.12,
        showarrow=False
    )

    # add buy/sell labels if provided
    if annotate is not None:
        # get buy/sell dates and prices
        buy_date = df_trans.loc[annotate['date'], 'from_date']
        buy_price = df_trans.loc[buy_date, 'close']
        sell_date = df_trans.loc[annotate['date'], 'to_date']
        sell_price = df_trans.loc[sell_date, 'close']

        fig.add_annotation(
            x=buy_date,
            y=buy_price,
            text="Buy",
            showarrow=True,
            yshift=annotate['shift'],
            arrowhead=1,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e"
        )
        fig.add_annotation(
            x=sell_date,
            y=sell_price,
            text='Sell',
            arrowhead=True,
            yshift=annotate['shift'],
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e"
        )

    fig.update_shapes(dict(xref='x', yref='paper'))
    fig.update_layout(
        title='Trading Strategy Evaluation',
        template='plotly_white',
        width=1000,
        height=500,
        legend=dict(
            yanchor='bottom',
            y=0.01,
            xanchor='right',
            x=0.935,
        )
    )

    return fig


def plot_news_count(df_final, end_date):
    """
    Plots the bitcoin news volume and moving average volume
      df_final : the consolidated dataframe of market data
    """
    # subset by dates
    df = df_final[:end_date].copy()

    # create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # daily volume
    fig.add_trace(
        go.Scatter(
            name='Daily News Volume',
            x=df.index,
            y=df.news_count,
            fill='tozeroy',
            opacity=0.5,
            marker_color='#a8b8d0',
            marker_line_color='#a8b8d0'
        ), secondary_y=False
    )

    # Moving average daily volume
    fig.add_trace(
        go.Scatter(
            name='20-Day Moving Average News Volume',
            x=df.index,
            y=df.ma_news_count,
            line_color='gold'
        ), secondary_y=False
    )

    # moving average sentiment of the news titles
    fig.add_trace(
        go.Scatter(
            name='Upper-Bollinger (Sentiment-article titles)',
            x=df.index,
            y=df['b-upper-ma_sentiment_title'],
            line_color='orange',
        ), secondary_y=True
    )

    # moving average sentiment of the news titles
    fig.add_trace(
        go.Scatter(
            name='5-Day Moving Average Sentiment (article titles)',
            x=df.index,
            y=df['b-middle-ma_sentiment_title'],
            line={'color': 'orange', 'width': 3},
            fill='tonexty',
            opacity=0.1
        ), secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            name='Lower-Bollinger (Sentiment-article titles)',
            x=df.index,
            y=df['b-lower-ma_sentiment_title'],
            line_color='orange',
            fill='tonexty',
            opacity=0.1
        ), secondary_y=True
    )

    fig.update_layout(title='News Count & Sentiment (multiple sources)',
                      template='plotly_white',
                      width=1000,
                      height=500,
                      legend=dict(
                          yanchor="top",
                          y=1,
                          xanchor="left",
                          x=0.01)
                      )
    fig.update_yaxes(title_text="News Count", secondary_y=False)
    fig.update_yaxes(title_text="News Sentiment (article titles)", secondary_y=True, range=[-1, 1])
    return fig


def plot_sentiment(df_all, end_date):
    """
  Plots the daily weighted average sentiment and moving average sentiment
  Params:
    df_all: a dataframe containing all data (not-the final one, need ma_sentiment to be included)
    end_date  : to this date if provided
  """

    # filter by date if provided
    df = df_all.loc[:end_date, :].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['weighted_sentiment'],
                   name='Daily Sentiment',
                   line_color='#a8b8d0',
                   opacity=0.5,
                   )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ma_sentiment_20'],
            name='20-Day Moving Average Sentiment',
            line={'color': 'orange', 'width': 3}
        )
    )

    fig.update_layout(template='plotly_white',
                      height=500,
                      width=700,
                      title='News Sentiment (Forbes-Full Article Text)',
                      legend=dict(
                          yanchor="bottom",
                          y=0.1,
                          xanchor="left",
                          x=0.2)
                      )
    return fig


def plot_fear_greed_index(df_final, end_date):
    """
    Plots the fear and greed index with bollinger bands
    params:
      df: a dataframe containing the fear/greed index
      from_: start date
      to_: end date
    """

    df = df_final.loc[:end_date, :].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name='Bitcoin Fear-Greed Index',
            x=df.index,
            y=df.idx_fear_greed,
            line_color='orange'
        )
    )

    fig.update_layout(
        title='Bitcoin Fear-Greed Index',
        template='plotly_white',
        width=800,
        height=500
    )
    return fig


def plot_google_trends(df_all, end_date):
    """
  Plots google trend data
  params:
    df_all: a dataframe of all market data (not final)
    end_date: end date

  """

    df = df_all.loc[:end_date, :].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name='Search Volume',
            x=df.index,
            y=df.google_trends,
            line_color='orange',
            fill='tozeroy'
        )
    )
    fig.update_layout(
        title='Google Trends: Search Term Volumes',
        template='plotly_white',
        width=1000,
        height=500,
        yaxis_title='Number of searches per day'
    )
    return fig


# updated in practicum 2 (added stochastic oscillator)
def plot_technicals(df_all, end_date):
    """
  Plots technical indicators
  Input:
    df: a dataframe with technical indicators
    end_date: filters by end date
  """

    # filter by end date
    df = df_all.loc[:end_date, :].copy()

    # make subplots
    fig = make_subplots(rows=2, cols=2,
                        shared_xaxes=True,
                        subplot_titles=('Bollinger Bands', 'MACD', 'Stochastic Oscillator', 'RSI'),
                        vertical_spacing=0.07)

    # bollinger bands
    # ********************************************************************************************
    fig.add_trace(go.Scatter(x=df.index, y=df['b-upper'], name='Bollinger-Upper', line_color="#DFEBF9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['b-lower'], name='Bollinger-Lower', line_color='#DFEBF9', fill='tonexty'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.close, name='Closing Price', line_color='#a8b8d0'), row=1, col=1)

    # MACD
    # ********************************************************************************************
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line_color='#F7DAC6'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='MACD Signal', line_color='#E68A4C'), row=1, col=2)
    try:
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD Hist', marker_color='#E06D1F',
                             marker_line_color='#E06D1F'), row=1, col=2)
    except:
        # skip mcad histogram for crypto
        print('')

    # stochastic oscillator
    fig.add_trace(
        go.Scatter(
            name='stoch_K',
            x=df.index,
            y=df.stoch_K,
            marker_color='blue'
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            name='stoch_D',
            x=df.index,
            y=df.stoch_D,
            marker_color='skyblue'
        ), row=2, col=1
    )

    # # RSI
    # ********************************************************************************************
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line_color='#F00F3C'), row=2, col=2)
    fig.add_shape(type='rect',
                  x0=min(df.index),
                  x1=max(df.index),
                  y0=30.0,
                  y1=70.0,
                  line=dict(color='#F00F3C'),
                  fillcolor='#F00F3C',
                  opacity=0.1,
                  row=2, col=2)
    fig.update_shapes(dict(xref='x', yref='y'), row=2, col=1)

    # Set template
    fig.update_layout(template='plotly_white', width=1000, height=800, title='Technical Indicators')
    return fig
