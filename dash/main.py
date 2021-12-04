# import scripts
import functions as f
import layout
import pandas as pd

# constants
PATH = 'data/'
TICKER = 'BTC'

# used for plotting
model_dict = {
    'short-1day': {'col_name': 'lstm_short_1day', 'name': 'LSTM-Short (1day)', 'weight': 0.20,
                   'color': 'cornflowerblue'},
    'short-3day': {'col_name': 'lstm_short_3day', 'name': 'LSTM-Short (3day)', 'weight': 0.20,
                   'color': 'cornflowerblue'},
    'medium-1day': {'col_name': 'lstm_medium_1day', 'name': 'LSTM-Medium (1day)', 'weight': 0.20, 'color': 'goldenrod'},
    'medium-3day': {'col_name': 'lstm_medium_3day', 'name': 'LSTM-Medium (3day)', 'weight': 0.20, 'color': 'goldenrod'},
    'long-1day': {'col_name': 'lstm_long_1day', 'name': 'LSTM-Long (1day)', 'weight': 0.20, 'color': '#ff7f0e'},
    'long-3day': {'col_name': 'lstm_long_3day', 'name': 'LSTM-Long (3day)', 'weight': 0.20, 'color': '#ff7f0e'},
    'prophet-1day': {'col_name': 'prophet_1day', 'name': 'Prophet (1day)', 'weight': 0.20, 'color': 'skyblue'},
    'prophet-3day': {'col_name': 'prophet_3day', 'name': 'Prophet (3day)', 'weight': 0.20, 'color': 'skyblue'},
    'ensemble-1day': {'col_name': 'ensemble_short', 'name': 'Ensemble (1day)', 'color': 'crimson'},
    'ensemble-3day': {'col_name': 'ensemble_long', 'name': 'Ensemble (3day)', 'color': 'crimson'},
}

# creates the ensemble model (short/long)
ensemble_dict = {
    'short': {'models': ['lstm_short_1day', 'naive'], 'weights': [0.80, 0.20]},
    'long': {'models': ['lstm_long_3day', 'prophet_3day'], 'weights': [0.80, 0.20]}
}

# Get the data feed
# finalized data after feature selection
df_final = pd.read_pickle(f'{PATH}{TICKER}_market_data_final.pickle')

# all consolidated data (some features not selected are used for plotting)
df_all = pd.read_pickle(f'{PATH}{TICKER}_market_data.pickle')


if __name__ == '__main__':
    # run the dash app
    layout.app.run_server(debug=True)



