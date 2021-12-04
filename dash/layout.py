from datetime import date

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import main
import functions as f

# get today's date
DATE_START = date(2021, 1, 27)
DATE_END = date.today()

# start the dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.config['suppress_callback_exceptions'] = True

# LAYOUT ------------------------------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        # dcc.Store(id="store"),
        html.H1("Crypto Price Predictions"),
        html.Hr(),

        dbc.Tabs(
            [
                dbc.Tab(label="Predictions", tab_id="predictions"),
                dbc.Tab(label="Technicals", tab_id="technicals"),
                dbc.Tab(label="Sentiment", tab_id='sentiment'),
                dbc.Tab(label='News', tab_id='news'),
                dbc.Tab(label='Trends', tab_id='trends')
            ],
            id="tabs",
            active_tab="predictions",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)


# CALLBACKS--------------------------------------------------------------------------------------------------
# tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    if active_tab == "predictions":
        return html.Div([
            dbc.Row(
                [
                    dbc.Col(html.H6("Cash"), width={'size': 1, 'order': 1}),
                    dbc.Col(dcc.Input(id='cash', type='number', min=40000, max=100000, step=5000, value=40000),
                            width={'size': 2, 'order': 2}),
                    dbc.Col(html.H6("Date Range"), width={'size': 1, 'order': 3}),
                    dbc.Col(dcc.DatePickerSingle(
                        id='date-start',
                        min_date_allowed=date(2018, 1, 1),
                        max_date_allowed=DATE_END,
                        initial_visible_month=DATE_START,
                        date=DATE_START
                    ), width={'size': 2, 'order': 4}
                    ),
                    dbc.Col(dcc.DatePickerSingle(
                        id='date-end',
                        min_date_allowed=date(2018, 1, 1),
                        max_date_allowed=DATE_END,
                        initial_visible_month=DATE_END,
                        date=DATE_END
                    ), width={'size': 2, 'order': 5}
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.H6("Tolerance(%)"), width={'order': 1, 'size': 1}),
                    dbc.Col(dcc.Input(id='tolerance', type='number', min=1, max=10, step=1, value=1),
                            width={'size': 2, 'order': 2}),
                    dbc.Col(html.H6("Max Loss(%)"), width={'size': 2, 'order': 3}),
                    dbc.Col(dcc.Input(id='max-loss', type='number', min=-25, max=0, step=1, value=-15),
                            width={'size': 2, 'order': 4})
                ]
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(dcc.Graph(id='plot-predictions'), width=12)
            )
        ])
    elif active_tab == "technicals":
        return html.Div([
            dbc.Row([
                dbc.Col(html.H6("End Date"), width=1),
                dbc.Col(dcc.DatePickerSingle(
                    id='date-end',
                    min_date_allowed=date(2021, 1, 1),
                    max_date_allowed=DATE_END,
                    initial_visible_month=DATE_END,
                    date=DATE_END
                ), width=5),

            ]),
            dbc.Row(
                dbc.Col(dcc.Graph(id='plot-technicals'), width=12)
            )
        ])

    elif active_tab == 'news':
        return html.Div([
            dbc.Row([
                dbc.Col(html.H6("End Date"), width=1),
                dbc.Col(dcc.DatePickerSingle(
                    id='date-end',
                    min_date_allowed=date(2021, 1, 1),
                    max_date_allowed=DATE_END,
                    initial_visible_month=DATE_END,
                    date=DATE_END
                ), width=5
                )
            ]
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id='plot-news'), width=12)
            )
        ])

    elif active_tab == 'sentiment':
        return html.Div([
            dbc.Row([
                dbc.Col(html.H6("End Date"), width=1),
                dbc.Col(dcc.DatePickerSingle(
                    id='date-end',
                    min_date_allowed=date(2021, 1, 1),
                    max_date_allowed=DATE_END,
                    initial_visible_month=DATE_END,
                    date=DATE_END
                ), width=5),

            ]),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id='plot-fear-greed'), width=6),
                    dbc.Col(dcc.Graph(id='plot-articles'), width=6),
                ]
            )
        ])

    else:
        return html.Div([
            dbc.Row([
                dbc.Col(html.H6("End Date"), width=1),
                dbc.Col(dcc.DatePickerSingle(
                    id='date-end',
                    min_date_allowed=date(2021, 1, 1),
                    max_date_allowed=DATE_END,
                    initial_visible_month=DATE_END,
                    date=DATE_END
                ), width=5),

            ]),
            dbc.Row(
                dbc.Col(dcc.Graph(id='plot-trends'), width=12),
            )
        ])


@app.callback(
    Output("plot-predictions", "figure"), [
        Input("date-start", "date"),
        Input("date-end", "date"),
        Input('cash', 'value'),
        Input('tolerance', 'value'),
        Input('max-loss', 'value')]
)
def plot_predictions(start_date, end_date, cash, tolerance, max_loss):
    # get trading strategy
    tol = float(tolerance / 100)
    loss = float(max_loss)
    dff = f.create_trading_strategy(main.ensemble_dict, ['short', 'long', 'prophet', 'naive'], tolerance=tol,
                                    max_loss=loss, cash=cash, start_date=start_date, end_date=end_date)
    return f.plot_trading_strategy(dff)


@app.callback(
    Output('plot-news', 'figure'), Input('date-end', 'date')
)
def plot_news(end_date):
    return f.plot_news_count(main.df_final, end_date)


@app.callback(
    [Output('plot-articles', 'figure'), Output('plot-fear-greed', 'figure')],
    Input('date-end', 'date')
)
def plot_news(end_date):
    return f.plot_sentiment(main.df_all, end_date), f.plot_fear_greed_index(main.df_final, end_date)


@app.callback(
    Output('plot-trends', 'figure'),
    Input('date-end', 'date')
)
def plot_trends(end_date):
    return f.plot_google_trends(main.df_all, end_date)


@app.callback(
    Output('plot-technicals', 'figure'),
    Input('date-end', 'date')
)
def plot_technicals(end_date):
    return f.plot_technicals(main.df_all, end_date)
