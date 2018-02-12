# -*- coding: utf-8 -*-
#   _        _
#    \_("/)_/
# 
# II Hackathon DXC CEIN 
# Santiago de Compostela, 09-10/02/2018
# 
# @author:  Fontenla Cadavid, Manuel
# @author:  Núñez Fernández, Adolfo  
# @author:  Rodríguez Veiga, Jorge  
# @author:  Reyes Valenzuela, Patricio  


# imports
# dashboard
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
# Pandas, plotly and numpy
import plotly.graph_objs as go
import pandas as pd
from pandas import Series
import numpy as np
# Statistics
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


# globals (to be avoided in future)
global data_value
global price
global data_social

# APP
app = dash.Dash()

# Classes
class Player():
    '''Class for Obradoiro's basketball players'''
    def __init__(self, id, name, picture=''):
        self.id = id
        self.name = name
        self.picture = picture

# Definitions
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

players_list = [
    Player('AlbertoCorby33','Alberto Corbacho', 'http://www.obradoirocab.com//webobra/cache/270x270_705a0c3495_33-corbacho.jpg'),
    Player('pepepozas4', 'Pepe Pozas', 'http://www.obradoirocab.com//webobra/cache/270x270_705a0c3495_44-pozas.jpg')
    #Player{'nachollovet', Nacho Llovet', http://www.obradoirocab.com//webobra/cache/270x270_705a0c3495_9-llovet.jpg'}
    ]

# Fixed default values (Some not used)
value = {'now':4,'next':5}
social = {'now':2,'next':3}
price = {'now':4,'next':13}

# Databases
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
# Statistics for players (value)
data_value = pd.read_csv('./data/player_date_game.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)
data_value = data_value.dropna(how="any", axis=0, subset=["game_id"])
# Social for players 
data_social = pd.read_csv('./data/player_date_twitter_sentiment.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)


# Function
def time_series(ts_dict, num_pred = 7, title="Efficiency"):
    '''Models and predicts time series from data'''
    data = []
    for k in ts_dict:
        ts = ts_dict[k]
        train, test = ts[1:len(ts)-num_pred], ts[len(ts)-num_pred:]

        # train autoregression
        model = AR(train, freq="W")
        model_fit = model.fit()

        # make predictions
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

        # Create a trace results
        predict = pd.concat([ts[len(ts)-8:len(ts)-7], predictions])

        line_predict = go.Scatter(x=predict.index, y=predict.values, name="prediccion " + k)# , marker={'color': 'rgb(0,255,0)'})
        line_hist = go.Scatter(x=train.index, y=train.values,        name="historicos " + k)

        data += [line_hist, line_predict]

    layout=go.Layout(title=title)
    figure=go.Figure(data=data,layout=layout)
    
    return({'figure': figure, 'curr_val': train[-1], 'first_pred': predictions[0]})

resume_plot = [
    {
        'data': [
            {'x': ['1'], 'y': [6], 'type': 'bar', 'name': 'Value'},
            {'x': ['2'], 'y': [9], 'type': 'bar', 'name': 'Social'},
            {'x': ['3'], 'y': price['now'], 'type': 'bar', 'name': '$'}
        ],
        'layout': {
            'title': 'Actual',
            'showticklabels':True,
            'xaxis':{'ticktext':['Value','Social','$'], 'tickvals':[1,2,3]}
        }
    },
    {
        'data': [
            {'x': ['1'], 'y': [6], 'type': 'bar', 'name': 'Value'},
            {'x': ['2'], 'y': [9], 'type': 'bar', 'name': 'Social'},
            {'x': ['3'], 'y': np.random.random_integers(0,40,1), 'type': 'bar', 'name': '$'}
        ],
        'layout': {
            'title': 'Estimado',
            'showticklabels':True,
            'xaxis':{'ticktext':['Value','Social','$'], 'tickvals':[1,2,3]}
        }
    }

]

# options for dropdown
dropdown_options = []
for pl in players_list:
    dropdown_options.append({'label': pl.name, 'value': pl.id})

# APP layout   
app.layout = html.Div(children=[
    html.H1(children='Cazatalentos', 
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'fontSize': '48'
        }),
     html.Div(id='leftblock', style={'float':'left','width':'20%'}, children=[
        html.Div(id='output-container'),
    
    dcc.Dropdown(
        id='dropdown-players',
        options=dropdown_options,
        value=players_list[0].id
    )]),
    html.Div(id='rightblock', style={'float':'left','width':'80%'}, children=[
        dcc.Graph(
            style={'float':'left','width':'50%', 'height':'300'},
            id='resume_plot_current',            
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            style={'float':'left','width':'50%', 'height':'300'},
            id='resume_plot_next',
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            style={'float':'left','width':'50%', 'height':'300'},
            id='value-series',
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            style={'float':'left','width':'50%', 'height':'300'},
            id='social-series',
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            style={'float':'left','width':'100%', 'height':'300'},
            id='index-series', 
            config={'displayModeBar': False}
        )
    ])
])


# Callbacks
@app.callback(Output('output-container', 'children'),[Input('dropdown-players', 'value')])
def update_picture(value):
    '''
    Gets the players picture
    Returns HTML for image
    '''

    aPlayer = [pl for pl in players_list if pl.id == value]
    player = aPlayer[0]
    return html.Img(id=player.id, src=player.picture)

@app.callback(Output('value-series', 'figure'), [Input('dropdown-players', 'value')])
def update_value_series_plot(value):
    '''
    Returns the plot of the value time series
    '''

    # Input data
    start_date = '2015-10-01'
    end_date = '2016-10-01'
    
    # Filter data
    # filter player
    player = data_value[data_value["twitter"]==value]
    #filter by efficiency and date
    efficiency = player['efficiency'][start_date:end_date]
    # clean data (NAN, INF)
    ts = efficiency.replace([np.inf, -np.inf], np.nan).dropna()
    # Generate time series plot
    return_value = time_series({'valor':ts},title="Valoracion")['figure']
    
    return return_value

@app.callback(Output('social-series', 'figure'), [Input('dropdown-players', 'value')])
def update_social_series_plot(value):
    '''
    Returns the plot of the social time series
    '''

    # Input data
    start_date = '2015-10-01'
    end_date = '2016-10-01'

    # Filter data
    # filter player
    player = data_social[data_social["twitter"]==value]
    # filter by sentiment and date
    positive = player['positive'][start_date:end_date]
    negative = player['negative'][start_date:end_date]
    # Generate time series plot
    return_value = time_series({'positivo':positive,'negativo':-negative},title="Sentimento")['figure']
    
    return return_value

@app.callback(Output('index-series', 'figure'), [Input('dropdown-players', 'value')])
def update_index_series_plot(value):
    '''
    Returns the plot of the social index time series
    '''

    # Input data
    start_date = '2015-10-01'
    end_date = '2016-10-01'

    # Filter data
    player = data_social[data_social["twitter"]==value]
    positive = player['positive'][start_date:end_date]
    negative = player['negative'][start_date:end_date]
    social_index = positive-negative
    return_value = time_series({'indice social':social_index},title="Índice social")['figure']
    
    return return_value

@app.callback(Output('resume_plot_current', 'figure'), [Input('dropdown-players', 'value')])
def update_current_bars(value):
    '''Updates bars for current data'''
    
    # Input data
    start_date = '2015-10-01'
    end_date = '2016-10-01'

    # Filter data
    player_value_data = data_value[data_value["twitter"]==value]
    efficiency = player_value_data['efficiency'][start_date:end_date]
    ts = efficiency.replace([np.inf, -np.inf], np.nan).dropna()
        
    player_social_data = data_social[data_social["twitter"]==value]
    positive = player_social_data['positive'][start_date:end_date]
    negative = player_social_data['negative'][start_date:end_date]
    ts2 = positive - negative

    return_value = {
        'data': [
            {'x': ['1'], 'y': [time_series({'a': ts})['curr_val']], 'type': 'bar', 'name': 'Valoracion'},
            {'x': ['2'], 'y': [time_series({'a': ts2})['curr_val']], 'type': 'bar', 'name': 'Social'},
            {'x': ['3'], 'y': [price['now']], 'type': 'bar', 'name': '$'}
        ],
        'layout': {
            'title': 'Actual',
            'showticklabels':True,
            'xaxis':{'ticktext':['Valoracion','Social','$'], 'tickvals':[1,2,3]}
        }
    }
    
    return return_value 

@app.callback(Output('resume_plot_next', 'figure'), [Input('dropdown-players', 'value')])
def update_next_bars(value):
    '''Updates bars for predictions'''

    # Input data
    start_date = '2015-10-01'
    end_date = '2016-10-01'

    # Filter data
    player_value_data = data_value[data_value["twitter"]==value]
    efficiency = player_value_data['efficiency'][start_date:end_date]
    ts = efficiency.replace([np.inf, -np.inf], np.nan).dropna()
        
    player_social_data = data_social[data_social["twitter"]==value]
    positive = player_social_data['positive'][start_date:end_date]
    negative = player_social_data['negative'][start_date:end_date]
    ts2 = positive - negative

    return_value = {
        'data': [
            {'x': ['1'], 'y': [time_series({'a': ts})['first_pred']], 'type': 'bar', 'name': 'Valoracion'},
            {'x': ['2'], 'y': [time_series({'a': ts2})['first_pred']], 'type': 'bar', 'name': 'Social'},
            {'x': ['3'], 'y': [price['next']], 'type': 'bar', 'name': '$'}
        ],
        'layout': {
            'title': 'Futuro',
            'showticklabels':True,
            'xaxis':{'ticktext':['Valoracion','Social','$'], 'tickvals':[1,2,3]}
        }
    }
           
    return return_value 

if __name__ == '__main__':
    app.run_server(debug=True)

    # _        _
    #  \_("/)_/
