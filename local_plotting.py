# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:24:20 2024

@author: TedLove
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
from plotly.subplots import make_subplots

        
def bid_ask_quotes(T, M, N, v_max, d_ask_dict, d_bid_dict):
    t = np.linspace(0, T, M+1)
    v = np.linspace(1e-10, v_max, N)
    t, v = np.meshgrid(t, v)
    keys = [-10, -5, -2, 0, 2, 5, 10]
    
    dicts = [d_ask_dict, d_bid_dict]
    titles = ['Ask Quotes', 'Bid Quotes']
    for title, dictionary in zip(titles, dicts):
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{'type': 'surface'} for _ in range(4)] for _ in range(2)], 
            subplot_titles=[ rf"$\delta \text{{ from mid for inventory q=}} {k}$" for k in keys],  
            vertical_spacing=0.01,  # Decrease vertical spacing
            horizontal_spacing=0.01  # Decrease horizontal spacing
        )
        
        for i, key in enumerate(keys):
            row, col = divmod(i, 4)
            fig.add_trace(
                go.Surface(z=dictionary[key], x=t[0], y=v[:,0] ** .5 * np.sqrt(252) * 100, colorscale='Viridis', showscale=False),
                row=row + 1, col=col + 1
            )
            
        for i in range(1, len(keys) + 1):
            fig.layout[f'scene{i}']['xaxis']['title'] = 'Time'
            fig.layout[f'scene{i}']['yaxis']['title'] = 'Vol % (Annualised)'
            fig.layout[f'scene{i}']['zaxis']['title'] = 'Value'
            fig.layout[f'scene{i}'].update(aspectmode='cube')
            fig.layout[f'scene{i}'].update(camera_eye=dict(x=-1.8, y=-1.8, z=1.8)) 
        
        # Update layout options
        fig.update_layout(
            title=title,
            height=900, 
            width=1600
        )
        
        for annotation in fig['layout']['annotations']:
            annotation['y'] -= 0.05
        
        fig.show()


def backtest_algorithm(S_vec, pa, pb):
        
    x = np.linspace(0 , 1, len(S_vec))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = x , y=S_vec, name='Stock Price'))
    fig.add_trace(go.Scatter(x = x , y=pa, name='Bid Price' ))
    fig.add_trace(go.Scatter(x = x , y=pb , name = 'Ask Price'))
    fig.update_layout(
        title='1 observation of market making',
        height=900, 
        width=1600,
        xaxis_title='Time till eod' ,
        xaxis_title_font={'size': 18},
        xaxis_tickfont_size=18,
        yaxis_title='Price ($)',
        yaxis_title_font={'size': 18},
        yaxis_tickfont_size=18
    )
    
    fig.show()
    

        
