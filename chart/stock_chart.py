# chart/stock_chart.py

import streamlit as st
import plotly.graph_objects as go
from data_handler import fetch_data_pv
import pandas as pd

def plot_stock_price_and_volume(ticker, start_date, end_date):
    data = fetch_data_pv(ticker, start_date, end_date)
    
    if data.empty:
        st.warning("No data available for the selected date range.")
        return None

    # Ensure the Datetime column is used as the index
    if 'Datetime' in data.columns:
        data.set_index('Datetime', inplace=True)
    
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Filter out days with zero volume
    data = data[data['Volume'] > 0]

    # Filter for trading days (Monday to Friday)
    data = data[data.index.dayofweek < 5]

    if data.empty:
        st.warning("No trading data available for the selected date range.")
        return None

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='blue'),
        hovertemplate='Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color='rgba(0,0,0,0.2)'),
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ))
    
    # Group by date and get the last data point of each day
    daily_data = data.groupby(data.index.date).last()
    
    fig.update_layout(
        title=f'{ticker} Stock Price and Trading Volume',
        xaxis=dict(
            title='Date',
            tickmode='array',
            tickvals=daily_data.index,
            ticktext=[date.strftime('%Y-%m-%d') for date in daily_data.index],
            rangeslider=dict(visible=False),
        ),
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    
    return fig

def display_stock_chart(ticker, start_date, end_date):
    fig = plot_stock_price_and_volume(ticker, start_date, end_date)
    if fig:
        st.plotly_chart(fig, use_container_width=True)