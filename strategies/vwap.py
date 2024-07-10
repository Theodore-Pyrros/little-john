import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

class VWAPStrategy(Strategy):
    vwap_periods = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.vwap = self.I(lambda: self.calculate_vwap(self.vwap_periods))
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def rolling_sum(self, array, n):
        result = np.full_like(array, np.nan)
        for i in range(n-1, len(array)):
            result[i] = np.sum(array[i-n+1:i+1])
        return result

    def calculate_vwap(self, n):
        pv = self.data.Close * self.data.Volume
        cumulative_pv = self.rolling_sum(pv, n)
        cumulative_volume = self.rolling_sum(self.data.Volume, n)
        return cumulative_pv / cumulative_volume

    def next(self):
        if self.position:
            if self.position_type == 'long':
                if self.enable_stop_loss and self.data.Close[-1] <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and self.data.Close[-1] >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.enable_stop_loss and self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
        if not self.position:
            if self.data.Close[-1] > self.vwap[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.data.Close[-1] < self.vwap[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def vwap_viz(data, vwap_periods=20):
    data = data[data['Volume'] > 0]    
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    cumulative_pv = (data['Close'] * data['Volume']).rolling(window=vwap_periods).sum()
    cumulative_volume = data['Volume'].rolling(window=vwap_periods).sum()
    vwap = cumulative_pv / cumulative_volume
    
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Price', color='blue')
    ax.plot(data.index, vwap, label=f'VWAP({vwap_periods})', color='red')
    ax.set_title('VWAP Visualization')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30)
    
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

def run_vwap(ticker, start_date, end_date, cash, commission, vwap_periods, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    VWAPStrategy.vwap_periods = vwap_periods
    VWAPStrategy.stop_loss_pct = stop_loss_pct
    VWAPStrategy.take_profit_pct = take_profit_pct
    VWAPStrategy.enable_shorting = enable_shorting
    VWAPStrategy.enable_stop_loss = enable_stop_loss
    VWAPStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(VWAPStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None