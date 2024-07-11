import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

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


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def vwap_viz(data, vwap_periods=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    cumulative_pv = (data['Close'] * data['Volume']).rolling(window=vwap_periods).sum()
    cumulative_volume = data['Volume'].rolling(window=vwap_periods).sum()
    vwap = cumulative_pv / cumulative_volume
    
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='none')

    # Set transparent background
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # Remove the outline of the axes
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(data.index, data['Close'], label='Price', color='green')  # Flashier pine green
    ax.plot(data.index, vwap, label=f'VWAP({vwap_periods})', color='purple')
    
    ax.set_title('VWAP Visualization', fontproperties=title_font_properties, color='white')
    ax.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax.set_ylabel('Price', fontproperties=font_properties, color='white')
    
    # Limit the number of x-ticks to 7
    xtick_locs = np.linspace(0, len(data) - 1, 7, dtype=int)
    xtick_labels = [data.iloc[i]['Date'].strftime('%Y-%m-%d') for i in xtick_locs]

    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels, fontproperties=font_properties, color='white')
    
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties)

    ax.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax.grid(False, axis='x')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)




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
