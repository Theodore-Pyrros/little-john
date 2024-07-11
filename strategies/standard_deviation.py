import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def rolling_std(array, n):
    result = np.full_like(array, np.nan)
    for i in range(n-1, len(array)):
        result[i] = np.std(array[i-n+1:i+1])
    return result

def rolling_mean(array, n):
    result = np.full_like(array, np.nan)
    for i in range(n-1, len(array)):
        result[i] = np.mean(array[i-n+1:i+1])
    return result

class StdDevStrategy(Strategy):
    std_period = 20
    std_multiplier = 2
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = self.data.Close
        self.sma = self.I(rolling_mean, close, self.std_period)
        self.std = self.I(rolling_std, close, self.std_period)
        self.upper = self.I(lambda: self.sma + self.std_multiplier * self.std)
        self.lower = self.I(lambda: self.sma - self.std_multiplier * self.std)
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

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
            if self.data.Close[-1] < self.lower[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.data.Close[-1] > self.upper[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def std_dev_viz(data, std_period=20, std_multiplier=2):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    sma = data['Close'].rolling(window=std_period).mean()
    std = data['Close'].rolling(window=std_period).std()
    upper = sma + std_multiplier * std
    lower = sma - std_multiplier * std

    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, facecolor='none')

    # Set transparent background
    fig.patch.set_alpha(0)
    ax1.set_facecolor('none')
    ax2.set_facecolor('none')

    # Remove the outline of the axes
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax1.plot(data.index, data['Close'], label='Price', color='pink')  # Flashier pine green
    ax1.plot(data.index, upper, label='Upper Band', color='cyan', linestyle='--')
    ax1.plot(data.index, lower, label='Lower Band', color='green', linestyle='--')
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, std, label='Standard Deviation', color='purple')
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.set_ylabel('Standard Deviation', fontproperties=font_properties, color='white')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')

    # Limit the number of x-ticks to 7
    xtick_locs = np.linspace(0, len(data) - 1, 7, dtype=int)
    xtick_labels = [data.iloc[i]['Date'].strftime('%Y-%m-%d') for i in xtick_locs]

    ax2.set_xticks(xtick_locs)
    ax2.set_xticklabels(xtick_labels, rotation=30, fontproperties=font_properties, color='white')
    
    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='white', labelsize=12)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('Standard Deviation Strategy Visualization', fontproperties=title_font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)






def run_standard_deviation(ticker, start_date, end_date, cash, commission, std_period, std_multiplier, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    StdDevStrategy.std_period = std_period
    StdDevStrategy.std_multiplier = std_multiplier
    StdDevStrategy.stop_loss_pct = stop_loss_pct
    StdDevStrategy.take_profit_pct = take_profit_pct
    StdDevStrategy.enable_shorting = enable_shorting
    StdDevStrategy.enable_stop_loss = enable_stop_loss
    StdDevStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(StdDevStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
