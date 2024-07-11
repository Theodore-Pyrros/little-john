import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def ema(data, period):
    alpha = 2 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

class MACDStrategy(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = self.data.Close
        def macd(macd_fast, macd_slow, macd_signal):
            fast_ema = ema(close, macd_fast)
            slow_ema = ema(close, macd_slow)
            macd_line = fast_ema - slow_ema
            signal_line = ema(macd_line, macd_signal)
            return macd_line, signal_line
        self.macd_line, self.signal_line = self.I(macd, self.macd_fast, self.macd_slow, self.macd_signal)
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
            if crossover(self.macd_line, self.signal_line):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.signal_line, self.macd_line) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def macd_viz(data, macd_fast=12, macd_slow=26, macd_signal=9):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    data['ema_fast'] = data['Close'].ewm(span=macd_fast, adjust=False).mean()
    data['ema_slow'] = data['Close'].ewm(span=macd_slow, adjust=False).mean()
    data['macd'] = data['ema_fast'] - data['ema_slow']
    data['signal'] = data['macd'].ewm(span=macd_signal, adjust=False).mean()
    data['histogram'] = data['macd'] - data['signal']
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

    ax1.plot(data.index, data['Close'], label='Price', color='#00A86B')  # Flashier pine green
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, data['macd'], label='MACD', color='green')
    ax2.plot(data.index, data['signal'], label='Signal', color='purple')
    ax2.bar(data.index, data['histogram'], label='Histogram', color='gray', alpha=0.3)
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.set_ylabel('MACD', fontproperties=font_properties, color='white')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')

    # Limit the number of x-ticks to 7
    xtick_locs = np.linspace(0, len(data) - 1, 7, dtype=int)
    xtick_labels = [data.iloc[i]['Date'].strftime('%Y-%m-%d') for i in xtick_locs]

    ax2.set_xticks(xtick_locs)
    ax2.set_xticklabels(xtick_labels, fontproperties=font_properties, color='white')
    
    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='white', labelsize=12)
    
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('MACD Strategy Visualization', fontproperties=title_font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)






def run_macd(ticker, start_date, end_date, cash, commission, macd_fast, macd_slow, macd_signal, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    MACDStrategy.macd_fast = macd_fast
    MACDStrategy.macd_slow = macd_slow
    MACDStrategy.macd_signal = macd_signal
    MACDStrategy.stop_loss_pct = stop_loss_pct
    MACDStrategy.take_profit_pct = take_profit_pct
    MACDStrategy.enable_shorting = enable_shorting
    MACDStrategy.enable_stop_loss = enable_stop_loss
    MACDStrategy.enable_take_profit = enable_take_profit
    
    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(MACDStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
