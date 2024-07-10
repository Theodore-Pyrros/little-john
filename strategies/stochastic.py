import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def calculate_stochastic(high, low, close, stoch_k, stoch_d):
    k = np.zeros_like(close)
    d = np.zeros_like(close)
    
    for i in range(stoch_k, len(close)):
        low_min = np.min(low[i-stoch_k+1:i+1])
        high_max = np.max(high[i-stoch_k+1:i+1])
        k[i] = 100 * (close[i] - low_min) / (high_max - low_min)
    
    d = np.convolve(k, np.ones(stoch_d)/stoch_d, mode='same')
    
    return k, d

class StochStrategy(Strategy):
    stoch_k = 14
    stoch_d = 3
    stoch_overbought = 80
    stoch_oversold = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.k, self.d = self.I(calculate_stochastic, self.data.High, self.data.Low, self.data.Close, 
                                self.stoch_k, self.stoch_d)
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
            if self.k[-1] < self.stoch_oversold and self.k[-1] > self.d[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.k[-1] > self.stoch_overbought and self.k[-1] < self.d[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def stoch_viz(data, stoch_k=14, stoch_d=3, stoch_overbought=80, stoch_oversold=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    k, d = calculate_stochastic(data['High'], data['Low'], data['Close'], stoch_k, stoch_d)
    data['k'] = k
    data['d'] = d
    data['Date'] = data['Datetime'].dt.date
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

    ax1.plot(data.index, data['Close'], label='Price', color='pink')
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, data['k'], label='%K', color='yellow')
    ax2.plot(data.index, data['d'], label='%D', color='cyan')
    ax2.axhline(stoch_overbought, color='red', linestyle='--', label='Overbought')
    ax2.axhline(stoch_oversold, color='red', linestyle='--', label='Oversold')
    ax2.set_ylabel('Stochastic Oscillator', fontproperties=font_properties, color='white')
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')
    ax2.set_ylim(-5, 105)  # Set y-axis limits for the Stochastic Oscillator

    ax1.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30, fontproperties=font_properties, color='white')

    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='white', labelsize=12)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('Stochastic Oscillator Strategy Visualization', fontproperties=title_font_properties, color='white')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)




def run_stochastic(ticker, start_date, end_date, cash, commission, stoch_k, stoch_d, stoch_overbought, stoch_oversold, 
                       stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    StochStrategy.stoch_k = stoch_k
    StochStrategy.stoch_d = stoch_d
    StochStrategy.stoch_overbought = stoch_overbought
    StochStrategy.stoch_oversold = stoch_oversold
    StochStrategy.stop_loss_pct = stop_loss_pct
    StochStrategy.take_profit_pct = take_profit_pct
    StochStrategy.enable_shorting = enable_shorting
    StochStrategy.enable_stop_loss = enable_stop_loss
    StochStrategy.enable_take_profit = enable_take_profit
    
    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(StochStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
