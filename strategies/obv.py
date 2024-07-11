import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def calculate_obv(close, volume):
    obv = np.zeros_like(volume)
    obv[0] = volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

class OBVStrategy(Strategy):
    obv_periods = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.obv = self.I(calculate_obv, self.data.Close, self.data.Volume)
        self.obv_sma = self.I(lambda: pd.Series(self.obv).rolling(self.obv_periods).mean())
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
            if self.obv[-1] > self.obv_sma[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.obv[-1] < self.obv_sma[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def obv_viz(data, obv_periods=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    obv = calculate_obv(data['Close'], data['Volume'])
    obv_sma = pd.Series(obv).rolling(window=obv_periods).mean()
    
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

    ax1.plot(data.index, data['Close'], label='Price', color='#00A86B')  # Flashier pine green
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, obv, label='OBV', color='purple')
    ax2.plot(data.index, obv_sma, label=f'OBV SMA({obv_periods})', color='yellow')
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.set_ylabel('OBV', fontproperties=font_properties, color='white')
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

    fig.suptitle('OBV Strategy Visualization', fontproperties=title_font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)






def run_obv(ticker, start_date, end_date, cash, commission, obv_periods, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    OBVStrategy.obv_periods = obv_periods
    OBVStrategy.stop_loss_pct = stop_loss_pct
    OBVStrategy.take_profit_pct = take_profit_pct
    OBVStrategy.enable_shorting = enable_shorting
    OBVStrategy.enable_stop_loss = enable_stop_loss
    OBVStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(OBVStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
