import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def calculate_adx(high, low, close, adx_period=14):
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        
        plus_dm[i] = max(h_diff, 0) if h_diff > l_diff else 0
        minus_dm[i] = max(l_diff, 0) if l_diff > h_diff else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr = pd.Series(tr).rolling(window=adx_period).sum()
    plus_dm = pd.Series(plus_dm).rolling(window=adx_period).sum()
    minus_dm = pd.Series(minus_dm).rolling(window=adx_period).sum()
    
    plus_di = 100 * plus_dm / tr
    minus_di = 100 * minus_dm / tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = pd.Series(dx).rolling(window=adx_period).mean()
    
    return plus_di, minus_di, adx

class ADXStrategy(Strategy):
    adx_period = 14
    adx_threshold = 25
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.plus_di, self.minus_di, self.adx = self.I(calculate_adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
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
            if self.adx[-1] > self.adx_threshold:
                if crossover(self.plus_di, self.minus_di):
                    self.buy()
                    self.entry_price = self.data.Close[-1]
                    self.position_type = 'long'
                elif crossover(self.minus_di, self.plus_di) and self.enable_shorting:
                    self.sell()
                    self.entry_price = self.data.Close[-1]
                    self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def adx_viz(data, adx_period=14, adx_threshold=25):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    plus_di, minus_di, adx = calculate_adx(data['High'], data['Low'], data['Close'], adx_period)
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

    ax2.plot(data.index, plus_di, label='+DI', color='green')
    ax2.plot(data.index, minus_di, label='-DI', color='red')
    ax2.plot(data.index, adx, label='ADX', color='purple')
    ax2.axhline(y=adx_threshold, color='gray', linestyle='--', label=f'ADX Threshold ({adx_threshold})')
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.set_ylabel('ADX / DI', fontproperties=font_properties, color='white')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')
    
    # Set y-axis limits for the second subplot
    y_min = min(plus_di.min(), minus_di.min(), adx.min(), adx_threshold)
    y_max = max(plus_di.max(), minus_di.max(), adx.max(), adx_threshold)
    ax2.set_ylim(max(0, y_min - 5), min(100, y_max + 5))

    # Limit the number of x-ticks to 7
    xtick_locs = np.linspace(0, len(data) - 1, 7, dtype=int)
    xtick_labels = [data.iloc[i]['Date'].strftime('%Y-%m-%d') for i in xtick_locs]
    
    ax1.set_xticks(xtick_locs)
    ax1.set_xticklabels(xtick_labels, rotation=30, fontproperties=font_properties, color='white')
    
    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='white', labelsize=12)
    
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('ADX Strategy Visualization', fontproperties=title_font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)




def run_adx(ticker, start_date, end_date, cash, commission, adx_period, adx_threshold, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    ADXStrategy.adx_period = adx_period
    ADXStrategy.adx_threshold = adx_threshold
    ADXStrategy.stop_loss_pct = stop_loss_pct
    ADXStrategy.take_profit_pct = take_profit_pct
    ADXStrategy.enable_shorting = enable_shorting
    ADXStrategy.enable_stop_loss = enable_stop_loss
    ADXStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(ADXStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
