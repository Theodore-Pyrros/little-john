import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

def calculate_atr(high, low, close, atr_period=14):
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    return atr

class ATRStrategy(Strategy):
    atr_period = 14
    atr_multiplier = 2
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.prev_close = self.I(lambda: self.data.Close)
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def next(self):
        current_price = self.data.Close[-1]
        prev_close = self.prev_close[-2]  # Use the second-to-last value
        current_atr = self.atr[-1]

        if self.position:
            if self.position_type == 'long':
                if self.enable_stop_loss and current_price <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and current_price >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif current_price < prev_close - self.atr_multiplier * current_atr:
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.enable_stop_loss and current_price >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif current_price > prev_close + self.atr_multiplier * current_atr:
                    self.position.close()
                    self.position_type = None

        if not self.position:
            if current_price > prev_close + self.atr_multiplier * current_atr:
                self.buy()
                self.entry_price = current_price
                self.position_type = 'long'
            elif current_price < prev_close - self.atr_multiplier * current_atr and self.enable_shorting:
                self.sell()
                self.entry_price = current_price
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def atr_viz(data, atr_period=14, atr_multiplier=2):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    atr = calculate_atr(data['High'].values, data['Low'].values, data['Close'].values, atr_period)
    data['ATR'] = atr
    data['Upper'] = data['Close'].shift(1) + atr_multiplier * data['ATR']
    data['Lower'] = data['Close'].shift(1) - atr_multiplier * data['ATR']

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
    ax1.plot(data.index, data['Upper'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower'], label='Lower Band', color='green', linestyle='--')
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, data['ATR'], label='ATR', color='purple')
    ax2.set_xlabel('Time', fontproperties=font_properties, color='white')
    ax2.set_ylabel('ATR', fontproperties=font_properties, color='white')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')
    
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

    fig.suptitle('ATR Strategy Visualization', fontproperties=title_font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)



def run_atr(ticker, start_date, end_date, cash, commission, atr_period, atr_multiplier, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    ATRStrategy.atr_period = atr_period
    ATRStrategy.atr_multiplier = atr_multiplier
    ATRStrategy.stop_loss_pct = stop_loss_pct
    ATRStrategy.take_profit_pct = take_profit_pct
    ATRStrategy.enable_shorting = enable_shorting
    ATRStrategy.enable_stop_loss = enable_stop_loss
    ATRStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(ATRStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
