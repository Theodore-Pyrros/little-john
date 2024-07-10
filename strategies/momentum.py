import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

class MomentumStrategy(Strategy):
    mom_period = 14
    mom_threshold = 2.0
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = self.data.Close
        self.roc = self.I(self.calculate_roc, close, self.mom_period)
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def calculate_roc(self, close, n):
        roc = np.zeros_like(close)
        roc[n:] = (close[n:] / close[:-n] - 1) * 100
        return roc

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
            if self.roc[-1] > self.mom_threshold:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.roc[-1] < -self.mom_threshold and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def momentum_viz(data, mom_period=14, mom_threshold=0):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    roc = (data['Close'] / data['Close'].shift(mom_period) - 1) * 100
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor='none')

    # Set transparent background
    fig.patch.set_alpha(0)
    ax1.set_facecolor('none')

    # Remove the outline of the axes
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    ax1.plot(data.index, data['Close'], label='Price', color='#00A86B')  # Flashier pine green
    ax1.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax1.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2 = ax1.twinx()
    ax2.plot(data.index[mom_period:], roc[mom_period:], label=f'ROC({mom_period})', color='orange')
    ax2.axhline(y=mom_threshold, color='r', linestyle='--', label='mom_threshold')
    ax2.axhline(y=-mom_threshold, color='r', linestyle='--')
    ax2.set_ylabel('Rate of Change (%)', fontproperties=font_properties, color='orange')
    ax2.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop=font_properties)

    ax1.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30, fontproperties=font_properties, color='white')
    
    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='orange', labelsize=12)
    
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('Momentum Strategy Visualization', fontproperties=title_font_properties, color='white')
    ax1.set_xlabel('Time', fontproperties=font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)





def run_momentum(ticker, start_date, end_date, cash, commission, mom_period, mom_threshold, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    MomentumStrategy.mom_period = mom_period
    MomentumStrategy.mom_threshold = mom_threshold
    MomentumStrategy.stop_loss_pct = stop_loss_pct
    MomentumStrategy.take_profit_pct = take_profit_pct
    MomentumStrategy.enable_shorting = enable_shorting
    MomentumStrategy.enable_stop_loss = enable_stop_loss
    MomentumStrategy.enable_take_profit = enable_take_profit
    
    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(MomentumStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
