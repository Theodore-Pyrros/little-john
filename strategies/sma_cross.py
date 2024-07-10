import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import matplotlib.font_manager as fm

class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
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
            if crossover(self.sma1, self.sma2):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.sma2, self.sma1) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


# Define font properties
font_path = "Times New Roman.ttf"
font_properties = fm.FontProperties(fname=font_path, size=14)
title_font_properties = fm.FontProperties(fname=font_path, size=16, weight='bold')

def sma_cross_viz(data, n1=10, n2=20, trades=None):
    data = data[data['Volume'] > 0].copy()    
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    short_sma = np.convolve(data['Close'], np.ones(n1)/n1, mode='valid')
    long_sma = np.convolve(data['Close'], np.ones(n2)/n2, mode='valid')

    start_idx = max(n1, n2) - 1

    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='none')

    # Set transparent background
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # Remove the outline of the axes
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(data.index[start_idx:], data['Close'][start_idx:], label='Price', color='#00A86B')  # Flashier pine green
    ax.plot(data.index[start_idx:], short_sma[start_idx - n1 + 1:], label=f'SMA({n1})', color='orange')
    ax.plot(data.index[start_idx:], long_sma[start_idx - n2 + 1:], label=f'SMA({n2})', color='green')

    # Plot trade markers if trades data is provided
    if trades is not None and isinstance(trades, pd.DataFrame):
        for _, trade in trades.iterrows():
            if 'EntryBar' in trade and 'ExitBar' in trade:
                entry_bar = trade['EntryBar']
                exit_bar = trade['ExitBar']
                if entry_bar >= start_idx and exit_bar < len(data):
                    if trade['Size'] > 0:  # Long trade
                        ax.plot(entry_bar, data['Close'][entry_bar], '^', color='g', markersize=10)
                        ax.plot(exit_bar, data['Close'][exit_bar], 'v', color='r', markersize=10)
                    else:  # Short trade
                        ax.plot(entry_bar, data['Close'][entry_bar], 'v', color='r', markersize=10)
                        ax.plot(exit_bar, data['Close'][exit_bar], '^', color='g', markersize=10)

    ax.set_ylabel('Price', fontproperties=font_properties, color='white')
    ax.legend(prop=font_properties, facecolor='white', framealpha=0.5)
    ax.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax.grid(False, axis='x')

    ax.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30, fontproperties=font_properties, color='white')
    
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties)

    fig.suptitle('SMA Cross Visualization with Trades', fontproperties=title_font_properties, color='white')
    ax.set_xlabel('Time', fontproperties=font_properties, color='white')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)




def run_sma_cross(ticker, start_date, end_date, cash, commission, n1, n2, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    SmaCross.n1 = n1
    SmaCross.n2 = n2
    SmaCross.stop_loss_pct = stop_loss_pct
    SmaCross.take_profit_pct = take_profit_pct
    SmaCross.enable_shorting = enable_shorting
    SmaCross.enable_stop_loss = enable_stop_loss
    SmaCross.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(SmaCross, data, cash, commission)
        
        # Visualize the strategy with trades
        # st.subheader('SMA Cross Visualization with Trades')
        # if '_trades' in output and not output['_trades'].empty:
            # sma_cross_viz(data, n1, n2, trades=output['_trades'])
        # else:
            # st.warning("No trades were executed in this backtest.")
            # sma_cross_viz(data, n1, n2)  # Still show the visualization even if no trades
        
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
