import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

class MeanReversion(Strategy):
    mr_period = 20  # lookback period for calculating mean
    mr_entry_std = 2.0  # number of standard deviations for entry
    mr_exit_std = 0.5  # number of standard deviations for exit
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.sma = self.I(SMA, self.data.Close, self.mr_period)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.mr_period).std(), self.data.Close)
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
                elif self.data.Close[-1] >= self.sma[-1] + self.exit_std * self.std[-1]:
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.enable_stop_loss and self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] <= self.sma[-1] - self.exit_std * self.std[-1]:
                    self.position.close()
                    self.position_type = None
        if not self.position:
            if self.data.Close[-1] <= self.sma[-1] - self.entry_std * self.std[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.data.Close[-1] >= self.sma[-1] + self.entry_std * self.std[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def mean_reversion_viz(data, mr_period=20,mr_entry_std=2.0,mr_exit_std=0.5):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    data['SMA'] = data['Close'].rolling(window=mr_period).mean()
    data['STD'] = data['Close'].rolling(window=mr_period).std()
    data['Upper_Entry'] = data['SMA'] +mr_entry_std * data['STD']
    data['Lower_Entry'] = data['SMA'] -mr_entry_std * data['STD']
    data['Upper_Exit'] = data['SMA'] +mr_exit_std * data['STD']
    data['Lower_Exit'] = data['SMA'] -mr_exit_std * data['STD']
    
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Price', color='blue')
    ax.plot(data.index, data['SMA'], label=f'SMA({n})', color='orange')
    ax.plot(data.index, data['Upper_Entry'], label='Upper Entry', color='red', linestyle='--')
    ax.plot(data.index, data['Lower_Entry'], label='Lower Entry', color='green', linestyle='--')
    ax.plot(data.index, data['Upper_Exit'], label='Upper Exit', color='purple', linestyle=':')
    ax.plot(data.index, data['Lower_Exit'], label='Lower Exit', color='brown', linestyle=':')
    
    ax.set_title('Mean Reversion Strategy Visualization')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30)
    
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def run_mean_reversion(ticker, start_date, end_date, cash, commission, mr_period,mr_entry_std,mr_exit_std, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    MeanReversion.mr_period = mr_period
    MeanReversion.entry_std =mr_entry_std
    MeanReversion.exit_std =mr_exit_std
    MeanReversion.stop_loss_pct = stop_loss_pct
    MeanReversion.take_profit_pct = take_profit_pct
    MeanReversion.enable_shorting = enable_shorting
    MeanReversion.enable_stop_loss = enable_stop_loss
    MeanReversion.enable_take_profit = enable_take_profit
    
    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(MeanReversion, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None