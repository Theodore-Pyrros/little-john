import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

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

def momentum_viz(data, mom_period=14, mom_threshold=0):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    roc = (data['Close'] / data['Close'].shift(mom_period) - 1) * 100
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('Momentum Strategy Visualization')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(data.index[mom_period:], roc[mom_period:], label=f'ROC({mom_period})', color='orange')
    ax2.axhline(y=mom_threshold, color='r', linestyle='--', label='mom_threshold')
    ax2.axhline(y=-mom_threshold, color='r', linestyle='--')
    ax2.set_ylabel('Rate of Change (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.xticks(
        ticks=[data[data['Date'] == date].index[0] for date in daily_indices],
        labels=[date.strftime('%Y-%m-%d') for date in daily_indices],
        rotation=30
    )
    
    plt.tight_layout()
    st.pyplot(fig)

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