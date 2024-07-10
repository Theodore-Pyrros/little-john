import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

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

def obv_viz(data, obv_periods=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    obv = calculate_obv(data['Close'], data['Volume'])
    obv_sma = pd.Series(obv).rolling(window=obv_periods).mean()
    
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('OBV Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, obv, label='OBV', color='orange')
    ax2.plot(data.index, obv_sma, label=f'OBV SMA({obv_periods})', color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('OBV')
    ax2.legend()
    ax2.grid(True)
    
    plt.xticks([data[data['Date'] == date].index[0] for date in daily_indices],
               [date.strftime('%Y-%m-%d') for date in daily_indices],
               rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)

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