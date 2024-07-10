import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

def calculate_cci(high, low, close, cci_period=20):
    tp = (high + low + close) / 3
    sma = np.zeros_like(tp)
    mad = np.zeros_like(tp)
    cci = np.zeros_like(tp)
    
    for i in range(cci_period, len(tp)):
        sma[i] = np.mean(tp[i-cci_period+1:i+1])
        mad[i] = np.mean(np.abs(tp[i-cci_period+1:i+1] - sma[i]))
        cci[i] = (tp[i] - sma[i]) / (0.015 * mad[i])
    
    return cci

class CCIStrategy(Strategy):
    cci_period = 20
    cci_overbought = 100
    cci_oversold = -100
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.cci = self.I(calculate_cci, self.data.High, self.data.Low, self.data.Close, self.cci_period)
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
            if self.cci[-1] < self.cci_oversold:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.cci[-1] > self.cci_overbought and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def cci_viz(data, cci_period=20, cci_overbought=100, cci_oversold=-100):
    data = data[data['Volume'] > 0]    
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    cci = calculate_cci(data['High'], data['Low'], data['Close'], cci_period)

    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('CCI Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, cci, label='CCI', color='orange')
    ax2.axhline(y=cci_overbought, color='red', linestyle='--', label=f'Overbought ({cci_overbought})')
    ax2.axhline(y=cci_oversold, color='green', linestyle='--', label=f'Oversold ({cci_oversold})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('CCI')
    ax2.legend()
    ax2.grid(True)
    
    plt.xticks([data[data['Date'] == date].index[0] for date in daily_indices],
               [date.strftime('%Y-%m-%d') for date in daily_indices],
               rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)

def run_cci(ticker, start_date, end_date, cash, commission, cci_period, cci_overbought, cci_oversold, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    CCIStrategy.cci_period = cci_period
    CCIStrategy.cci_overbought = cci_overbought
    CCIStrategy.cci_oversold = cci_oversold
    CCIStrategy.stop_loss_pct = stop_loss_pct
    CCIStrategy.take_profit_pct = take_profit_pct
    CCIStrategy.enable_shorting = enable_shorting
    CCIStrategy.enable_stop_loss = enable_stop_loss
    CCIStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(CCIStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None