import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

class BollingerBandsStrategy(Strategy):
    bb_period = 20  # Period for moving average
    bb_std_dev = 2   # Number of standard deviations for the bands
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = self.data.Close
        self.sma = self.I(SMA, close, self.bb_period)
        self.std = self.I(self.rolling_std, close, self.bb_period)
        self.upper = self.I(lambda: self.sma + self.bb_std_dev * self.std)
        self.lower = self.I(lambda: self.sma - self.bb_std_dev * self.std)
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def rolling_std(self, array, bb_period):
        result = np.full_like(array, np.nan)
        for i in range(bb_period-1, len(array)):
            result[i] = np.std(array[i-bb_period+1:i+1])
        return result

    def next(self):
        if np.isnan(self.upper[-1]) or np.isnan(self.lower[-1]):
            return

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
            if self.data.Close[-1] < self.lower[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.data.Close[-1] > self.upper[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def bollinger_bands_viz(data, bb_period=20, bb_std_dev=2):
    data = data[data['Volume'] > 0]
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index

    close = data['Close']
    ma = close.rolling(window=bb_period, min_periods=1).mean()
    std = close.rolling(window=bb_period, min_periods=1).std()
    upper_band = ma + (std * bb_std_dev)
    lower_band = ma - (std * bb_std_dev)

    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index

    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(data.index, data['Close'], label='Price', color='blue')
    ax.plot(data.index, upper_band, label='Upper Bollinger Band', linestyle='--', color='red')
    ax.plot(data.index, lower_band, label='Lower Bollinger Band', linestyle='--', color='green')
    ax.fill_between(data.index, lower_band, upper_band, color='gray', alpha=0.1)

    ax.set_title('Bollinger Bands Visualization')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30)
    
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

def run_bollinger_bands(ticker, start_date, end_date, cash, commission, bb_period, bb_std_dev, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    BollingerBandsStrategy.bb_period = bb_period
    BollingerBandsStrategy.bb_std_dev = bb_std_dev
    BollingerBandsStrategy.stop_loss_pct = stop_loss_pct
    BollingerBandsStrategy.take_profit_pct = take_profit_pct
    BollingerBandsStrategy.enable_shorting = enable_shorting
    BollingerBandsStrategy.enable_stop_loss = enable_stop_loss
    BollingerBandsStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(BollingerBandsStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None