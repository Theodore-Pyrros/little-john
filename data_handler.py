import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])

    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[data['Volume'] > 0]
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    return data

def fetch_data_pv(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    data = data[data['Volume'] > 0]
    
    data = data.reset_index()
    data = data.rename(columns={'index': 'Datetime'})
    
    return data