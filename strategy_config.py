import streamlit as st
from strategies.sma_cross import sma_cross_viz
from strategies.rsi_cross import rsi_cross_viz
from strategies.bollinger_bands import bollinger_bands_viz
from strategies.macd import macd_viz
from strategies.vwap import vwap_viz
from strategies.stochastic import stoch_viz
from strategies.mean_reversion import mean_reversion_viz
from strategies.momentum import momentum_viz
from strategies.adx import adx_viz
from strategies.cci import cci_viz
from strategies.dpo import dpo_viz
from strategies.obv import obv_viz
from strategies.atr import atr_viz
from strategies.standard_deviation import std_dev_viz

def get_strategy_params(strategy):
    if strategy == 'SMA Cross':
        with st.expander('SMA Cross Parameters', expanded=True):
            sma_short = st.slider('Short SMA', min_value=5, max_value=50, value=10, key='sma_short')
            sma_long = st.slider('Long SMA', min_value=10, max_value=100, value=20, key='sma_long')
        return {'sma_short': sma_short, 'sma_long': sma_long}

    elif strategy == 'RSI Cross':
        with st.expander('RSI Cross Parameters', expanded=True):
            rsi_period = st.slider('RSI Period', min_value=2, max_value=30, value=14)
            rsi_sma_short = st.slider('RSI Short SMA', min_value=5, max_value=50, value=10, key='rsi_short')
            rsi_sma_long = st.slider('RSI Long SMA', min_value=10, max_value=100, value=20, key='rsi_long')
        return {'rsi_period': rsi_period, 'rsi_sma_short': rsi_sma_short, 'rsi_sma_long': rsi_sma_long}

    elif strategy == 'Bollinger Bands':
        with st.expander('Bollinger Bands Parameters', expanded=True):
            bb_period = st.slider('MA Period', min_value=5, max_value=50, value=20, key='bb_period')
            bb_std_dev = st.slider('Std Dev Multiplier', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='bb_std_dev')
        return {'bb_period': bb_period, 'bb_std_dev': bb_std_dev}

    elif strategy == 'MACD':
        with st.expander('MACD Parameters', expanded=True):
            macd_fast = st.slider('Fast Period', min_value=5, max_value=50, value=12, key='macd_fast')
            macd_slow = st.slider('Slow Period', min_value=10, max_value=100, value=26, key='macd_slow')
            macd_signal = st.slider('Signal Period', min_value=5, max_value=50, value=9, key='macd_signal')
        return {'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': macd_signal}

    elif strategy == 'VWAP':
        with st.expander('VWAP Parameters', expanded=True):
            vwap_periods = st.slider('VWAP Periods', min_value=5, max_value=50, value=20, key='vwap_periods')
        return {'vwap_periods': vwap_periods}

    elif strategy == 'Stochastic':
        with st.expander('Stochastic Parameters', expanded=True):
            stoch_k = st.slider('K Period', min_value=5, max_value=50, value=14, key='stoch_k')
            stoch_d = st.slider('D Period', min_value=1, max_value=10, value=3, key='stoch_d')
            stoch_overbought = st.slider('Overbought Level', min_value=50, max_value=95, value=80, key='stoch_overbought')
            stoch_oversold = st.slider('Oversold Level', min_value=5, max_value=50, value=20, key='stoch_oversold')
        return {'stoch_k': stoch_k, 'stoch_d': stoch_d, 'stoch_overbought': stoch_overbought, 'stoch_oversold': stoch_oversold}

    elif strategy == 'Mean Reversion':
        with st.expander('Mean Reversion Parameters', expanded=True):
            mr_period = st.slider('Lookback Period', min_value=5, max_value=50, value=20, key='mr_period')
            mr_entry_std = st.slider('Entry Std Dev', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='mr_entry_std')
            mr_exit_std = st.slider('Exit Std Dev', min_value=0.1, max_value=2.0, value=0.5, step=0.1, key='mr_exit_std')
        return {'mr_period': mr_period, 'mr_entry_std': mr_entry_std, 'mr_exit_std': mr_exit_std}

    elif strategy == 'Momentum':
        with st.expander('Momentum Parameters', expanded=True):
            mom_period = st.slider('ROC Period', min_value=5, max_value=50, value=14, key='mom_period')
            mom_threshold = st.slider('ROC Threshold', min_value=0.0, max_value=5.0, value=2.0, step=0.1, key='mom_threshold')
        return {'mom_period': mom_period, 'mom_threshold': mom_threshold}

    elif strategy == 'ADX':
        with st.expander('ADX Parameters', expanded=True):
            adx_period = st.slider('ADX Period', min_value=5, max_value=50, value=14, key='adx_period')
            adx_threshold = st.slider('ADX Threshold', min_value=10, max_value=50, value=25, key='adx_threshold')
        return {'adx_period': adx_period, 'adx_threshold': adx_threshold}

    elif strategy == 'CCI':
        with st.expander('CCI Parameters', expanded=True):
            cci_period = st.slider('CCI Period', min_value=5, max_value=50, value=20, key='cci_period')
            cci_overbought = st.slider('Overbought Level', min_value=50, max_value=200, value=100, key='cci_overbought')
            cci_oversold = st.slider('Oversold Level', min_value=-200, max_value=-50, value=-100, key='cci_oversold')
        return {'cci_period': cci_period, 'cci_overbought': cci_overbought, 'cci_oversold': cci_oversold}

    elif strategy == 'DPO':
        with st.expander('DPO Parameters', expanded=True):
            dpo_period = st.slider('DPO Period', min_value=5, max_value=50, value=20, key='dpo_period')
            dpo_threshold = st.slider('DPO Threshold', min_value=0.0, max_value=5.0, value=0.5, step=0.1, key='dpo_threshold')
        return {'dpo_period': dpo_period, 'dpo_threshold': dpo_threshold}

    elif strategy == 'OBV':
        with st.expander('OBV Parameters', expanded=True):
            obv_periods = st.slider('OBV SMA Periods', min_value=5, max_value=50, value=20, key='obv_periods')
        return {'obv_periods': obv_periods}

    elif strategy == 'ATR':
        with st.expander('ATR Parameters', expanded=True):
            atr_period = st.slider('ATR Period', min_value=5, max_value=50, value=14, key='atr_period')
            atr_multiplier = st.slider('ATR Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='atr_multiplier')
        return {'atr_period': atr_period, 'atr_multiplier': atr_multiplier}

    elif strategy == 'Standard Deviation':
        with st.expander('Standard Deviation Parameters', expanded=True):
            std_period = st.slider('Period', min_value=5, max_value=50, value=20, key='std_period')
            std_multiplier = st.slider('Std Dev Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='std_multiplier')
        return {'std_period': std_period, 'std_multiplier': std_multiplier}

    else:
        return {}

def visualize_strategy(strategy, data, params):
    if strategy == 'SMA Cross':
        sma_cross_viz(data, params['sma_short'], params['sma_long'])
    elif strategy == 'RSI Cross':
        rsi_cross_viz(data, params['rsi_sma_short'], params['rsi_sma_long'], params['rsi_period'])
    elif strategy == 'Bollinger Bands':
        bollinger_bands_viz(data, params['bb_period'], params['bb_std_dev'])
    elif strategy == 'MACD':
        macd_viz(data, params['macd_fast'], params['macd_slow'], params['macd_signal'])
    elif strategy == 'VWAP':
        vwap_viz(data, params['vwap_periods'])
    elif strategy == 'Stochastic':
        stoch_viz(data, params['stoch_k'], params['stoch_d'], params['stoch_overbought'], params['stoch_oversold'])
    elif strategy == 'Mean Reversion':
        mean_reversion_viz(data, params['mr_period'], params['mr_entry_std'], params['mr_exit_std'])
    elif strategy == 'Momentum':
        momentum_viz(data, params['mom_period'], params['mom_threshold'])
    elif strategy == 'ADX':
        adx_viz(data, params['adx_period'], params['adx_threshold'])
    elif strategy == 'CCI':
        cci_viz(data, params['cci_period'], params['cci_overbought'], params['cci_oversold'])
    elif strategy == 'DPO':
        dpo_viz(data, params['dpo_period'], params['dpo_threshold'])
    elif strategy == 'OBV':
        obv_viz(data, params['obv_periods'])
    elif strategy == 'ATR':
        atr_viz(data, params['atr_period'], params['atr_multiplier'])
    elif strategy == 'Standard Deviation':
        std_dev_viz(data, params['std_period'], params['std_multiplier'])