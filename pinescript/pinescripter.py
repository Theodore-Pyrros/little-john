import json
import os


def pinescripter(json_file_path, output_folder='strategy_export'):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        strategy_data = json.load(file)
    
    # Check if the strategy type is supported
    strategy_type = strategy_data['type'].lower()
    supported_strategies = ['rsi cross', 'bollinger bands', 'atr', 'adx', 'dpo', 'cci', 'macd', 'momentum', 'mean reversion', 'obv', 'sma cross', 'standard deviation', 'stochastic', 'vwap']
    if strategy_type not in supported_strategies:
        print(f"Unsupported strategy type: {strategy_data['type']}")
        return
    
    # Extract parameters
    params = strategy_data['parameters']
    
    # Generate PineScript based on strategy type
    strategy_functions = {
        'rsi cross': pinescripter_rsi,
        'bollinger bands': pinescripter_bollinger,
        'atr': pinescripter_atr,
        'adx': pinescripter_adx,
        'dpo': pinescripter_dpo,
        'cci': pinescripter_cci,
        'macd': pinescripter_macd,
        'momentum': pinescripter_momentum,
        'mean reversion': pinescripter_mean_reversion,
        'obv': pinescripter_obv,
        'sma cross': pinescripter_sma_cross,
        'standard deviation': pinescripter_standard_deviation,
        'stochastic': pinescripter_stochastic,
        'vwap': pinescripter_vwap
    }
    
    pinescript = strategy_functions[strategy_type](strategy_data, params)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write PineScript to file
    output_file_path = os.path.join(output_folder, f"{strategy_data['name']}_{strategy_type.replace(' ', '_')}.pine")
    with open(output_file_path, 'w') as file:
        file.write(pinescript)
    
    print(f"PineScript generated and saved to: {output_file_path}")



def pinescripter_rsi(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - RSI Cross Strategy", overlay=true)

// Input parameters
rsi_sma_short = input.int({params['rsi_sma_short']}, "RSI SMA Short", minval=1)
rsi_sma_long = input.int({params['rsi_sma_long']}, "RSI SMA Long", minval=1)
rsi_period = input.int({params['rsi_period']}, "RSI Period", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate RSI
up = ta.rma(math.max(ta.change(close), 0), rsi_period)
down = ta.rma(-math.min(ta.change(close), 0), rsi_period)
rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))

// Calculate SMAs of RSI
rsi_sma_short_line = ta.sma(rsi, rsi_sma_short)
rsi_sma_long_line = ta.sma(rsi, rsi_sma_long)

// Plotting
plot(rsi, color=color.purple, title="RSI")
plot(rsi_sma_short_line, color=color.orange, title="RSI SMA Short")
plot(rsi_sma_long_line, color=color.green, title="RSI SMA Long")

// Trading logic
long_condition = ta.crossover(rsi_sma_short_line, rsi_sma_long_line)
short_condition = ta.crossunder(rsi_sma_short_line, rsi_sma_long_line) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_bollinger(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - Bollinger Bands Strategy", overlay=true)

// Input parameters
bb_period = input.int({params['bb_period']}, "Bollinger Bands Period", minval=1)
bb_std_dev = input.float({params['bb_std_dev']}, "Standard Deviation", minval=0.1, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate Bollinger Bands
[middle, upper, lower] = ta.bb(close, bb_period, bb_std_dev)

// Plotting
plot(middle, color=color.blue, title="Middle Band")
plot(upper, color=color.red, title="Upper Band")
plot(lower, color=color.green, title="Lower Band")
fill(upper, lower, color=color.rgb(33, 150, 243, 95), title="Background")

// Trading logic
long_condition = close < lower
short_condition = close > upper and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_atr(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - ATR Strategy", overlay=true)

// Input parameters
atr_period = input.int({params['atr_period']}, "ATR Period", minval=1)
atr_multiplier = input.float({params['atr_multiplier']}, "ATR Multiplier", minval=0.1, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate ATR
atr = ta.atr(atr_period)
upper_band = ta.sma(close, atr_period) + atr * atr_multiplier
lower_band = ta.sma(close, atr_period) - atr * atr_multiplier

// Plotting
plot(upper_band, color=color.red, title="Upper ATR Band")
plot(lower_band, color=color.green, title="Lower ATR Band")

// Trading logic
long_condition = ta.crossover(close, lower_band)
short_condition = ta.crossunder(close, upper_band) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_adx(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - ADX Strategy", overlay=true)

// Input parameters
adx_period = input.int({params['adx_period']}, "ADX Period", minval=1)
adx_threshold = input.int({params['adx_threshold']}, "ADX Threshold", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate ADX
[diplus, diminus, adx] = ta.dmi(adx_period, adx_period)

// Plotting
plot(adx, color=color.blue, title="ADX")
hline(adx_threshold, color=color.gray, linestyle=hline.style_dashed, title="ADX Threshold")

// Trading logic
long_condition = adx > adx_threshold and diplus > diminus
short_condition = adx > adx_threshold and diminus > diplus and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_dpo(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - DPO Strategy", overlay=false)

// Input parameters
dpo_period = input.int({params['dpo_period']}, "DPO Period", minval=1)
dpo_threshold = input.float({params['dpo_threshold']}, "DPO Threshold", minval=0.0, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate DPO
dpo = close - ta.sma(close, dpo_period)[dpo_period / 2 + 1]

// Plotting
plot(dpo, color=color.blue, title="DPO")
hline(dpo_threshold, color=color.green, linestyle=hline.style_dashed, title="Upper Threshold")
hline(-dpo_threshold, color=color.red, linestyle=hline.style_dashed, title="Lower Threshold")

// Trading logic
long_condition = ta.crossover(dpo, dpo_threshold)
short_condition = ta.crossunder(dpo, -dpo_threshold) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_cci(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - CCI Strategy", overlay=false)

// Input parameters
cci_period = input.int({params['cci_period']}, "CCI Period", minval=1)
cci_overbought = input.int({params['cci_overbought']}, "CCI Overbought", minval=1)
cci_oversold = input.int({params['cci_oversold']}, "CCI Oversold", maxval=-1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate CCI
cci = ta.cci(close, cci_period)

// Plotting
plot(cci, color=color.blue, title="CCI")
hline(cci_overbought, color=color.red, linestyle=hline.style_dashed, title="Overbought")
hline(cci_oversold, color=color.green, linestyle=hline.style_dashed, title="Oversold")

// Trading logic
long_condition = ta.crossover(cci, cci_oversold)
short_condition = ta.crossunder(cci, cci_overbought) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_macd(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - MACD Strategy", overlay=false)

// Input parameters
fast_length = input.int({params['macd_fast']}, "Fast Length", minval=1)
slow_length = input.int({params['macd_slow']}, "Slow Length", minval=1)
signal_length = input.int({params['macd_signal']}, "Signal Smoothing", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate MACD
[macd_line, signal_line, hist] = ta.macd(close, fast_length, slow_length, signal_length)

// Plotting
plot(macd_line, color=color.blue, title="MACD")
plot(signal_line, color=color.orange, title="Signal")
plot(hist, color=color.red, title="Histogram", style=plot.style_histogram)

// Trading logic
long_condition = ta.crossover(macd_line, signal_line)
short_condition = ta.crossunder(macd_line, signal_line) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_momentum(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - Momentum Strategy", overlay=false)

// Input parameters
mom_period = input.int({params['mom_period']}, "Momentum Period", minval=1)
mom_threshold = input.float({params['mom_threshold']}, "Momentum Threshold", minval=0.0, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate Momentum
momentum = close - close[mom_period]

// Plotting
plot(momentum, color=color.blue, title="Momentum")
hline(mom_threshold, color=color.green, linestyle=hline.style_dashed, title="Upper Threshold")
hline(-mom_threshold, color=color.red, linestyle=hline.style_dashed, title="Lower Threshold")

// Trading logic
long_condition = ta.crossover(momentum, mom_threshold)
short_condition = ta.crossunder(momentum, -mom_threshold) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_mean_reversion(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - Mean Reversion Strategy", overlay=true)

// Input parameters
mr_period = input.int({params['mr_period']}, "Mean Reversion Period", minval=1)
mr_entry_std = input.float({params['mr_entry_std']}, "Entry Standard Deviation", minval=0.1, step=0.1)
mr_exit_std = input.float({params['mr_exit_std']}, "Exit Standard Deviation", minval=0.1, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate Mean and Standard Deviation
sma = ta.sma(close, mr_period)
std_dev = ta.stdev(close, mr_period)

upper_band = sma + mr_entry_std * std_dev
lower_band = sma - mr_entry_std * std_dev
exit_upper = sma + mr_exit_std * std_dev
exit_lower = sma - mr_exit_std * std_dev

// Plotting
plot(sma, color=color.blue, title="SMA")
plot(upper_band, color=color.red, title="Upper Entry Band")
plot(lower_band, color=color.green, title="Lower Entry Band")
plot(exit_upper, color=color.orange, title="Upper Exit Band")
plot(exit_lower, color=color.purple, title="Lower Exit Band")

// Trading logic
long_condition = close < lower_band
short_condition = close > upper_band and enable_shorting
exit_long = close > exit_upper
exit_short = close < exit_lower

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)
if (short_condition)
    strategy.entry("Short", strategy.short)

// Exit trades
if (exit_long)
    strategy.close("Long")
if (exit_short)
    strategy.close("Short")

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_obv(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - OBV Strategy", overlay=false)

// Input parameters
obv_periods = input.int({params['obv_periods']}, "OBV SMA Periods", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate OBV and its SMA
obv = ta.obv
obv_sma = ta.sma(obv, obv_periods)

// Plotting
plot(obv, color=color.blue, title="OBV")
plot(obv_sma, color=color.red, title="OBV SMA")

// Trading logic
long_condition = ta.crossover(obv, obv_sma)
short_condition = ta.crossunder(obv, obv_sma) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_sma_cross(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - SMA Cross Strategy", overlay=true)

// Input parameters
sma_short = input.int({params['sma_short']}, "Short SMA Period", minval=1)
sma_long = input.int({params['sma_long']}, "Long SMA Period", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate SMAs
sma_short_line = ta.sma(close, sma_short)
sma_long_line = ta.sma(close, sma_long)

// Plotting
plot(sma_short_line, color=color.blue, title="Short SMA")
plot(sma_long_line, color=color.red, title="Long SMA")

// Trading logic
long_condition = ta.crossover(sma_short_line, sma_long_line)
short_condition = ta.crossunder(sma_short_line, sma_long_line) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_standard_deviation(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - Standard Deviation Strategy", overlay=true)

// Input parameters
std_period = input.int({params['std_period']}, "Standard Deviation Period", minval=1)
std_multiplier = input.float({params['std_multiplier']}, "Standard Deviation Multiplier", minval=0.1, step=0.1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate Standard Deviation
sma = ta.sma(close, std_period)
std_dev = ta.stdev(close, std_period)
upper_band = sma + std_multiplier * std_dev
lower_band = sma - std_multiplier * std_dev

// Plotting
plot(sma, color=color.blue, title="SMA")
plot(upper_band, color=color.red, title="Upper Band")
plot(lower_band, color=color.green, title="Lower Band")

// Trading logic
long_condition = close < lower_band
short_condition = close > upper_band and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_stochastic(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - Stochastic Strategy", overlay=false)

// Input parameters
stoch_k = input.int({params['stoch_k']}, "%K Length", minval=1)
stoch_d = input.int({params['stoch_d']}, "%D Smoothing", minval=1)
stoch_overbought = input.int({params['stoch_overbought']}, "Overbought Level", minval=50, maxval=100)
stoch_oversold = input.int({params['stoch_oversold']}, "Oversold Level", minval=0, maxval=50)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate Stochastic
[k, d] = ta.stoch(close, high, low, stoch_k)
k_line = k
d_line = ta.sma(k, stoch_d)

// Plotting
plot(k_line, color=color.blue, title="%K")
plot(d_line, color=color.red, title="%D")
hline(stoch_overbought, color=color.red, linestyle=hline.style_dashed)
hline(stoch_oversold, color=color.green, linestyle=hline.style_dashed)

// Trading logic
long_condition = ta.crossover(k_line, d_line) and k_line < stoch_oversold
short_condition = ta.crossunder(k_line, d_line) and k_line > stoch_overbought and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.bottom, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.top, color=color.red, style=shape.triangledown, size=size.small)
"""

def pinescripter_vwap(strategy_data, params):
    return f"""
//@version=5
strategy("{strategy_data['name']} - VWAP Strategy", overlay=true)

// Input parameters
vwap_periods = input.int({params['vwap_periods']}, "VWAP Periods", minval=1)
stop_loss_pct = input.float({params['stop_loss_pct']}, "Stop Loss %", minval=0.0, step=0.1) / 100
take_profit_pct = input.float({params['take_profit_pct']}, "Take Profit %", minval=0.0, step=0.1) / 100
enable_shorting = input.bool({str(params['enable_shorting']).lower()}, "Enable Shorting")
enable_stop_loss = input.bool({str(params['enable_stop_loss']).lower()}, "Enable Stop Loss")
enable_take_profit = input.bool({str(params['enable_take_profit']).lower()}, "Enable Take Profit")

// Calculate VWAP
vwap = ta.vwap(hlc3, vwap_periods)

// Plotting
plot(vwap, color=color.blue, title="VWAP")

// Trading logic
long_condition = ta.crossover(close, vwap)
short_condition = ta.crossunder(close, vwap) and enable_shorting

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop Loss and Take Profit
long_stop_price = strategy.position_avg_price * (1 - stop_loss_pct)
long_profit_price = strategy.position_avg_price * (1 + take_profit_pct)
short_stop_price = strategy.position_avg_price * (1 + stop_loss_pct)
short_profit_price = strategy.position_avg_price * (1 - take_profit_pct)

if (strategy.position_size > 0)
    if (enable_stop_loss)
        strategy.exit("Long Stop", "Long", stop=long_stop_price)
    if (enable_take_profit)
        strategy.exit("Long Profit", "Long", limit=long_profit_price)

if (strategy.position_size < 0)
    if (enable_stop_loss)
        strategy.exit("Short Stop", "Short", stop=short_stop_price)
    if (enable_take_profit)
        strategy.exit("Short Profit", "Short", limit=short_profit_price)

// Plotting signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_condition, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)
"""