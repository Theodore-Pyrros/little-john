import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
import matplotlib.dates as mdates
import streamlit as st


def display_metrics(output):
    metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
               'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
               'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
               'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
               'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
               'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]', 'Number of Trades']
    
    result = {k: output[k] for k in metrics if k in output}
    
    # Add Number of Trades explicitly if it's not already in the output
    if 'Number of Trades' not in result and 'Trades' in result:
        result['Number of Trades'] = result['Trades']
    
    return result


def rsi_cross_viz(data, rsi_sma_short=10, rsi_sma_long=20, rsi_period=14):
    plt.rcParams['font.family'] = 'Times New Roman'

    data = data[data['Volume'] > 0]
    data.reset_index(inplace=True)

    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index

    # Calculate RSI
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Fill NaN values with 50 (neutral)
    short_rsi = rsi.rolling(window=rsi_sma_short, min_periods=1).mean()
    long_rsi = rsi.rolling(window=rsi_sma_long, min_periods=1).mean()

    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, facecolor='none')

    # Set transparent background
    fig.patch.set_alpha(0)
    ax1.set_facecolor('none')
    ax2.set_facecolor('none')

    # Remove the outline of the axes
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_ylabel('Price', color='white')
    ax1.legend()
    ax1.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax1.grid(False, axis='x')

    ax2.plot(data.index, rsi, label='RSI', color='purple')
    ax2.plot(data.index, short_rsi, label=f'RSI SMA({rsi_sma_short})', color='orange')
    ax2.plot(data.index, long_rsi, label=f'RSI SMA({rsi_sma_long})', color='green')
    ax2.set_ylabel('RSI', color='white')
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax2.grid(False, axis='x')

    plt.title('RSI Cross Visualization', color='white')
    plt.xlabel('Time', color='white')

    ax1.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30, color='white')

    # Change tick colors to white
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    plt.tight_layout()

    st.pyplot(fig)



def run_backtest(strategy, data, cash=10000, commission=0.1):
    commission_decimal = commission / 100  # Convert percentage to decimal
    bt = Backtest(data, strategy, cash=cash, commission=commission_decimal)
    output = bt.run()
    return output





def display_performance_metrics(output):
    """
    Displays key performance metrics.
    """
    st.subheader('Performance Metrics')
    
    key_metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
                   'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
                   'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
                   'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
                   'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
                   'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]']
    
    metrics = output.drop(['_strategy', '_equity_curve', '_trades'])
    selected_metrics = {k: metrics[k] for k in key_metrics if k in metrics}
    df_metrics = pd.DataFrame(selected_metrics, index=['Value']).T
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{df_metrics.loc['Return [%]', 'Value']:.2f}%")
    col2.metric("Sharpe Ratio", f"{df_metrics.loc['Sharpe Ratio', 'Value']:.2f}")
    col3.metric("Max Drawdown", f"{df_metrics.loc['Max. Drawdown [%]', 'Value']:.2f}%")
    
    strategy_return = df_metrics.loc['Return [%]', 'Value']
    bh_return = df_metrics.loc['Buy & Hold Return [%]', 'Value']
    outperformance = strategy_return - bh_return
    st.metric("Strategy vs. Buy & Hold", f"{outperformance:.2f}%", 
              delta=f"{outperformance:.2f}%", delta_color="normal")
    
    st.metric("Win Rate", f"{df_metrics.loc['Win Rate [%]', 'Value']:.2f}%")
    
    with st.expander("View All Metrics"):
        st.dataframe(df_metrics, use_container_width=True)

def plot_return_comparison(strategy_return, bh_return):
    """
    Plots a bar chart comparing strategy return to buy & hold return.
    """
    fig_return_comparison = go.Figure(data=[
        go.Bar(name='Strategy', x=['Return'], y=[strategy_return]),
        go.Bar(name='Buy & Hold', x=['Return'], y=[bh_return])
    ])
    fig_return_comparison.update_layout(title='Strategy vs. Buy & Hold Return Comparison')
    st.plotly_chart(fig_return_comparison, use_container_width=True)

def display_metrics_with_title(title, output, detail_level='simple'):
    st.header(title)
    if detail_level == 'simple':
        # Display a simple subset of metrics
        simple_metrics = ['Return [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Max. Drawdown [%]']
        result = {k: output[k] for k in simple_metrics if k in output}
        st.write(result)
    else:
        # Display all metrics as originally done in display_metrics
        display_metrics(output)
