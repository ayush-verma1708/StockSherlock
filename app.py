import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import ta
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# Configure page
st.set_page_config(
    page_title="Stock Data Visualizer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Real-Time Stock Analyzer")
st.markdown("Advanced stock analysis tool for both intraday and long-term trading with live signals")

# Sidebar configuration
st.sidebar.title("Trading Configuration")

# Trading mode selection
trading_mode = st.sidebar.radio(
    "Trading Mode:",
    ["Intraday Trading", "Long-term Investment"]
)

# Input for stock symbol
col1, col2 = st.columns([3, 1])
with col1:
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS):", "RELIANCE.NS").upper()

# Different time periods based on trading mode
with col2:
    if trading_mode == "Intraday Trading":
        period_options = ["1d", "5d", "1mo"]
        interval_options = ["1m", "2m", "5m", "15m", "30m", "1h"]
        period = st.selectbox("Time Period:", options=period_options, index=0)
        interval = st.selectbox("Interval:", options=interval_options, index=4)  # Default to 30m for intraday
    else:  # Long-term Investment
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        interval_options = ["1d", "5d", "1wk", "1mo", "3mo"]
        period = st.selectbox("Time Period:", options=period_options, index=2)  # Default to 6mo
        interval = st.selectbox("Interval:", options=interval_options, index=0)  # Default to 1d

# Function to get stock data
@st.cache_data(ttl=60)  # Cache data for just 1 minute to ensure real-time data, especially for intraday
def get_stock_data(symbol, period, interval):
    try:
        # Get stock information
        stock = yf.Ticker(symbol)
        
        # Check if the stock exists by trying to get info
        info = stock.info
        if not info or 'regularMarketPrice' not in info:
            return None, None
        
        # Get historical market data with the specified interval
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return None, None
        
        # Add technical indicators
        if len(hist) > 14:  # Ensure we have enough data points
            # Add MACD
            macd = MACD(close=hist['Close'])
            hist['macd'] = macd.macd()
            hist['macd_signal'] = macd.macd_signal()
            hist['macd_diff'] = macd.macd_diff()
            
            # Add RSI
            rsi = RSIIndicator(close=hist['Close'])
            hist['rsi'] = rsi.rsi()
            
            # Add moving averages
            hist['sma_20'] = SMAIndicator(close=hist['Close'], window=20).sma_indicator()
            hist['sma_50'] = SMAIndicator(close=hist['Close'], window=50).sma_indicator()
            hist['sma_200'] = SMAIndicator(close=hist['Close'], window=200).sma_indicator()
            
            hist['ema_12'] = EMAIndicator(close=hist['Close'], window=12).ema_indicator()
            hist['ema_26'] = EMAIndicator(close=hist['Close'], window=26).ema_indicator()
            
            # Add Bollinger Bands
            bollinger = BollingerBands(close=hist['Close'])
            hist['bollinger_mavg'] = bollinger.bollinger_mavg()
            hist['bollinger_high'] = bollinger.bollinger_hband()
            hist['bollinger_low'] = bollinger.bollinger_lband()
            
            # Add Stochastic Oscillator
            stoch = StochasticOscillator(high=hist['High'], low=hist['Low'], close=hist['Close'])
            hist['stoch_k'] = stoch.stoch()
            hist['stoch_d'] = stoch.stoch_signal()
            
            # Calculate trading signals
            # MACD crossover
            hist['macd_signal_line_crossover'] = np.where((hist['macd'] > hist['macd_signal']) & 
                                                         (hist['macd'].shift(1) <= hist['macd_signal'].shift(1)), 1, 0)
            hist['macd_signal_line_crossunder'] = np.where((hist['macd'] < hist['macd_signal']) & 
                                                          (hist['macd'].shift(1) >= hist['macd_signal'].shift(1)), 1, 0)
            
            # RSI signals
            hist['rsi_oversold'] = np.where(hist['rsi'] < 30, 1, 0)
            hist['rsi_overbought'] = np.where(hist['rsi'] > 70, 1, 0)
            
            # Moving average crossovers
            hist['ma_crossover'] = np.where((hist['sma_20'] > hist['sma_50']) & 
                                           (hist['sma_20'].shift(1) <= hist['sma_50'].shift(1)), 1, 0)
            hist['ma_crossunder'] = np.where((hist['sma_20'] < hist['sma_50']) & 
                                            (hist['sma_20'].shift(1) >= hist['sma_50'].shift(1)), 1, 0)
            
            # Bollinger Bands signals
            hist['price_above_upper_band'] = np.where(hist['Close'] > hist['bollinger_high'], 1, 0)
            hist['price_below_lower_band'] = np.where(hist['Close'] < hist['bollinger_low'], 1, 0)
        
        # We don't return the stock object directly to avoid pickling issues
        # Instead we'll return the necessary data components
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Main function to display stock data
def display_stock_data():
    if not stock_symbol:
        st.warning("Please enter a stock symbol.")
        return
    
    with st.spinner(f"Loading data for {stock_symbol}..."):
        info, hist = get_stock_data(stock_symbol, period, interval)
        
        if info is None or hist is None:
            st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
            return
        
        # Display basic stock info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(f"{info.get('shortName', stock_symbol)}")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            current_price = info.get('regularMarketPrice', 'N/A')
            previous_close = info.get('previousClose', 'N/A')
            if current_price != 'N/A' and previous_close != 'N/A':
                price_change = current_price - previous_close
                price_change_percent = (price_change / previous_close) * 100
                color = "green" if price_change >= 0 else "red"
                change_symbol = "+" if price_change >= 0 else ""
                st.markdown(f"**Current Price:** ₹{current_price:.2f}")
                st.markdown(f"**Change:** <span style='color:{color}'>{change_symbol}{price_change:.2f} ({change_symbol}{price_change_percent:.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Current Price:** ₹{current_price}")
                st.markdown("**Change:** N/A")
            
        with col3:
            st.markdown(f"**Market Cap:** ₹{info.get('marketCap', 'N/A'):,}")
            st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**52-Week Range:** ₹{info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ₹{info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
        
        st.markdown("---")
        
        # Create interactive stock price chart
        st.subheader("Historical Stock Price")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add volume as a bar chart on a secondary y-axis
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker=dict(color='lightgray')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{stock_symbol} Stock Price and Volume",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode="x unified",
            height=500,
            yaxis2=dict(
                title="Volume",
                title_font=dict(color="gray"),
                tickfont=dict(color="gray"),
                anchor="x",
                overlaying="y",
                side="right"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key financial metrics in a table
        st.subheader("Key Financial Metrics")
        
        metrics = {
            "Market Cap": info.get('marketCap', 'N/A'),
            "Enterprise Value": info.get('enterpriseValue', 'N/A'),
            "Trailing P/E": info.get('trailingPE', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Price to Book": info.get('priceToBook', 'N/A'),
            "Price to Sales": info.get('priceToSalesTrailing12Months', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "50-Day MA": info.get('fiftyDayAverage', 'N/A'),
            "200-Day MA": info.get('twoHundredDayAverage', 'N/A'),
            "52-Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52-Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Average Volume": info.get('averageVolume', 'N/A'),
            "Beta": info.get('beta', 'N/A'),
            "Return on Assets": info.get('returnOnAssets', 'N/A'),
            "Return on Equity": info.get('returnOnEquity', 'N/A'),
            "Revenue": info.get('totalRevenue', 'N/A'),
            "Profit Margin": info.get('profitMargins', 'N/A'),
            "Earnings Growth": info.get('earningsGrowth', 'N/A'),
            "Revenue Growth": info.get('revenueGrowth', 'N/A')
        }
        
        # Format metrics for display
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key in ["Market Cap", "Enterprise Value", "Revenue", "Average Volume"]:
                if value != 'N/A':
                    if value >= 1e9:
                        formatted_metrics[key] = f"₹{value/1e9:.2f}B"
                    elif value >= 1e6:
                        formatted_metrics[key] = f"₹{value/1e6:.2f}M"
                    else:
                        formatted_metrics[key] = f"₹{value:,.2f}"
                else:
                    formatted_metrics[key] = "N/A"
            elif isinstance(value, float) and key in ["Dividend Yield", "Return on Assets", "Return on Equity", "Profit Margin", "Earnings Growth", "Revenue Growth"]:
                if value != 'N/A':
                    formatted_metrics[key] = f"{value*100:.2f}%"
                else:
                    formatted_metrics[key] = "N/A"
            elif isinstance(value, float):
                if value != 'N/A':
                    formatted_metrics[key] = f"{value:.2f}"
                else:
                    formatted_metrics[key] = "N/A"
            else:
                formatted_metrics[key] = value
        
        # Create DataFrame for the metrics
        metrics_df = pd.DataFrame(list(formatted_metrics.items()), columns=["Metric", "Value"])
        
        # Display metrics in two columns
        col1, col2 = st.columns(2)
        half = len(metrics_df) // 2
        
        with col1:
            st.table(metrics_df.iloc[:half].set_index("Metric"))
        
        with col2:
            st.table(metrics_df.iloc[half:].set_index("Metric"))
        
        # Create a dataframe for download
        download_df = hist.reset_index()
        
        # Add metrics to the download dataframe
        raw_metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        
        # Technical Analysis and Trading Signals Section
        st.markdown("---")
        st.subheader("Technical Analysis & Trading Signals")
        
        # Trading Risk Settings
        st.sidebar.header("Trading Parameters")
        risk_appetite = st.sidebar.slider("Risk Appetite (1-10)", 1, 10, 5)
        investment_amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)
        
        # Different parameters for intraday vs long-term
        if trading_mode == "Intraday Trading":
            stop_loss_percent = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.0, 0.1)
            take_profit_percent = st.sidebar.slider("Take Profit (%)", 0.5, 10.0, 2.0, 0.1)
            st.sidebar.markdown("### Intraday Time Constraints")
            market_close_time = st.sidebar.time_input("Market Closing Time", datetime.now().replace(hour=15, minute=30).time())
            exit_buffer_minutes = st.sidebar.slider("Exit Before Close (minutes)", 5, 60, 15)
            max_holding_days = 1  # Fixed for intraday
            
            # Calculate exit time
            current_date = datetime.now().date()
            market_close_datetime = datetime.combine(current_date, market_close_time)
            exit_time = market_close_datetime - timedelta(minutes=exit_buffer_minutes)
            exit_time_str = exit_time.strftime("%H:%M")
            
            st.sidebar.markdown(f"**Exit all positions by:** {exit_time_str}")
            st.sidebar.markdown("---")
            
            # Add intraday-specific settings
            st.sidebar.markdown("### Intraday Settings")
            momentum_sensitivity = st.sidebar.slider("Momentum Sensitivity", 1, 10, 5)
            volatility_threshold = st.sidebar.slider("Volatility Threshold (%)", 0.5, 5.0, 1.5, 0.1)
        else:
            stop_loss_percent = st.sidebar.slider("Stop Loss (%)", 1.0, 15.0, 5.0, 0.5)
            take_profit_percent = st.sidebar.slider("Take Profit (%)", 1.0, 30.0, 10.0, 0.5)
            max_holding_days = st.sidebar.slider("Max Holding Days", 1, 90, 30)
        
        # Calculate the current trading signals
        current_signals = {}
        trade_recommendation = "HOLD"
        signal_strength = 0
        reason = []
        
        # Initialize latest_close to avoid UnboundLocalError
        latest_close = hist['Close'].iloc[-1] if not hist.empty else 0.0
        
        if 'rsi' in hist.columns and not hist.empty and not hist['rsi'].isna().all():
            # Get the latest values (latest_close is already defined above)
            latest_rsi = hist['rsi'].iloc[-1]
            latest_macd = hist['macd'].iloc[-1] if 'macd' in hist.columns else None
            latest_macd_signal = hist['macd_signal'].iloc[-1] if 'macd_signal' in hist.columns else None
            latest_sma_20 = hist['sma_20'].iloc[-1] if 'sma_20' in hist.columns else None
            latest_sma_50 = hist['sma_50'].iloc[-1] if 'sma_50' in hist.columns else None
            latest_upper_band = hist['bollinger_high'].iloc[-1] if 'bollinger_high' in hist.columns else None
            latest_lower_band = hist['bollinger_low'].iloc[-1] if 'bollinger_low' in hist.columns else None
            latest_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # RSI Signal
            if latest_rsi < 30:
                current_signals['RSI'] = "BUY (Oversold)"
                signal_strength += 2
                reason.append("RSI indicates oversold conditions")
            elif latest_rsi > 70:
                current_signals['RSI'] = "SELL (Overbought)"
                signal_strength -= 2
                reason.append("RSI indicates overbought conditions")
            else:
                current_signals['RSI'] = "NEUTRAL"
            
            # MACD Signal
            if latest_macd is not None and latest_macd_signal is not None:
                if latest_macd > latest_macd_signal:
                    current_signals['MACD'] = "BUY"
                    signal_strength += 1
                    if latest_macd > 0:
                        signal_strength += 0.5
                        reason.append("MACD is above signal line and positive")
                    else:
                        reason.append("MACD is above signal line")
                else:
                    current_signals['MACD'] = "SELL"
                    signal_strength -= 1
                    if latest_macd < 0:
                        signal_strength -= 0.5
                        reason.append("MACD is below signal line and negative")
                    else:
                        reason.append("MACD is below signal line")
            
            # Moving Average Signal
            if latest_sma_20 is not None and latest_sma_50 is not None:
                if latest_sma_20 > latest_sma_50:
                    current_signals['Moving Average'] = "BUY (Uptrend)"
                    signal_strength += 1.5
                    reason.append("Price is in an uptrend (20-day SMA > 50-day SMA)")
                else:
                    current_signals['Moving Average'] = "SELL (Downtrend)"
                    signal_strength -= 1.5
                    reason.append("Price is in a downtrend (20-day SMA < 50-day SMA)")
            
            # Bollinger Bands Signal
            if latest_upper_band is not None and latest_lower_band is not None:
                if latest_close > latest_upper_band:
                    current_signals['Bollinger Bands'] = "SELL (Above Upper Band)"
                    signal_strength -= 1
                    reason.append("Price is above the upper Bollinger Band")
                elif latest_close < latest_lower_band:
                    current_signals['Bollinger Bands'] = "BUY (Below Lower Band)"
                    signal_strength += 1
                    reason.append("Price is below the lower Bollinger Band")
                else:
                    current_signals['Bollinger Bands'] = "NEUTRAL"
            
            # Volume Analysis
            if latest_volume > avg_volume * 1.5:
                current_signals['Volume'] = "HIGH (Confirming Move)"
                if signal_strength > 0:
                    signal_strength += 1
                    reason.append("High volume confirming bullish move")
                elif signal_strength < 0:
                    signal_strength -= 1
                    reason.append("High volume confirming bearish move")
            else:
                current_signals['Volume'] = "NORMAL"
            
            # Determine overall recommendation
            if signal_strength >= 3:
                trade_recommendation = "STRONG BUY"
            elif signal_strength >= 1:
                trade_recommendation = "BUY"
            elif signal_strength <= -3:
                trade_recommendation = "STRONG SELL"
            elif signal_strength <= -1:
                trade_recommendation = "SELL"
            else:
                trade_recommendation = "HOLD"
            
            # Adjust recommendation based on risk appetite
            if risk_appetite < 3 and trade_recommendation in ["BUY", "STRONG BUY"]:
                if signal_strength < 4:  # Higher threshold for conservative investors
                    trade_recommendation = "HOLD"
                    reason.append("Low risk appetite suggests caution")
            elif risk_appetite > 7 and trade_recommendation == "HOLD":
                if signal_strength > 0:
                    trade_recommendation = "BUY"
                    reason.append("High risk tolerance suggests opportunistic entry")
                elif signal_strength < 0:
                    trade_recommendation = "SELL"
                    reason.append("High risk tolerance suggests opportunistic exit")
        
        # Display trading signals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Current Trading Signals for {stock_symbol}")
            
            # Display each signal
            for signal_type, signal_value in current_signals.items():
                signal_color = "green" if "BUY" in signal_value else "red" if "SELL" in signal_value else "gray"
                st.markdown(f"**{signal_type}:** <span style='color:{signal_color}'>{signal_value}</span>", unsafe_allow_html=True)
        
        with col2:
            # Display the overall recommendation
            rec_color = "green" if "BUY" in trade_recommendation else "red" if "SELL" in trade_recommendation else "gray"
            st.subheader("Trading Recommendation")
            st.markdown(f"<h2 style='color:{rec_color}'>{trade_recommendation}</h2>", unsafe_allow_html=True)
            
            if reason:
                st.markdown("**Based on:**")
                for point in reason:
                    st.markdown(f"• {point}")
        
        # Technical indicator charts
        st.subheader("Technical Indicators")
        
        # Add real-time update button and auto-refresh for intraday trading
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Data as of:** {hist.index[-1]}")
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
                
        # Auto-refresh for intraday trading
        if trading_mode == "Intraday Trading":
            st.checkbox("Auto-refresh (every minute)", key="auto_refresh")
            if st.session_state.get("auto_refresh", False):
                st.markdown("Auto-refreshing data every minute...")
                time_wait = 60
                st.cache_data.clear()
                time.sleep(1)  # To avoid excessive refreshing
                st.rerun()
        
        # Create tabs for technical indicators - different set based on trading mode
        if trading_mode == "Intraday Trading":
            tab_names = ["MACD", "RSI", "Bollinger Bands", "Moving Averages", "Volume Profile"]
        else:
            tab_names = ["MACD", "RSI", "Bollinger Bands", "Moving Averages"]
            
        # Create the tabs dynamically
        tabs = st.tabs(tab_names)
        
        # MACD tab
        with tabs[0]:
            # MACD Chart
            if 'macd' in hist.columns and 'macd_signal' in hist.columns:
                fig_macd = go.Figure()
                
                # Add MACD line
                fig_macd.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                ))
                
                # Add Signal line
                fig_macd.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=1.5)
                ))
                
                # Add Histogram
                fig_macd.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['macd_diff'],
                    name='Histogram',
                    marker=dict(
                        color=np.where(hist['macd_diff'] >= 0, 'green', 'red'),
                        opacity=0.7
                    )
                ))
                
                fig_macd.update_layout(
                    title='MACD (Moving Average Convergence Divergence)',
                    height=400,
                    xaxis_title='Date',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                st.markdown("""
                **MACD Interpretation:**
                - When the MACD line crosses above the signal line: Bullish signal
                - When the MACD line crosses below the signal line: Bearish signal
                - Histogram shows the difference between MACD and signal line
                """)
            else:
                st.warning("MACD data not available for the selected time period.")
        
        # RSI tab
        with tabs[1]:
            # RSI Chart
            if 'rsi' in hist.columns:
                fig_rsi = go.Figure()
                
                # Add RSI line
                fig_rsi.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1.5)
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_shape(
                    type="line", line=dict(dash='dash', width=1, color="red"),
                    y0=70, y1=70, x0=hist.index[0], x1=hist.index[-1]
                )
                fig_rsi.add_shape(
                    type="line", line=dict(dash='dash', width=1, color="green"),
                    y0=30, y1=30, x0=hist.index[0], x1=hist.index[-1]
                )
                
                fig_rsi.update_layout(
                    title='RSI (Relative Strength Index)',
                    height=400,
                    xaxis_title='Date',
                    yaxis=dict(title='RSI Value', range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.markdown("""
                **RSI Interpretation:**
                - RSI > 70: Stock is overbought (potential sell signal)
                - RSI < 30: Stock is oversold (potential buy signal)
                - 30-70 range: Neutral trend
                """)
            else:
                st.warning("RSI data not available for the selected time period.")
        
        # Bollinger Bands tab
        with tabs[2]:
            # Bollinger Bands Chart
            if all(x in hist.columns for x in ['bollinger_high', 'bollinger_low', 'bollinger_mavg']):
                fig_bb = go.Figure()
                
                # Add price line
                fig_bb.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=1.5)
                ))
                
                # Add Bollinger Bands
                fig_bb.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['bollinger_high'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['bollinger_low'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='green', width=1, dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['bollinger_mavg'],
                    mode='lines',
                    name='Middle Band (SMA)',
                    line=dict(color='gray', width=1)
                ))
                
                fig_bb.update_layout(
                    title='Bollinger Bands',
                    height=400,
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
                st.markdown("""
                **Bollinger Bands Interpretation:**
                - Price near upper band: Potentially overbought
                - Price near lower band: Potentially oversold
                - Width of bands indicates volatility
                - Price crossing from outside to inside bands may signal trend reversal
                """)
            else:
                st.warning("Bollinger Bands data not available for the selected time period.")
        
        # Moving Averages tab
        with tabs[3]:
            # Moving Averages Chart
            if 'sma_20' in hist.columns and 'sma_50' in hist.columns:
                fig_ma = go.Figure()
                
                # Add price line
                fig_ma.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=1)
                ))
                
                # Add Moving Averages
                fig_ma.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['sma_20'],
                    mode='lines',
                    name='20-day SMA',
                    line=dict(color='blue', width=1.5)
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['sma_50'],
                    mode='lines',
                    name='50-day SMA',
                    line=dict(color='orange', width=1.5)
                ))
                
                if 'sma_200' in hist.columns:
                    fig_ma.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['sma_200'],
                        mode='lines',
                        name='200-day SMA',
                        line=dict(color='red', width=1.5)
                    ))
                
                fig_ma.update_layout(
                    title='Moving Averages',
                    height=400,
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                st.markdown("""
                **Moving Averages Interpretation:**
                - 20-day SMA crosses above 50-day SMA: Bullish signal (Golden Cross if 50-day crosses above 200-day)
                - 20-day SMA crosses below 50-day SMA: Bearish signal (Death Cross if 50-day crosses below 200-day)
                - Price above moving averages: Uptrend
                - Price below moving averages: Downtrend
                """)
            else:
                st.warning("Moving Average data not available for the selected time period.")
                
        # Volume Profile tab - only for intraday trading
        if trading_mode == "Intraday Trading" and len(tabs) > 4:
            with tabs[4]:
                st.subheader("Volume Profile (Price by Volume)")
                
                if not hist.empty:
                    # Create volume profile analysis
                    price_range = hist['Close'].max() - hist['Close'].min()
                    bin_size = price_range / 20  # Create 20 price bins
                    
                    # Create price bins
                    if bin_size > 0:
                        bins = np.arange(hist['Close'].min(), hist['Close'].max() + bin_size, bin_size)
                        labels = [f"₹{round(b, 2)}" for b in bins[:-1]]
                        
                        # Categorize prices into bins
                        hist['price_bin'] = pd.cut(hist['Close'], bins=bins, labels=labels)
                        
                        # Group by price bins and sum volumes (with observed=True to silence warning)
                        volume_profile = hist.groupby('price_bin', observed=True)['Volume'].sum().reset_index()
                        
                        # Create horizontal bar chart for volume profile
                        fig_vp = go.Figure()
                        fig_vp.add_trace(go.Bar(
                            y=volume_profile['price_bin'],
                            x=volume_profile['Volume'],
                            orientation='h',
                            marker=dict(color='skyblue'),
                            name='Volume'
                        ))
                        
                        # Add vertical line for current price
                        fig_vp.add_vline(
                            x=0,  # Doesn't matter for y-axis reference
                            line=dict(color="red", width=2, dash="dash"),
                            annotation_text=f"Current Price: ₹{hist['Close'].iloc[-1]:.2f}",
                            annotation_position="top right"
                        )
                        
                        # Update layout
                        fig_vp.update_layout(
                            title="Volume Profile Analysis (Price by Volume)",
                            xaxis_title="Volume",
                            yaxis_title="Price Levels",
                            height=500,
                            hoverlabel=dict(bgcolor="white", font_size=12),
                            yaxis=dict(autorange="reversed")  # Higher prices at top
                        )
                        
                        st.plotly_chart(fig_vp, use_container_width=True)
                        
                        # Identify high volume nodes
                        volume_threshold = volume_profile['Volume'].mean() * 1.5
                        high_volume_nodes = volume_profile[volume_profile['Volume'] >= volume_threshold]
                        
                        # Display high volume nodes interpretation
                        if not high_volume_nodes.empty:
                            st.subheader("High Volume Price Levels (Support/Resistance)")
                            
                            for _, node in high_volume_nodes.iterrows():
                                price_level = node['price_bin']
                                current_price = hist['Close'].iloc[-1]
                                current_bin = pd.cut([current_price], bins=bins, labels=labels)[0]
                                
                                if price_level == current_bin:
                                    st.markdown(f"• **Current Trading Level** at {price_level} - High volume suggests strong current interest")
                                elif price_level < current_bin:
                                    st.markdown(f"• **Support Level** at {price_level} - High volume below current price")
                                else:
                                    st.markdown(f"• **Resistance Level** at {price_level} - High volume above current price")
                            
                            st.markdown("""
                            **Volume Profile Interpretation:**
                            - High volume price levels act as support below current price and resistance above
                            - Prices tend to move quickly through low volume areas
                            - Trading at a high volume node suggests strong interest at current level
                            - For intraday trading, these levels are key for placing stop loss and take profit orders
                            """)
                    else:
                        st.warning("Insufficient price range for volume profile analysis")
                else:
                    st.warning("Not enough data for volume profile analysis")
        
        # Trading Strategy Section
        st.markdown("---")
        st.subheader("Trading Strategy")
        
        # Entry and Exit Strategy
        if not hist.empty and latest_close > 0:
            strategy_col1, strategy_col2 = st.columns(2)
            
            with strategy_col1:
                st.markdown("### Entry Strategy")
                entry_strategy = []
                
                if trade_recommendation in ["BUY", "STRONG BUY"]:
                    entry_strategy.append("• **Enter now** at market price")
                    entry_strategy.append(f"• Set stop loss at ₹{latest_close * (1 - stop_loss_percent/100):.2f} ({stop_loss_percent}% below current price)")
                    entry_strategy.append(f"• Set take profit at ₹{latest_close * (1 + take_profit_percent/100):.2f} ({take_profit_percent}% above current price)")
                    
                    # Calculate position size based on risk
                    risk_per_trade = investment_amount * (risk_appetite / 100)
                    max_position_size = min(investment_amount, risk_per_trade * 10)
                    recommended_position = round(max_position_size)
                    entry_strategy.append(f"• Recommended position size: ₹{recommended_position:,}")
                    
                    # Volume confirmation
                    if current_signals.get('Volume') == "HIGH (Confirming Move)":
                        entry_strategy.append("• High volume confirms signal strength")
                    else:
                        entry_strategy.append("• Consider waiting for increased volume confirmation")
                else:
                    entry_strategy.append("• **Not recommended** for entry at current price")
                    if trade_recommendation == "HOLD":
                        entry_strategy.append("• Monitor for improved entry signals")
                        if any("uptrend" in s.lower() for s in reason):
                            entry_strategy.append("• Consider buying on dips if uptrend continues")
                    elif "SELL" in trade_recommendation:
                        entry_strategy.append("• Avoid buying in current downtrend")
                        entry_strategy.append("• Wait for trend reversal confirmation")
                
                for point in entry_strategy:
                    st.markdown(point)
            
            with strategy_col2:
                st.markdown("### Exit Strategy")
                exit_strategy = []
                
                if trade_recommendation in ["SELL", "STRONG SELL"]:
                    exit_strategy.append("• **Exit now** at market price")
                    exit_strategy.append("• Avoid holding during downtrend")
                elif trade_recommendation == "HOLD" and any("downtrend" in s.lower() for s in reason):
                    exit_strategy.append("• Consider partial profit taking")
                    exit_strategy.append(f"• Tighten stop loss to ₹{latest_close * (1 - stop_loss_percent/200):.2f} ({stop_loss_percent/2}% below current price)")
                else:
                    exit_strategy.append(f"• Hold with stop loss at ₹{latest_close * (1 - stop_loss_percent/100):.2f}")
                    exit_strategy.append(f"• Maximum holding period: {max_holding_days} days")
                    exit_strategy.append(f"• Take profit at ₹{latest_close * (1 + take_profit_percent/100):.2f}")
                    
                    if trade_recommendation in ["BUY", "STRONG BUY"]:
                        exit_strategy.append("• Consider trailing stop loss as price increases")
                        exit_strategy.append("• Exit half position at first target, move stop loss to breakeven")
                
                for point in exit_strategy:
                    st.markdown(point)
                    
            # Risk Management
            st.markdown("### Risk Management")
            st.markdown(f"• Total position should not exceed {20 + risk_appetite * 3}% of your portfolio")
            st.markdown(f"• Maximum loss per trade: ₹{investment_amount * (stop_loss_percent/100):.2f}")
            if trading_mode == "Intraday Trading":
                st.markdown("• Exit all positions by end of day regardless of signals")
                # Get exit time from market close time and buffer
                current_date = datetime.now().date()
                market_close_datetime = datetime.combine(current_date, market_close_time) if 'market_close_time' in locals() else None
                if market_close_datetime:
                    exit_time = market_close_datetime - timedelta(minutes=exit_buffer_minutes) if 'exit_buffer_minutes' in locals() else market_close_datetime
                    exit_time_str = exit_time.strftime("%H:%M")
                    st.markdown(f"• Latest exit time: {exit_time_str}")
            st.markdown("• Track performance and adjust position sizing based on win/loss ratio")
        else:
            st.warning("Insufficient data to generate trading strategy recommendations.")
        
        # Download buttons
        st.markdown("---")
        st.subheader("Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_historical = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Historical Data CSV",
                data=csv_historical,
                file_name=f"{stock_symbol}_historical_data.csv",
                mime="text/csv",
            )
        
        with col2:
            csv_metrics = raw_metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Financial Metrics CSV",
                data=csv_metrics,
                file_name=f"{stock_symbol}_financial_metrics.csv",
                mime="text/csv",
            )

# Run the main function
if __name__ == "__main__":
    display_stock_data()
    
    # Add footer
    st.markdown("---")
    st.caption("Data provided by Yahoo Finance. This app is for informational purposes only and does not constitute financial advice.")
