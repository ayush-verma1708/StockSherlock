import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configure page
st.set_page_config(
    page_title="Stock Data Visualizer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Data Visualizer")
st.markdown("Enter a stock symbol to view financial data and interactive charts. Data provided by Yahoo Finance.")

# Input for stock symbol
col1, col2 = st.columns([3, 1])
with col1:
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL):", "AAPL").upper()
with col2:
    period = st.selectbox("Select Time Period:", 
                         options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                         index=3)

# Function to get stock data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol, period):
    try:
        # Get stock information
        stock = yf.Ticker(symbol)
        
        # Check if the stock exists by trying to get info
        info = stock.info
        if not info or 'regularMarketPrice' not in info:
            return None, None, None
        
        # Get historical market data
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None, None
        
        return stock, info, hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

# Main function to display stock data
def display_stock_data():
    if not stock_symbol:
        st.warning("Please enter a stock symbol.")
        return
    
    with st.spinner(f"Loading data for {stock_symbol}..."):
        stock, info, hist = get_stock_data(stock_symbol, period)
        
        if stock is None or info is None or hist is None:
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
                st.markdown(f"**Current Price:** ${current_price:.2f}")
                st.markdown(f"**Change:** <span style='color:{color}'>{change_symbol}{price_change:.2f} ({change_symbol}{price_change_percent:.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Current Price:** ${current_price}")
                st.markdown("**Change:** N/A")
            
        with col3:
            st.markdown(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}")
            st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**52-Week Range:** ${info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
        
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
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=500,
            yaxis2=dict(
                title="Volume",
                titlefont=dict(color="gray"),
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
                        formatted_metrics[key] = f"${value/1e9:.2f}B"
                    elif value >= 1e6:
                        formatted_metrics[key] = f"${value/1e6:.2f}M"
                    else:
                        formatted_metrics[key] = f"${value:,.2f}"
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
        
        # Download buttons
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
