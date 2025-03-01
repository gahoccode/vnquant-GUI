"""
Streamlit UI components module.
This module contains functions for creating and displaying UI components.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def setup_page_config():
    """Set up the page configuration"""
    st.set_page_config(
        page_title="VNQuant Stock Analyzer",
        page_icon="📈",
        layout="wide"
    )

def display_header():
    """Display the app header"""
    st.title("📊 VNQuant Stock Analyzer")
    st.markdown("""
    This app allows you to analyze Vietnamese stock data using the VNQuant library.
    """)

def sidebar_inputs():
    """
    Get user inputs from sidebar
    
    Returns:
        Dictionary with user inputs
    """
    st.sidebar.header("Data Parameters")
    
    # Option to use a file or API
    use_file = st.sidebar.checkbox("Use CSV file instead of API", value=False)
    
    file_path = None
    if use_file:
        file_path = st.sidebar.text_input("CSV file path", value="2025-03-01T14-04_export.csv")
    
    # Stock symbols
    symbols_input = st.sidebar.text_input("Stock Symbols (comma-separated)", value="REE,FMC")
    symbols = [symbol.strip() for symbol in symbols_input.split(",")]
    
    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value=start_date)
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", value=end_date)
    
    # Data source
    data_source = st.sidebar.selectbox("Data Source", options=["cafe", "VND"], index=0)
    
    # Table style
    table_style = st.sidebar.selectbox("Table Style", options=["prefix", "suffix"], index=0)
    
    # Advanced indicators for visualization
    st.sidebar.header("Visualization Parameters")
    
    advanced_indicators = []
    if st.sidebar.checkbox("Show Volume", value=True):
        advanced_indicators.append("volume")
    if st.sidebar.checkbox("Show MACD", value=False):
        advanced_indicators.append("macd")
    if st.sidebar.checkbox("Show RSI", value=False):
        advanced_indicators.append("rsi")
    
    return {
        "use_file": use_file,
        "file_path": file_path,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "data_source": data_source,
        "table_style": table_style,
        "advanced_indicators": advanced_indicators
    }

def display_about_sidebar():
    """Display about information in sidebar"""
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses the VNQuant library to analyze Vietnamese stock data.
    
    Data is loaded from the specified source and displayed in various formats.
    
    All stock performance calculations use adjusted prices as required by the rules.
    """)

def get_portfolio_weights(symbols):
    """
    Get portfolio weights from sidebar sliders
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        Dictionary with symbol weights
    """
    if len(symbols) <= 1:
        return None
    
    st.sidebar.header("Portfolio Weights")
    
    # Initialize weights dictionary
    weight_inputs = {}
    remaining_symbols = symbols.copy()
    
    # For all symbols except the last one, use sliders
    for i in range(len(symbols) - 1):
        symbol = remaining_symbols[0]
        remaining_symbols.remove(symbol)
        
        # Calculate maximum possible weight based on what's left to allocate
        max_weight = 100 - sum(weight_inputs.values()) if weight_inputs else 100
        default_weight = min(100.0 / len(symbols), max_weight)
        
        weight_inputs[symbol] = st.sidebar.slider(
            f"{symbol} weight (%)",
            min_value=0.0,
            max_value=float(max_weight),
            value=float(default_weight),
            step=1.0
        )
    
    # The last symbol gets whatever is left to make sum = 100%
    last_symbol = remaining_symbols[0]
    last_weight = 100.0 - sum(weight_inputs.values())
    weight_inputs[last_symbol] = last_weight
    
    # Display the last weight (not editable)
    st.sidebar.info(f"{last_symbol} weight: {last_weight:.1f}% (auto-calculated)")
    
    # Convert percentages to decimals
    return {symbol: weight / 100.0 for symbol, weight in weight_inputs.items()}

def display_data(data):
    """Display the loaded data"""
    st.header("Data")
    
    # Display data info
    st.subheader("Data Overview")
    st.write(f"Rows: {len(data)}")
    st.write(f"Columns: {len(data.columns)}")
    st.write(f"Date Range: {data.index.min()} to {data.index.max()}")
    
    # Display the data
    st.subheader("Data Preview")
    st.dataframe(data.head())

def display_success(message):
    """Display a success message"""
    st.success(message)

def display_error(message):
    """Display an error message"""
    st.error(message)

def display_warning(message):
    """Display a warning message"""
    st.warning(message)

def display_info(message):
    """Display an info message"""
    st.info(message)

def display_chart(fig):
    """Display a chart"""
    st.plotly_chart(fig, use_container_width=True)

def create_symbol_tabs(symbols):
    """Create tabs for each symbol"""
    return st.tabs(symbols)

def display_price_statistics(stats):
    """Display price statistics"""
    st.subheader("Price Statistics")
    
    # Create columns for metrics
    cols = st.columns(len(stats))
    
    # Display each metric in a column
    for i, (metric, value) in enumerate(stats.items()):
        with cols[i]:
            st.metric(metric, f"{value:,.2f}")

def display_returns(returns):
    """Display returns"""
    if returns:
        st.subheader("Returns")
        
        # Create columns for return metrics
        cols = st.columns(len(returns))
        
        # Display each return metric in a column
        for i, (period, value) in enumerate(returns.items()):
            with cols[i]:
                if value is not None:
                    st.metric(
                        period, 
                        f"{value:.2f}%", 
                        delta=f"{value:.2f}%",
                        delta_color="normal"
                    )
                else:
                    st.metric(period, "N/A")
    else:
        st.warning("Could not calculate returns. Insufficient data.")
