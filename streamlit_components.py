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
    uploaded_file = None
    if use_file:
        file_source = st.sidebar.radio("File Source", ["Upload File", "Specify Path"], index=0)
        
        if file_source == "Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        else:
            file_path = st.sidebar.text_input("CSV file path", value="2025-03-01T14-04_export.csv")
    
    # Stock symbols
    symbols_input = st.sidebar.text_input("Stock Symbols (comma-separated)", value="REE,FMC")
    symbols = [symbol.strip() for symbol in symbols_input.split(",")]
    
    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value=start_date)
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", value=end_date)
    
    # Add download button if data is available
    if 'data' in st.session_state and not st.session_state.data.empty:
        st.sidebar.download_button(
            label="📥 Download Price Data",
            data=st.session_state.data.to_csv(index=True),
            file_name=f"stock_prices_{start_date}_{end_date}.csv",
            mime="text/csv",
            help="Download the complete price data for the selected date range"
        )
    
    # Data source
    data_source = st.sidebar.selectbox("Data Source", options=["cafe", "VND"], index=0)
    
    # Table style
    table_style = st.sidebar.selectbox("Table Style", options=["prefix", "suffix"], index=0)
    
    # Add risk-free rate slider for MPT calculations
    st.sidebar.header("MPT Parameters")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, 
                                     help="Annual risk-free rate used in Sharpe ratio calculations")
    
    # Advanced indicators for visualization
    advanced_indicators = ["volume"]  # Default to showing volume only
    
    return {
        "use_file": use_file,
        "file_path": file_path,
        "uploaded_file": uploaded_file,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "data_source": data_source,
        "table_style": table_style,
        "advanced_indicators": advanced_indicators,
        "risk_free_rate": risk_free_rate / 100  # Convert percentage to decimal
    }

def display_about_sidebar():
    """Display about information in sidebar"""
    st.sidebar.header("About")
    st.sidebar.info("""
    ### Modern Portfolio Theory (MPT)
    Modern Portfolio Theory, developed by Harry Markowitz in 1952, is a mathematical framework for assembling a portfolio of assets that maximizes expected return for a given level of risk. The key insight of MPT is that an asset's risk and return should not be assessed individually, but by how it contributes to a portfolio's overall risk and return.
    
    In this application, MPT is implemented through:
    - Calculation of expected returns and covariance matrix from historical data
    - Generation of the efficient frontier showing optimal risk-return combinations
    - Identification of three key portfolios: Maximum Sharpe Ratio (best risk-adjusted return), Minimum Variance (lowest risk), and Maximum Return
    
    ### Monte Carlo Simulation
    Monte Carlo simulation is a computational technique that uses random sampling to obtain numerical results. In portfolio optimization:
    
    - We generate thousands (5,000) of random portfolio weights
    - Each portfolio's expected return, risk, and Sharpe ratio are calculated
    - This creates a "cloud" of possible portfolios in risk-return space
    - The efficient frontier emerges as the upper boundary of this cloud
    - Optimal portfolios are identified from these simulations
    
    This approach allows us to visualize the entire range of possible portfolios and identify those with the most desirable characteristics.
    """)
    
    # Add GitHub link at the bottom of the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/gahoccode/vnquant-GUI" target="_blank">
                <div style="background-color: #f8f9fa; width: 40px; height: 40px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin: 0 auto;">
                    <svg height="24" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true">
                        <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                    </svg>
                </div>
            </a>
            <p style="font-size: 0.8em; margin-top: 5px;">View on GitHub</p>
        </div>
        """, unsafe_allow_html=True
    )

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

def get_current_page():
    """
    Get the current page from the sidebar navigation
    
    Returns:
        String with the current page name
    """
    # Initialize session state for page navigation if it doesn't exist
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    st.sidebar.header("Navigation")
    
    # Create radio buttons for page navigation
    page = st.sidebar.radio(
        "Go to",
        options=["Home", "Modern Portfolio Theory"],
        index=0 if st.session_state.current_page == "Home" else 1,
        key="page_navigation"
    )
    
    # Update session state
    st.session_state.current_page = page
    
    return page

def display_data(data):
    """Display the loaded data"""
    st.header("Data")
    
    # Display data info
    st.subheader("Data Overview")
    st.write(f"Rows: {len(data)}")
    st.write(f"Columns: {len(data.columns)}")
    st.write(f"Date Range: {data.index.min()} to {data.index.max()}")
    
    # Function to display the data table
    st.subheader("Data Preview")
    st.dataframe(data.head(10))
    
    # Note: Download features have been removed

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
