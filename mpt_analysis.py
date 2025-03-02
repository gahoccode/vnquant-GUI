"""
Modern Portfolio Theory (MPT) analysis module.
This module contains functions for MPT calculations and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback
from plotly.subplots import make_subplots

# Import our custom modules
import data_loader as dl
import streamlit_components as sc
# Import visualization functions from visualization module
from visualization import (
    create_efficient_frontier_chart,
    create_optimal_portfolio_weights_chart,
    create_stock_price_time_series,
    create_mpt_comparison_chart
)

def display_mpt_page(data, inputs):
    """
    Display the Modern Portfolio Theory page
    
    Args:
        data: DataFrame with stock data
        inputs: Dictionary with user inputs
    """
    import streamlit as st
    import traceback
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from data_loader import get_price_columns
    
    # Create a title row 
    st.title("Modern Portfolio Theory (MPT) Analysis")
    
    # Introduction to MPT
    with st.expander("What is Modern Portfolio Theory?", expanded=False):
        st.markdown("""
        **Modern Portfolio Theory (MPT)** is a mathematical framework for constructing a portfolio of assets 
        to maximize expected return for a given level of risk. The theory was pioneered by Harry Markowitz in 1952.
        
        Key concepts of MPT:
        
        1. **Risk and Return**: Every investment has both risk and return components.
        2. **Diversification**: By combining assets with different correlations, you can reduce portfolio risk.
        3. **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk.
        4. **Optimal Portfolios**:
           - **Maximum Sharpe Ratio Portfolio**: Offers the best risk-adjusted return
           - **Minimum Variance Portfolio**: Has the lowest possible risk
           - **Maximum Return Portfolio**: Provides the highest expected return (but with higher risk)
        
        MPT helps investors make more informed decisions by quantifying the benefits of diversification and finding the optimal balance between risk and return.
        """)
    
    # Get symbols from inputs
    symbols = inputs.get('symbols', [])
    
    if len(symbols) >= 2:
        st.write("### Portfolio Optimization")
        st.write("Analyzing optimal portfolios based on historical data...")
        
        # Add Monte Carlo explanation
        st.info("""
        **Method**: This analysis uses Monte Carlo simulation to generate 5,000 random portfolio allocations. 
        From these simulations, we identify the optimal portfolios (Maximum Sharpe Ratio, Minimum Variance, and Maximum Return).
        """)
        
        with st.spinner("Calculating MPT metrics..."):
            try:
                # Calculate MPT metrics
                mpt_data = calculate_mpt_portfolio(
                    data=data,
                    symbols=symbols,
                    price_columns_func=get_price_columns,
                    table_style=inputs.get('table_style', 'prefix'),
                    num_port=5000
                )
                
                if mpt_data:
                    # Get price data for time series chart
                    price_data = pd.DataFrame()
                    for symbol in symbols:
                        _, adj_price = get_price_columns(data, symbol, inputs.get('table_style', 'prefix'))
                        if adj_price is not None:
                            price_data[symbol] = adj_price
                    
                    # Create dashboard layout
                    col1, col2 = st.columns([1, 1])
                    
                    # Set equal heights for both charts
                    chart_height = 500
                    
                    # Efficient Frontier in first column
                    with col1:
                        ef_chart = create_efficient_frontier_chart(mpt_data)
                        if ef_chart:
                            # Update height to match
                            ef_chart.update_layout(height=chart_height)
                            st.plotly_chart(ef_chart, use_container_width=True)
                    
                    # Time series chart in second column
                    with col2:
                        # Create tabs for different views
                        view_tabs = st.tabs(["Performance Over Time", "Portfolio Metrics"])
                        
                        with view_tabs[0]:
                            time_series = create_stock_price_time_series(mpt_data, price_data)
                            if time_series:
                                # Update height to match
                                time_series.update_layout(height=chart_height)
                                st.plotly_chart(time_series, use_container_width=True)
                        
                        with view_tabs[1]:
                            # Create a comparison chart for the three optimal portfolios
                            comparison_chart = create_mpt_comparison_chart(mpt_data)
                            if comparison_chart:
                                # Update height to match
                                comparison_chart.update_layout(height=chart_height)
                                st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Portfolio weights charts in a row below
                    st.write("### Optimal Portfolio Weights")
                    
                    weight_cols = st.columns(3)
                    
                    # Max Sharpe Ratio Portfolio
                    with weight_cols[0]:
                        max_sharpe = mpt_data['max_sharpe']
                        st.write("#### Maximum Sharpe Ratio")
                        st.metric("Expected Return", f"{max_sharpe['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{max_sharpe['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.2f}")
                        
                        weights_chart = create_optimal_portfolio_weights_chart(
                            max_sharpe['weights'],
                            "Max Sharpe Portfolio"
                        )
                        if weights_chart:
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Minimum Variance Portfolio
                    with weight_cols[1]:
                        min_var = mpt_data['min_variance']
                        st.write("#### Minimum Variance")
                        st.metric("Expected Return", f"{min_var['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{min_var['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{min_var['sharpe']:.2f}")
                        
                        weights_chart = create_optimal_portfolio_weights_chart(
                            min_var['weights'],
                            "Min Variance Portfolio"
                        )
                        if weights_chart:
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Maximum Return Portfolio
                    with weight_cols[2]:
                        max_ret = mpt_data['max_return']
                        st.write("#### Maximum Return")
                        st.metric("Expected Return", f"{max_ret['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{max_ret['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{max_ret['sharpe']:.2f}")
                        
                        weights_chart = create_optimal_portfolio_weights_chart(
                            max_ret['weights'],
                            "Max Return Portfolio"
                        )
                        if weights_chart:
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Add detailed explanation
                    with st.expander("Understanding the Results", expanded=False):
                        st.markdown("""
                        ### How to Interpret These Results
                        
                        **Monte Carlo Simulation**: This analysis uses Monte Carlo simulation to generate thousands of random portfolio allocations. Each dot in the scatter plot represents one possible portfolio with a unique allocation of assets.
                        
                        **Efficient Frontier Chart**: The curved line shows the optimal portfolios that offer the highest return for a given level of risk. The three highlighted portfolios represent key optimal strategies:
                        
                        1. **Maximum Sharpe Ratio Portfolio** (star): Offers the best risk-adjusted return, balancing risk and return optimally.
                        2. **Minimum Variance Portfolio** (circle): Has the lowest possible risk, ideal for conservative investors.
                        3. **Maximum Return Portfolio** (diamond): Provides the highest expected return, but with higher risk.
                        
                        **Portfolio Weights**: The pie charts show how your investment should be allocated across different assets for each optimal portfolio strategy. These weights are determined by the Monte Carlo simulation and represent the specific allocations that achieve each portfolio's objective.
                        
                        **Sharpe Ratio**: A measure of risk-adjusted return. Higher is better. Calculated as (portfolio return - risk-free rate) / portfolio standard deviation.
                        """)
                else:
                    st.warning("Could not calculate MPT metrics. Check if all symbols have sufficient data.")
            except Exception as e:
                st.error(f"Error calculating MPT metrics: {str(e)}")
                st.write(traceback.format_exc())
    else:
        st.warning("Modern Portfolio Theory requires at least two symbols. Please add more symbols in the sidebar.")


def calculate_mpt_portfolio(data, symbols, price_columns_func, table_style="prefix", num_port=5000):
    """
    Calculate Modern Portfolio Theory (MPT) metrics and optimal portfolios.
    According to MPT, we can find optimal portfolios that maximize return for a given level of risk.
    
    Args:
        data: DataFrame with stock data
        symbols: List of stock symbols
        price_columns_func: Function to get price columns
        table_style: Style of table ('prefix' or 'suffix')
        num_port: Number of random portfolios to simulate
        
    Returns:
        Dictionary with MPT metrics and optimal portfolios
    """
    if not symbols or len(symbols) < 2:
        return None
    
    # Get adjusted prices for each symbol
    price_data = pd.DataFrame()
    for symbol in symbols:
        _, adj_price = price_columns_func(data, symbol, table_style)
        if adj_price is not None:
            price_data[symbol] = adj_price
    
    if price_data.empty or len(price_data.columns) < 2:
        return None
    
    # Clean data by removing NaN values
    df_clean = price_data.dropna()
    
    # Calculate log returns
    log_ret = np.log(df_clean / df_clean.shift(1)).dropna()
    
    # Calculate covariance matrix of log returns (annualized)
    cov_mat = log_ret.cov() * 252
    
    # Initialize arrays for portfolio weights, returns, risk, and Sharpe ratios
    all_wts = np.zeros((num_port, len(df_clean.columns)))
    port_returns = np.zeros(num_port)
    port_risk = np.zeros(num_port)
    sharpe_ratio = np.zeros(num_port)
    
    # =====================================================================
    # MONTE CARLO SIMULATION FOR PORTFOLIO OPTIMIZATION
    # =====================================================================
    # This section implements a Monte Carlo approach to find optimal portfolios:
    # 1. Generate thousands of random portfolio weights
    # 2. Calculate return, risk, and Sharpe ratio for each portfolio
    # 3. Identify the portfolios with max Sharpe ratio, min variance, and max return
    # =====================================================================
    
    np.random.seed(42)  # Set seed for reproducibility
    for i in range(num_port):
        # Generate random portfolio weights
        wts = np.random.uniform(size=len(df_clean.columns))
        wts = wts / np.sum(wts)  # Normalize to ensure weights sum to 1
        all_wts[i, :] = wts
        
        # Calculate portfolio return (annualized)
        port_ret = np.sum(log_ret.mean() * wts)
        port_ret = (port_ret + 1) ** 252 - 1  # Annualize the return (252 trading days)
        port_returns[i] = port_ret
        
        # Calculate portfolio risk (standard deviation)
        port_sd = np.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))
        port_risk[i] = port_sd
        
        # Calculate Sharpe Ratio, assuming a risk-free rate of 0%
        sr = port_ret / port_sd if port_sd > 0 else 0
        sharpe_ratio[i] = sr
    
    # Identify portfolios with max Sharpe ratio, max return, and minimum variance
    # These represent the three key optimal portfolios in MPT
    max_sr_idx = sharpe_ratio.argmax()
    max_ret_idx = port_returns.argmax()
    min_var_idx = port_risk.argmin()
    
    # Extract metrics for optimal portfolios
    max_sr_ret = port_returns[max_sr_idx]
    max_sr_risk = port_risk[max_sr_idx]
    max_sr_w = all_wts[max_sr_idx, :]
    
    max_ret_ret = port_returns[max_ret_idx]
    max_ret_risk = port_risk[max_ret_idx]
    max_ret_w = all_wts[max_ret_idx, :]
    
    min_var_ret = port_returns[min_var_idx]
    min_var_risk = port_risk[min_var_idx]
    min_var_w = all_wts[min_var_idx, :]
    
    # Create dictionary of weights for each optimal portfolio
    max_sr_weights = {symbol: weight for symbol, weight in zip(df_clean.columns, max_sr_w)}
    max_ret_weights = {symbol: weight for symbol, weight in zip(df_clean.columns, max_ret_w)}
    min_var_weights = {symbol: weight for symbol, weight in zip(df_clean.columns, min_var_w)}
    
    # Prepare simulation data for efficient frontier visualization
    simulation_data = pd.DataFrame({
        'Returns': port_returns,  # Keep as decimal, not percentage
        'Risk': port_risk,        # Keep as decimal, not percentage
        'Sharpe': sharpe_ratio
    })
    
    return {
        'simulation_data': simulation_data,
        'max_sharpe': {
            'return': max_sr_ret,  # Keep as decimal, not percentage
            'risk': max_sr_risk,   # Keep as decimal, not percentage
            'sharpe': sharpe_ratio[max_sr_idx],
            'weights': max_sr_weights
        },
        'max_return': {
            'return': max_ret_ret,  # Keep as decimal, not percentage
            'risk': max_ret_risk,   # Keep as decimal, not percentage
            'sharpe': sharpe_ratio[max_ret_idx],
            'weights': max_ret_weights
        },
        'min_variance': {
            'return': min_var_ret,  # Keep as decimal, not percentage
            'risk': min_var_risk,   # Keep as decimal, not percentage
            'sharpe': sharpe_ratio[min_var_idx],
            'weights': min_var_weights
        }
    }
