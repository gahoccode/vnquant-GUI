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
import datetime

# Import our custom modules
import data_loader as dl
import streamlit_components as sc
# Import visualization module
import visualization as viz

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
        
        # Add return calculation method selection
        return_method = st.selectbox(
            "Return Calculation Method",
            ["pct_change", "log"],
            index=0,
            help="Select the method to calculate returns: percentage change (pct_change) or logarithmic returns (log)"
        )
        
        # Update inputs with the selected return method
        inputs['return_method'] = return_method
        
        # Add explanation about return calculation methods
        with st.expander("About Return Calculation Methods", expanded=False):
            st.markdown("""
            **Percentage Returns (pct_change)**
            - Simple percentage change from one period to the next
            - More intuitive and directly interpretable (e.g., 5% means price increased by 5%)
            - Commonly used in financial reporting and by many practitioners
            
            **Logarithmic Returns (log)**
            - Natural logarithm of the ratio of prices
            - Mathematically convenient for multi-period analysis (log returns are additive over time)
            - Better statistical properties (more normally distributed)
            - Often preferred in academic research and sophisticated risk models
            
            For most purposes, both methods will yield similar results, especially for short time horizons and small returns.
            However, the differences become more pronounced with larger price movements or when compounding over longer periods.
            """)
        
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
                    num_port=5000,
                    risk_free_rate=inputs.get('risk_free_rate', 0.03),
                    return_method=inputs.get('return_method', 'pct_change')
                )
                
                if mpt_data:
                    # Get price data for time series chart
                    price_data = pd.DataFrame()
                    for symbol in symbols:
                        _, adj_price = get_price_columns(data, symbol, inputs.get('table_style', 'prefix'))
                        if adj_price is not None:
                            price_data[symbol] = adj_price
                    
                    # Set equal heights for charts
                    chart_height = 500
                    pie_chart_height = 400
                    
                    # Portfolio weights charts at the top
                    st.write("### Optimal Portfolio Weights")
                    
                    weight_cols = st.columns(3)
                    
                    # Max Sharpe Ratio Portfolio
                    with weight_cols[0]:
                        max_sharpe = mpt_data['max_sharpe']
                        st.write("#### Maximum Sharpe Ratio")
                        st.metric("Expected Return", f"{max_sharpe['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{max_sharpe['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.2f}")
                        
                        weights_chart = viz.create_optimal_portfolio_weights_chart(
                            max_sharpe['weights'],
                            "Max Sharpe Portfolio"
                        )
                        if weights_chart:
                            weights_chart.update_layout(height=pie_chart_height)
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Minimum Variance Portfolio
                    with weight_cols[1]:
                        min_var = mpt_data['min_variance']
                        st.write("#### Minimum Variance")
                        st.metric("Expected Return", f"{min_var['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{min_var['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{min_var['sharpe']:.2f}")
                        
                        weights_chart = viz.create_optimal_portfolio_weights_chart(
                            min_var['weights'],
                            "Min Variance Portfolio"
                        )
                        if weights_chart:
                            weights_chart.update_layout(height=pie_chart_height)
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Maximum Return Portfolio
                    with weight_cols[2]:
                        max_ret = mpt_data['max_return']
                        st.write("#### Maximum Return")
                        st.metric("Expected Return", f"{max_ret['return']:.2f}%")
                        st.metric("Risk (Volatility)", f"{max_ret['risk']:.2f}%")
                        st.metric("Sharpe Ratio", f"{max_ret['sharpe']:.2f}")
                        
                        weights_chart = viz.create_optimal_portfolio_weights_chart(
                            max_ret['weights'],
                            "Max Return Portfolio"
                        )
                        if weights_chart:
                            weights_chart.update_layout(height=pie_chart_height)
                            st.plotly_chart(weights_chart, use_container_width=True)
                    
                    # Create dashboard layout for the main charts
                    col1, col2 = st.columns([1, 1])
                    
                    # Efficient Frontier in first column
                    with col1:
                        ef_chart = viz.create_efficient_frontier_chart(mpt_data)
                        if ef_chart:
                            # Update height to match
                            ef_chart.update_layout(height=chart_height)
                            st.plotly_chart(ef_chart, use_container_width=True)
                    
                    # Portfolio Metrics in second column
                    with col2:
                        # Create a comparison chart for the three optimal portfolios
                        comparison_chart = viz.create_mpt_comparison_chart(mpt_data)
                        if comparison_chart:
                            # Update height to match
                            comparison_chart.update_layout(height=chart_height)
                            st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Performance Over Time chart below Efficient Frontier (full width)
                    time_series = viz.create_stock_price_time_series(mpt_data, price_data)
                    if time_series:
                        # Update height to match
                        time_series.update_layout(height=chart_height)
                        st.plotly_chart(time_series, use_container_width=True)
                    
                    # Display the metrics and weights as tables instead of download options
                    st.subheader("Portfolio Metrics")
                    
                    # Create a DataFrame with the metrics
                    metrics_df = pd.DataFrame({
                        "Portfolio": ["Max Sharpe Ratio", "Min Variance", "Max Return"],
                        "Expected Return (%)": [
                            mpt_data["max_sharpe"]["return"] * 100,
                            mpt_data["min_variance"]["return"] * 100,
                            mpt_data["max_return"]["return"] * 100
                        ],
                        "Risk (%)": [
                            mpt_data["max_sharpe"]["risk"] * 100,
                            mpt_data["min_variance"]["risk"] * 100,
                            mpt_data["max_return"]["risk"] * 100
                        ],
                        "Sharpe Ratio": [
                            mpt_data["max_sharpe"]["sharpe"],
                            mpt_data["min_variance"]["sharpe"],
                            mpt_data["max_return"]["sharpe"]
                        ]
                    })
                    
                    # Display metrics table
                    st.dataframe(metrics_df)
                    
                    # Create a DataFrame with the weights
                    weights_data = {}
                    for symbol in symbols:
                        weights_data[symbol] = [
                            mpt_data["max_sharpe"]["weights"][symbol] * 100,
                            mpt_data["min_variance"]["weights"][symbol] * 100,
                            mpt_data["max_return"]["weights"][symbol] * 100
                        ]
                    
                    weights_df = pd.DataFrame(
                        weights_data,
                        index=["Max Sharpe Ratio", "Min Variance", "Max Return"]
                    )
                    
                    st.subheader("Portfolio Weights (%)")
                    st.dataframe(weights_df)
                    
                    # Add detailed explanation
                    with st.expander("Understanding the Results", expanded=False):
                        st.markdown(f"""
                        ### How to Interpret These Results
                        
                        **Monte Carlo Simulation**: This analysis uses Monte Carlo simulation to generate thousands of random portfolio allocations. Each dot in the scatter plot represents one possible portfolio with a unique allocation of assets.
                        
                        **Efficient Frontier**: The curved line represents the efficient frontier - portfolios that offer the highest expected return for a given level of risk.
                        
                        **Sharpe Ratio**: The Sharpe ratio measures risk-adjusted performance, calculated as (Return - Risk-Free Rate) / Risk. A higher Sharpe ratio indicates better risk-adjusted performance. The current risk-free rate is set to **{inputs.get('risk_free_rate', 0.03)*100:.1f}%**.
                        
                        **Three Optimal Portfolios**:
                        
                        1. **Maximum Sharpe Ratio Portfolio**: Offers the best risk-adjusted return (highest Sharpe ratio)
                        2. **Minimum Variance Portfolio**: Has the lowest risk (volatility)
                        3. **Maximum Return Portfolio**: Provides the highest expected return, typically with higher risk
                        
                        **Portfolio Weights**: The pie charts show the allocation of assets in each optimal portfolio. These weights represent the percentage of your investment that should be allocated to each stock.
                        """)
                else:
                    st.warning("Could not calculate MPT metrics. Check if all symbols have sufficient data.")
            except Exception as e:
                st.error(f"Error calculating MPT metrics: {str(e)}")
                st.write(traceback.format_exc())
    else:
        st.warning("Modern Portfolio Theory requires at least two symbols. Please add more symbols in the sidebar.")


def calculate_mpt_portfolio(data, symbols, price_columns_func, table_style="prefix", num_port=5000, risk_free_rate=0.03, return_method="pct_change"):
    """
    Calculate Modern Portfolio Theory (MPT) metrics and optimal portfolios.
    According to MPT, we can find optimal portfolios that maximize return for a given level of risk.
    
    Args:
        data: DataFrame with stock data
        symbols: List of stock symbols
        price_columns_func: Function to get price columns
        table_style: Style of table ('prefix' or 'suffix')
        num_port: Number of random portfolios to simulate
        risk_free_rate: Annual risk-free rate (decimal) for Sharpe ratio calculation
        return_method: Method to calculate returns ('pct_change' or 'log')
        
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
    
    # Sort index chronologically
    price_data = price_data.sort_index()
    
    # Clean data by removing NaN values
    df_clean = price_data.dropna()
    
    # Calculate returns based on selected method
    if return_method == "log":
        # Calculate log returns
        returns = np.log(df_clean / df_clean.shift(1)).dropna()
    else:
        # Calculate percentage returns using pct_change
        returns = df_clean.pct_change().dropna()
    
    # Calculate covariance matrix of returns (annualized)
    cov_mat = returns.cov() * 252
    
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
        port_ret = np.sum(returns.mean() * wts)
        
        # Annualize the return (252 trading days)
        if return_method == "log":
            port_ret = (port_ret + 1) ** 252 - 1
        else:
            port_ret = (1 + port_ret) ** 252 - 1
            
        port_returns[i] = port_ret * 100  # Convert to percentage
        
        # Calculate portfolio risk (standard deviation)
        port_sd = np.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))
        port_risk[i] = port_sd * 100  # Convert to percentage
        
        # Calculate Sharpe Ratio using the specified risk-free rate
        sr = (port_ret - risk_free_rate) / port_sd if port_sd > 0 else 0
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
        'Returns': port_returns,  # Already in percentage
        'Risk': port_risk,        # Already in percentage
        'Sharpe': sharpe_ratio
    })
    
    return {
        'simulation_data': simulation_data,
        'max_sharpe': {
            'return': max_sr_ret,  # Already in percentage
            'risk': max_sr_risk,   # Already in percentage
            'sharpe': sharpe_ratio[max_sr_idx],
            'weights': max_sr_weights
        },
        'max_return': {
            'return': max_ret_ret,  # Already in percentage
            'risk': max_ret_risk,   # Already in percentage
            'sharpe': sharpe_ratio[max_ret_idx],
            'weights': max_ret_weights
        },
        'min_variance': {
            'return': min_var_ret,  # Already in percentage
            'risk': min_var_risk,   # Already in percentage
            'sharpe': sharpe_ratio[min_var_idx],
            'weights': min_var_weights
        }
    }
