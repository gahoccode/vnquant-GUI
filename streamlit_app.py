"""
Main Streamlit application for VNQuant Stock Analyzer.
This file connects the frontend components with the backend analysis.
"""

import streamlit as st
import pandas as pd
import traceback

# Import our custom modules
import data_loader as dl
import stock_analysis as sa
import visualization as viz
import streamlit_components as sc

# Set up the page configuration
sc.setup_page_config()

# Display the app header
sc.display_header()

# Get user inputs from sidebar
inputs = sc.sidebar_inputs()

# Display about information in sidebar
sc.display_about_sidebar()

# Get portfolio weights from sidebar if multiple symbols
weights = None
if len(inputs["symbols"]) > 1:
    weights = sc.get_portfolio_weights(inputs["symbols"])

# Load data based on user inputs
try:
    if inputs["use_file"]:
        try:
            data = dl.load_stock_data_from_file(inputs["file_path"])
            sc.display_success(f"Data loaded from {inputs['file_path']}")
        except Exception as e:
            sc.display_error(f"Error loading file: {str(e)}")
            # Fallback to API data
            data = dl.load_stock_data(
                inputs["symbols"], 
                inputs["start_date"], 
                inputs["end_date"], 
                inputs["data_source"], 
                table_style=inputs["table_style"]
            )
    else:
        data = dl.load_stock_data(
            inputs["symbols"], 
            inputs["start_date"], 
            inputs["end_date"], 
            inputs["data_source"], 
            table_style=inputs["table_style"]
        )
    
    # Display the data
    sc.display_data(data)
    
    # Visualization section
    st.header("Stock Visualization")
    
    # Generate and display chart when button is clicked
    if st.button("Generate Chart"):
        with st.spinner("Generating chart..."):
            try:
                # For visualization, we can only use one symbol at a time
                if len(inputs["symbols"]) > 1:
                    sc.display_warning("Visualization is only available for a single symbol. Using the first symbol for the chart.")
                    chart_symbol = inputs["symbols"][0]
                else:
                    chart_symbol = inputs["symbols"][0]
                
                # Generate the chart
                fig = viz.generate_candlestick_chart(
                    chart_symbol, 
                    inputs["start_date"], 
                    inputs["end_date"], 
                    inputs["data_source"], 
                    inputs["advanced_indicators"]
                )
                
                # Display the chart
                sc.display_chart(fig)
                
                # Display statistics if data is available
                if len(data) > 0:
                    st.header("Statistics")
                    
                    # For multiple symbols, display stats for each symbol in tabs
                    if len(inputs["symbols"]) > 1:
                        # Create tabs for each symbol
                        symbol_tabs = sc.create_symbol_tabs(inputs["symbols"])
                        
                        for i, symbol in enumerate(inputs["symbols"]):
                            with symbol_tabs[i]:
                                try:
                                    # Get price data for this symbol
                                    close_prices, adjust_prices = dl.get_price_columns(
                                        data, 
                                        symbol, 
                                        inputs["table_style"]
                                    )
                                    
                                    if close_prices is not None and adjust_prices is not None:
                                        # Calculate and display price statistics with both regular and adjusted prices
                                        price_stats = sa.calculate_price_statistics(close_prices, adjust_prices)
                                        sc.display_price_statistics(price_stats)
                                        
                                        # Calculate and display returns using adjusted prices
                                        returns = sa.calculate_returns(adjust_prices)
                                        sc.display_returns(returns)
                                        
                                        # Show returns chart
                                        st.subheader(f"{symbol} Returns Chart")
                                        returns_chart = viz.create_returns_chart(returns, symbol)
                                        if returns_chart:
                                            st.plotly_chart(returns_chart, use_container_width=True)
                                    else:
                                        sc.display_warning(f"No close or adjust price data found for {symbol}")
                                except Exception as e:
                                    sc.display_error(f"Error displaying statistics for {symbol}: {str(e)}")
                        
                        # Add portfolio performance section for multiple symbols
                        st.header("Portfolio Performance")
                        st.info("Portfolio calculations use adjusted prices as required by the rules.")
                        
                        # Calculate and display portfolio performance
                        try:
                            portfolio_stats = sa.calculate_portfolio_performance(
                                data, 
                                inputs["symbols"],
                                dl.get_price_columns,
                                inputs["table_style"], 
                                weights
                            )
                            
                            if portfolio_stats:
                                # Display portfolio metrics
                                portfolio_cols = st.columns(len(portfolio_stats))
                                for i, (metric, value) in enumerate(portfolio_stats.items()):
                                    with portfolio_cols[i]:
                                        st.metric(
                                            metric, 
                                            f"{value:.2f}" if metric == "Sharpe Ratio" else f"{value:.2f}%"
                                        )
                                
                                # Show portfolio performance chart
                                st.subheader("Portfolio Performance Chart")
                                portfolio_chart = viz.create_portfolio_comparison_chart(portfolio_stats)
                                if portfolio_chart:
                                    st.plotly_chart(portfolio_chart, use_container_width=True)
                            else:
                                st.warning("Could not calculate portfolio performance. Check if all symbols have valid data.")
                        except Exception as e:
                            st.error(f"Error calculating portfolio performance: {str(e)}")
                    else:
                        # Single symbol stats
                        close_prices, adjust_prices = dl.get_price_columns(
                            data, 
                            inputs["symbols"][0], 
                            inputs["table_style"]
                        )
                        
                        if close_prices is not None and adjust_prices is not None:
                            # Calculate and display price statistics with both regular and adjusted prices
                            price_stats = sa.calculate_price_statistics(close_prices, adjust_prices)
                            sc.display_price_statistics(price_stats)
                            
                            # Calculate and display returns using adjusted prices
                            returns = sa.calculate_returns(adjust_prices)
                            sc.display_returns(returns)
                            
                            # Show returns chart
                            st.subheader(f"{inputs['symbols'][0]} Returns Chart")
                            returns_chart = viz.create_returns_chart(returns, inputs["symbols"][0])
                            if returns_chart:
                                st.plotly_chart(returns_chart, use_container_width=True)
                            
                            # Show risk metrics
                            risk_metrics = sa.calculate_risk_metrics(adjust_prices)
                            if risk_metrics:
                                st.subheader("Risk Metrics")
                                risk_cols = st.columns(len(risk_metrics))
                                for i, (metric, value) in enumerate(risk_metrics.items()):
                                    with risk_cols[i]:
                                        st.metric(metric, f"{value:.2f}%")
                        else:
                            sc.display_warning(f"No close or adjust price data found for {inputs['symbols'][0]}")
            except Exception as e:
                sc.display_error(f"Error displaying chart: {str(e)}")
                sc.display_info("Try changing the parameters or data source.")
                # Print detailed error for debugging
                st.write(traceback.format_exc())
except Exception as e:
    sc.display_error(f"Error: {str(e)}")
    # Print detailed error for debugging
    st.write(traceback.format_exc())

# Instructions on how to run the app
if __name__ == "__main__":
    # This code won't run in Streamlit, but it's here for documentation
    # Run the app with: streamlit run streamlit_app.py
    pass
