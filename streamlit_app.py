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
import mpt_analysis as mpt

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

# Get current page from navigation
current_page = sc.get_current_page()

# Load data based on user inputs
try:
    # Load data
    if inputs.get("use_file", False):
        try:
            if inputs.get("uploaded_file") is not None:
                # Load from uploaded file
                data = dl.load_stock_data_from_uploaded_file(inputs["uploaded_file"])
                st.success(f"Data loaded from uploaded file: {inputs['uploaded_file'].name}")
            elif inputs.get("file_path"):
                # Load from file path
                data = dl.load_stock_data_from_file(inputs["file_path"])
                st.success(f"Data loaded from {inputs['file_path']}")
            else:
                st.error("No file provided. Please upload a file or specify a file path.")
                # Fallback to API data
                data = dl.load_stock_data(
                    inputs["symbols"], 
                    inputs["start_date"], 
                    inputs["end_date"], 
                    inputs["data_source"],
                    table_style=inputs["table_style"]
                )
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
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
        
        # Store data in session state for download button
        if data is not None and not data.empty:
            st.session_state.data = data
    
    if data is not None and not data.empty:
        # Display different pages based on navigation
        if current_page == "Home":
            # Display the data
            sc.display_data(data)
            
            # Visualization section
            st.header("Stock Visualization")
            
            # Generate and display chart when button is clicked
            if st.button("Generate Chart"):
                with st.spinner("Generating chart..."):
                    try:
                        # Generate chart based on user inputs
                        fig = viz.generate_candlestick_chart(
                            symbol=inputs["symbols"][0],
                            start_date=inputs["start_date"],
                            end_date=inputs["end_date"],
                            data_source=inputs["data_source"],
                            advanced_indicators=inputs["advanced_indicators"]
                        )
                        
                        # Display the chart
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Could not generate chart. Please check your inputs.")
                        
                        # Display statistics if data is available
                        if len(data) > 0:
                            st.header("Statistics")
                            
                            # For multiple symbols, display stats for each symbol in tabs
                            if len(inputs["symbols"]) > 1:
                                # Create tabs for each symbol
                                symbol_tabs = st.tabs(inputs["symbols"])
                                
                                # Display stats for each symbol in its tab
                                for i, symbol in enumerate(inputs["symbols"]):
                                    with symbol_tabs[i]:
                                        try:
                                            # Calculate statistics for the symbol
                                            stats = sa.calculate_statistics(data, symbol, dl.get_price_columns, inputs["table_style"])
                                            
                                            if stats:
                                                # Display statistics in columns
                                                cols = st.columns(len(stats))
                                                for j, (metric, value) in enumerate(stats.items()):
                                                    with cols[j]:
                                                        st.metric(
                                                            metric, 
                                                            f"{value:.2f}%" if metric != "Beta" else f"{value:.2f}"
                                                        )
                                            else:
                                                st.warning(f"Could not calculate statistics for {symbol}.")
                                        except Exception as e:
                                            st.error(f"Error calculating statistics for {symbol}: {str(e)}")
                                
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
                                # For a single symbol, display stats directly
                                try:
                                    # Calculate statistics for the symbol
                                    stats = sa.calculate_statistics(data, inputs["symbols"][0], dl.get_price_columns, inputs["table_style"])
                                    
                                    if stats:
                                        # Display statistics in columns
                                        cols = st.columns(len(stats))
                                        for i, (metric, value) in enumerate(stats.items()):
                                            with cols[i]:
                                                st.metric(
                                                    metric, 
                                                    f"{value:.2f}%" if metric != "Beta" else f"{value:.2f}"
                                                )
                                    else:
                                        st.warning(f"Could not calculate statistics for {inputs['symbols'][0]}.")
                                except Exception as e:
                                    st.error(f"Error calculating statistics: {str(e)}")
                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")
                        st.write(traceback.format_exc())
        
        elif current_page == "Modern Portfolio Theory":
            # Display the MPT page
            mpt.display_mpt_page(data, inputs)
    else:
        st.error("No data available. Please check your inputs and try again.")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write(traceback.format_exc())

if __name__ == "__main__":
    # This code won't run in Streamlit, but it's here for documentation
    # Run the app with: streamlit run streamlit_app.py
    pass
