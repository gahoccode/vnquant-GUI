"""
Visualization module for stock data charts and plots.
This module handles all visualization functions for the app.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import data_loader as dl

def generate_candlestick_chart(symbol, start_date, end_date, data_source, advanced_indicators):
    """
    Generate candlestick chart using Plotly directly
    
    Args:
        symbol: Stock symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_source: Source of data ('cafe' or 'VND')
        advanced_indicators: List of advanced indicators to show
        
    Returns:
        Plotly figure object
    """
    try:
        # Load the data
        data = dl.load_stock_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source
        )
        
        if data is None or data.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected parameters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Determine the column prefix based on data structure
        # If the column names start with the symbol (e.g., 'REE_open'), use that prefix
        # Otherwise, use no prefix
        prefix = f"{symbol}_" if any(col.startswith(f"{symbol}_") for col in data.columns) else ""
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data[f'{prefix}open'],
                high=data[f'{prefix}high'],
                low=data[f'{prefix}low'],
                close=data[f'{prefix}close'],
                name=symbol,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume trace
        volume_col = f'{prefix}volume_match' if f'{prefix}volume_match' in data.columns else f'{prefix}volume'
        
        # Create a color array for volume bars based on price change
        colors = ['green' if data[f'{prefix}close'][i] >= data[f'{prefix}open'][i] 
                 else 'red' for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[volume_col],
                name='Volume',
                marker=dict(
                    color=colors,
                    opacity=0.7
                )
            ),
            row=2, col=1
        )
        
        # Add advanced indicators if requested
        if advanced_indicators and len(advanced_indicators) > 0:
            # Calculate and add moving averages
            if 'MA' in advanced_indicators:
                # Add 20-day moving average
                ma20 = data[f'{prefix}close'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma20,
                        name='MA20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
                
                # Add 50-day moving average
                ma50 = data[f'{prefix}close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma50,
                        name='MA50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Calculate and add Bollinger Bands
            if 'BOLL' in advanced_indicators:
                # Calculate 20-day moving average
                ma20 = data[f'{prefix}close'].rolling(window=20).mean()
                # Calculate standard deviation
                std20 = data[f'{prefix}close'].rolling(window=20).std()
                
                # Upper Bollinger Band (MA20 + 2*std)
                upper_band = ma20 + 2 * std20
                # Lower Bollinger Band (MA20 - 2*std)
                lower_band = ma20 - 2 * std20
                
                # Add upper band
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=upper_band,
                        name='Upper BB',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
                    ),
                    row=1, col=1
                )
                
                # Add lower band
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=lower_band,
                        name='Lower BB',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.1)'
                    ),
                    row=1, col=1
                )
            
            # Calculate and add RSI (Relative Strength Index)
            if 'RSI' in advanced_indicators:
                # Create a new row for RSI
                fig = make_subplots(
                    rows=3, 
                    cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI (14)'),
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Re-add the candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data[f'{prefix}open'],
                        high=data[f'{prefix}high'],
                        low=data[f'{prefix}low'],
                        close=data[f'{prefix}close'],
                        name=symbol,
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ),
                    row=1, col=1
                )
                
                # Re-add the volume chart
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[volume_col],
                        name='Volume',
                        marker=dict(
                            color=colors,
                            opacity=0.7
                        )
                    ),
                    row=2, col=1
                )
                
                # Calculate RSI
                delta = data[f'{prefix}close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Add RSI trace
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rsi,
                        name='RSI (14)',
                        line=dict(color='purple', width=1)
                    ),
                    row=3, col=1
                )
                
                # Add RSI reference lines (30 and 70)
                fig.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=30,
                    x1=data.index[-1],
                    y1=30,
                    line=dict(color="red", width=1, dash="dash"),
                    row=3, col=1
                )
                
                fig.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=70,
                    x1=data.index[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                    row=3, col=1
                )
                
                # Add reference text
                fig.add_annotation(
                    x=data.index[0],
                    y=30,
                    text="30",
                    showarrow=False,
                    xshift=-30,
                    row=3, col=1
                )
                
                fig.add_annotation(
                    x=data.index[0],
                    y=70,
                    text="70",
                    showarrow=False,
                    xshift=-30,
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} from {start_date} to {end_date}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        if 'RSI' in advanced_indicators:
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        return fig
    except Exception as e:
        # Return an error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_returns_chart(returns_data, symbol=None):
    """
    Create a bar chart showing returns over different time periods
    
    Args:
        returns_data: Dictionary with return metrics
        symbol: Optional symbol name for the title
        
    Returns:
        Plotly figure object
    """
    if not returns_data:
        return None
        
    # Filter out None values
    filtered_returns = {k: v for k, v in returns_data.items() if v is not None}
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for each return period
    for period, value in filtered_returns.items():
        fig.add_trace(go.Bar(
            x=[period],
            y=[value],
            name=period,
            marker_color='#21918c' if value >= 0 else '#440154'  # Teal and dark purple from Viridis
        ))
    
    # Update layout
    title = "Returns Analysis" if symbol is None else f"{symbol} Returns Analysis"
    fig.update_layout(
        title=title,
        yaxis_title="Return (%)",
        showlegend=False,
        height=400
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(filtered_returns) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Add value labels on top of bars
    for i, (_, value) in enumerate(filtered_returns.items()):
        fig.add_annotation(
            x=i,
            y=value,
            text=f"{value:.2f}%",
            showarrow=False,
            yshift=10 if value >= 0 else -10
        )
    
    return fig

def create_portfolio_comparison_chart(portfolio_data, benchmark_data=None):
    """
    Create a comparison chart for portfolio performance vs benchmark
    
    Args:
        portfolio_data: Dictionary with portfolio performance metrics
        benchmark_data: Optional dictionary with benchmark performance metrics
        
    Returns:
        Plotly figure object
    """
    try:
        # Create the figure
        fig = go.Figure()
        
        # Add portfolio performance
        fig.add_trace(go.Scatter(
            x=portfolio_data['dates'],
            y=portfolio_data['values'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#21918c', width=2)  # Teal from Viridis
        ))
        
        # Add benchmark if provided
        if benchmark_data:
            fig.add_trace(go.Scatter(
                x=benchmark_data['dates'],
                y=benchmark_data['values'],
                mode='lines',
                name='Benchmark',
                line=dict(color='#3b528b', width=2, dash='dash')  # Purple from Viridis
            ))
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Legend',
            height=500,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        print(f"Error creating portfolio comparison chart: {str(e)}")
        return None


# =====================================================================
# MODERN PORTFOLIO THEORY VISUALIZATION FUNCTIONS
# =====================================================================

def create_efficient_frontier_chart(mpt_data):
    """
    Create an efficient frontier chart from Modern Portfolio Theory calculations
    
    Args:
        mpt_data: Dictionary with MPT metrics and optimal portfolios
        
    Returns:
        Plotly figure object
    """
    if not mpt_data or 'simulation_data' not in mpt_data:
        return None
    
    # Extract simulation data
    sim_data = mpt_data['simulation_data']
    
    # Extract optimal portfolios
    max_sharpe = mpt_data['max_sharpe']
    min_var = mpt_data['min_variance']
    max_ret = mpt_data['max_return']
    
    # Create the figure
    fig = go.Figure()
    
    # Simulated portfolios
    fig.add_trace(go.Scatter(
        x=sim_data['Risk'] * 100,  # Convert decimal to percentage
        y=sim_data['Returns'] * 100,  # Convert decimal to percentage
        mode='markers',
        marker=dict(
            size=6,
            color=sim_data['Sharpe'],
            colorscale='Viridis',
            colorbar=dict(
                title='Sharpe Ratio',
                titleside='right',
                titlefont=dict(size=12, color='black'),
                tickfont=dict(size=10, color='black')
            ),
            line=dict(width=0)
        ),
        name='Simulated Portfolios',
        hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>',
        showlegend=False  # Hide from legend
    ))
    
    # Efficient frontier line - sort by risk for a smooth line
    ef_data = sim_data.sort_values('Risk')
    
    # =====================================================================
    # EFFICIENT FRONTIER CALCULATION
    # =====================================================================
    # The efficient frontier represents the set of optimal portfolios that
    # offer the highest expected return for a defined level of risk.
    # We approximate it by finding the highest return portfolio for each risk level.
    # =====================================================================
    
    # Get the efficient frontier points (upper edge of the cloud)
    # For each risk level, get the highest return
    ef_points = []
    risk_levels = np.linspace(ef_data['Risk'].min(), ef_data['Risk'].max(), 100)
    
    for risk in risk_levels:
        # Find all portfolios with similar risk
        similar_risk = ef_data[(ef_data['Risk'] >= risk*0.99) & (ef_data['Risk'] <= risk*1.01)]
        if not similar_risk.empty:
            # Get the one with the highest return
            best = similar_risk.loc[similar_risk['Returns'].idxmax()]
            ef_points.append((best['Risk'], best['Returns']))
    
    ef_risk = [p[0] * 100 for p in ef_points]  # Convert decimal to percentage
    ef_return = [p[1] * 100 for p in ef_points]  # Convert decimal to percentage
    
    fig.add_trace(go.Scatter(
        x=ef_risk,
        y=ef_return,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Efficient Frontier',
        hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>',
        showlegend=True  # Changed from False to True
    ))
    
    # Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[min_var['risk'] * 100],  # Convert decimal to percentage
        y=[min_var['return'] * 100],  # Convert decimal to percentage
        mode='markers+text',
        marker=dict(
            size=10,
            color='#3b528b',  # Purple from Viridis
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        name='Min Variance Portfolio',
        text=['Min Variance'],
        textposition='top center',
        textfont=dict(
            color='#3b528b',  # Purple from Viridis
            size=12,
            family='Arial, sans-serif'
        ),
        hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{min_var['sharpe']:.2f}" + '<extra></extra>',
        showlegend=True  # Show in legend
    ))
    
    # Maximum Sharpe Ratio Portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe['risk'] * 100],  # Convert decimal to percentage
        y=[max_sharpe['return'] * 100],  # Convert decimal to percentage
        mode='markers+text',
        marker=dict(
            size=14,
            color='#21918c',  # Teal from Viridis
            symbol='star',
            line=dict(width=1, color='black')
        ),
        name='Max Sharpe Ratio Portfolio',
        text=['Max Sharpe'],
        textposition='top center',  
        textfont=dict(
            color='#21918c',  # Teal from Viridis
            size=12,
            family='Arial, sans-serif'
        ),
        hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{max_sharpe['sharpe']:.2f}" + '<extra></extra>',
        showlegend=True  # Show in legend
    ))
    
    # Maximum Return Portfolio
    # Check if Max Return is very close to Max Sharpe, if so, adjust the text position
    is_close = abs(max_ret['risk'] - max_sharpe['risk']) < 0.5 and abs(max_ret['return'] - max_sharpe['return']) < 0.5
    text_position = 'top right' if not is_close else 'bottom right'  
    
    fig.add_trace(go.Scatter(
        x=[max_ret['risk'] * 100],  # Convert decimal to percentage
        y=[max_ret['return'] * 100],  # Convert decimal to percentage
        mode='markers+text',
        marker=dict(
            size=10,
            color='#5ec962',  # Green from Viridis
            symbol='diamond',
            line=dict(width=1, color='black')
        ),
        name='Max Return Portfolio',
        text=['Max Return'],
        textposition=text_position,
        textfont=dict(
            color='#5ec962',  # Green from Viridis
            size=12,
            family='Arial, sans-serif'
        ),
        hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{max_ret['sharpe']:.2f}" + '<extra></extra>',
        showlegend=True  # Show in legend
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Efficient Frontier',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='black')
        },
        xaxis_title={
            'text': 'Volatility, or risk (standard deviation)',
            'font': dict(color='black', size=14)
        },
        yaxis_title={
            'text': 'Annual return',
            'font': dict(color='black', size=14)
        },
        legend=dict(
            orientation="v",
            yanchor="bottom",  
            y=0.02,  
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(family="Arial, sans-serif", size=12, color='black')
        ),
        height=500,
        width=None,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        template='plotly_white'
    )
    
    # Format axes as percentages
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        tickformat='.1%',
        tickfont=dict(color='black')
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        tickformat='.1%',
        tickfont=dict(color='black')
    )
    
    return fig


def create_optimal_portfolio_weights_chart(weights, title):
    """
    Create a pie chart showing the weights of an optimal portfolio
    
    Args:
        weights: Dictionary of weights for each symbol
        title: Title of the chart
        
    Returns:
        Plotly figure object
    """
    try:
        # Create pie chart
        labels = list(weights.keys())
        values = list(weights.values())
        
        # Format values as percentages
        text = [f"{v*100:.1f}%" for v in values]
        
        # Brighter Viridis colors for the pie chart (focusing on the brightest end of the palette)
        viridis_colors = ['#fde725', '#95d840', '#5ec962', '#21918c', '#26828e', '#31688e', '#3b528b', '#482878', '#440154', '#3e4989']
        # If we have more segments than colors, we'll cycle through them
        colors = [viridis_colors[i % len(viridis_colors)] for i in range(len(labels))]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            text=text,
            textinfo='label+text',
            hoverinfo='label+percent',
            marker=dict(
                colors=colors,  # Use Viridis colors
                line=dict(color='#000000', width=1)
            ),
            textfont=dict(color='black')
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'font': dict(color='white', size=16)
            },
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        print(f"Error creating portfolio weights chart: {str(e)}")
        return None


def create_stock_price_time_series(mpt_data, price_data):
    """
    Create a time series chart of stock prices with the optimal portfolio performance
    
    Args:
        mpt_data: Dictionary with MPT calculation results
        price_data: DataFrame with stock price data
        
    Returns:
        Plotly figure object
    """
    try:
        if price_data.empty:
            return None
            
        # Normalize prices to start at 100 for comparison
        normalized_data = price_data.copy()
        for col in normalized_data.columns:
            normalized_data[col] = 100 * (normalized_data[col] / normalized_data[col].iloc[0])
        
        # Create figure
        fig = go.Figure()
        
        # Add lines for each stock
        for symbol in normalized_data.columns:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[symbol],
                mode='lines',
                name=symbol,
                opacity=0.7,
                line=dict(width=1.5)
            ))
        
        # Add a line for the optimal portfolio (max sharpe ratio)
        # This is a hypothetical performance based on the optimal weights
        if 'max_sharpe' in mpt_data and 'weights' in mpt_data['max_sharpe']:
            weights = mpt_data['max_sharpe']['weights']
            portfolio_performance = pd.Series(0.0, index=normalized_data.index)
            
            for symbol, weight in weights.items():
                if symbol in normalized_data.columns:
                    portfolio_performance += normalized_data[symbol] * weight
            
            fig.add_trace(go.Scatter(
                x=portfolio_performance.index,
                y=portfolio_performance.values,
                mode='lines',
                name='Max Sharpe Portfolio',
                line=dict(color='#21918c', width=3, dash='solid')  # Teal from Viridis
            ))
        
        # Add a line for the minimum variance portfolio
        if 'min_variance' in mpt_data and 'weights' in mpt_data['min_variance']:
            weights = mpt_data['min_variance']['weights']
            min_var_performance = pd.Series(0.0, index=normalized_data.index)
            
            for symbol, weight in weights.items():
                if symbol in normalized_data.columns:
                    min_var_performance += normalized_data[symbol] * weight
            
            fig.add_trace(go.Scatter(
                x=min_var_performance.index,
                y=min_var_performance.values,
                mode='lines',
                name='Min Variance Portfolio',
                line=dict(color='#3b528b', width=3, dash='dot')  # Purple from Viridis
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Normalized Price Performance (Starting Value = 100)',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16, color='black')
            },
            xaxis_title={
                'text': 'Date',
                'font': dict(color='black', size=14)
            },
            yaxis_title={
                'text': 'Normalized Price',
                'font': dict(color='black', size=14)
            },
            legend_title={
                'text': 'Symbols',
                'font': dict(color='black', size=14)
            },
            legend=dict(
                font=dict(color='black', size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="lightgray",
                borderwidth=1
            ),
            height=500,
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickfont=dict(color='black')
        )
        
        # Update y-axis
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickfont=dict(color='black')
        )
        
        return fig
    except Exception as e:
        print(f"Error creating stock price time series chart: {str(e)}")
        return None


def create_mpt_comparison_chart(mpt_data):
    """
    Create a comparison chart for the three optimal portfolios
    
    Args:
        mpt_data: Dictionary with MPT calculation results
        
    Returns:
        Plotly figure object
    """
    try:
        # Extract data for the three optimal portfolios
        portfolios = {
            'Maximum Sharpe Ratio': mpt_data['max_sharpe'],
            'Minimum Variance': mpt_data['min_variance'],
            'Maximum Return': mpt_data['max_return']
        }
        
        # Create metrics for comparison
        metrics = ['return', 'risk', 'sharpe']
        metric_names = ['Expected Return (%)', 'Risk/Volatility (%)', 'Sharpe Ratio']
        
        # Create figure with subplots
        fig = make_subplots(rows=1, cols=3, subplot_titles=metric_names)
        
        # Viridis colorscale colors for each portfolio
        colors = {
            'Maximum Sharpe Ratio': '#21918c',  # Teal from Viridis
            'Minimum Variance': '#3b528b',      # Purple from Viridis
            'Maximum Return': '#5ec962'         # Green from Viridis
        }
        
        # Add bars for each metric
        for i, metric in enumerate(metrics):
            values = [portfolios[p][metric] * 100 for p in portfolios.keys()]  # Convert decimal to percentage
            portfolio_names = list(portfolios.keys())
            
            fig.add_trace(
                go.Bar(
                    x=portfolio_names,
                    y=values,
                    marker_color=[colors[p] for p in portfolio_names],
                    showlegend=False,
                    text=[f"{v:.2f}%" for v in values],
                    textposition='auto',
                    textfont=dict(color='black')
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Comparison of Optimal Portfolios',
                'font': {'color': 'black', 'size': 16}
            },
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            template='plotly_white',
            paper_bgcolor='white'
        )
        
        # Update subplot titles to black
        for i in fig['layout']['annotations']:
            i['font'] = dict(color='black', size=14)
        
        # Update axes text to black
        fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
        fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
        
        return fig
    except Exception as e:
        print(f"Error creating MPT comparison chart: {str(e)}")
        return None
