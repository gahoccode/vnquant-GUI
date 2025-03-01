"""
Visualization module for stock data charts and plots.
This module handles all visualization functions for the app.
"""

import plotly.graph_objects as go
from vnquant import plot as pl

def generate_candlestick_chart(symbol, start_date, end_date, data_source, advanced_indicators):
    """
    Generate candlestick chart using VNQuant
    
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
        # Call the vnquant function
        result = pl.vnquant_candle_stick(
            data=symbol,
            title=f'{symbol} from {start_date} to {end_date}',
            xlab='Date', 
            ylab='Price',
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            show_advanced=advanced_indicators
        )
        
        # If the result is already a plotly figure, return it
        if isinstance(result, go.Figure):
            return result
        # If it's a dictionary (which might be the case), convert it to a figure
        elif isinstance(result, dict):
            return go.Figure(result)
        else:
            # For other cases, we'll create a basic figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="Chart generation failed. Try different parameters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    except Exception as e:
        # Create an error figure
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
            marker_color='green' if value >= 0 else 'red'
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
    if not portfolio_data:
        return None
    
    # Create the figure
    fig = go.Figure()
    
    # Add portfolio data
    metrics = list(portfolio_data.keys())
    values = list(portfolio_data.values())
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        name="Portfolio",
        marker_color='blue'
    ))
    
    # Add benchmark data if provided
    if benchmark_data:
        benchmark_values = [benchmark_data.get(metric, 0) for metric in metrics]
        fig.add_trace(go.Bar(
            x=metrics,
            y=benchmark_values,
            name="Benchmark",
            marker_color='orange'
        ))
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance",
        barmode='group',
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig
