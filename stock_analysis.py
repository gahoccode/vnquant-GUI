"""
Backend module for stock data analysis and calculations.
This module handles performance calculations and metrics.
"""

import pandas as pd

def calculate_returns(prices):
    """
    Calculate daily, weekly, monthly, and total returns from price data.
    According to rules, all stock performance calculations must use adjusted prices.
    
    Args:
        prices: Series of adjusted prices
        
    Returns:
        Dictionary with return metrics
    """
    if len(prices) < 2:
        return None
    
    # Sort prices chronologically
    prices = prices.sort_index()
    
    # Calculate returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate average daily return
    avg_daily_return = daily_returns.mean() * 100
    
    # Calculate weekly return (last 5 trading days)
    if len(prices) >= 5:
        weekly_return = ((prices.iloc[-1] / prices.iloc[-5]) - 1) * 100
    else:
        weekly_return = None
    
    # Calculate monthly return (last 21 trading days)
    if len(prices) >= 21:
        monthly_return = ((prices.iloc[-1] / prices.iloc[-21]) - 1) * 100
    else:
        monthly_return = None
    
    # Calculate total return
    total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
    
    return {
        "Daily Avg": avg_daily_return,
        "Weekly": weekly_return,
        "Monthly": monthly_return,
        "Total": total_return
    }

def calculate_price_statistics(prices, adjusted_prices=None):
    """
    Calculate basic price statistics
    
    Args:
        prices: Series of regular prices
        adjusted_prices: Series of adjusted prices (required for performance calculations)
        
    Returns:
        Dictionary with price statistics
    """
    stats = {
        "Latest Price": float(prices.iloc[-1]),
        "Highest Price": float(prices.max()),
        "Lowest Price": float(prices.min()),
        "Average Price": float(prices.mean())
    }
    
    # Add adjusted price info if available
    if adjusted_prices is not None:
        stats["Latest Adj. Price"] = float(adjusted_prices.iloc[-1])
    
    return stats

def calculate_portfolio_performance(data, symbols, price_columns_func, table_style="prefix", weights=None):
    """
    Calculate portfolio performance using adjusted prices as required by rules.
    
    Args:
        data: DataFrame with stock data
        symbols: List of stock symbols
        price_columns_func: Function to get price columns
        table_style: Style of table ('prefix' or 'suffix')
        weights: Dictionary of symbol weights, if None equal weights are used
        
    Returns:
        Dictionary with portfolio performance metrics
    """
    if not symbols:
        return None
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
    
    # Get adjusted prices for each symbol
    symbol_prices = {}
    for symbol in symbols:
        _, adjust_prices = price_columns_func(data, symbol, table_style)
        if adjust_prices is not None:
            symbol_prices[symbol] = adjust_prices
    
    if not symbol_prices:
        return None
    
    # Calculate daily returns for each symbol
    symbol_returns = {}
    for symbol, prices in symbol_prices.items():
        symbol_returns[symbol] = prices.pct_change().dropna()
    
    # Calculate weighted portfolio returns
    portfolio_returns = pd.Series(0.0, index=next(iter(symbol_returns.values())).index)
    for symbol, returns in symbol_returns.items():
        portfolio_returns += returns * weights.get(symbol, 0)
    
    # Calculate portfolio metrics
    avg_daily_return = portfolio_returns.mean() * 100
    total_return = ((1 + portfolio_returns).prod() - 1) * 100
    
    # Calculate volatility (risk)
    volatility = portfolio_returns.std() * 100 * (252 ** 0.5)  # Annualized
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = (avg_daily_return * 252) / volatility if volatility > 0 else 0
    
    return {
        "Daily Avg Return": avg_daily_return,
        "Total Return": total_return,
        "Volatility (Annual)": volatility,
        "Sharpe Ratio": sharpe_ratio
    }

def calculate_risk_metrics(prices):
    """
    Calculate risk metrics for a stock or portfolio
    
    Args:
        prices: Series of adjusted prices
        
    Returns:
        Dictionary with risk metrics
    """
    if len(prices) < 30:  # Need sufficient data for meaningful risk metrics
        return None
    
    # Sort prices chronologically
    prices = prices.sort_index()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate volatility (annualized)
    volatility = returns.std() * 100 * (252 ** 0.5)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * 100 * (252 ** 0.5) if len(downside_returns) > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    return {
        "Volatility (Annual)": volatility,
        "Downside Deviation": downside_deviation,
        "Maximum Drawdown": max_drawdown
    }
