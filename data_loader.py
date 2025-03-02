"""
Data loading module for stock market data.
This module handles loading data from various sources.
"""

import pandas as pd
import vnquant.data as dt

def load_stock_data(symbols, start_date, end_date, data_source, minimal=True, table_style="prefix"):
    """
    Load stock data from VNQuant API
    
    Args:
        symbols: List of stock symbols or a single symbol string
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_source: Source of data ('cafe' or 'VND')
        minimal: Whether to load minimal data
        table_style: Style of table ('prefix' or 'suffix')
        
    Returns:
        DataFrame with stock data
    """
    # Check if symbols is a string or a list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    loader = dt.DataLoader(symbols, start_date, end_date, data_source=data_source, minimal=minimal, table_style=table_style)
    return loader.download()

def load_stock_data_from_file(file_path):
    """
    Load stock data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with stock data
    """
    data = pd.read_csv(file_path)
    # Convert date column to datetime and set as index
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    return data

def load_stock_data_from_uploaded_file(uploaded_file):
    """
    Load stock data from an uploaded CSV file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame with stock data
    """
    data = pd.read_csv(uploaded_file)
    # Convert date column to datetime and set as index
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    return data

def get_price_columns(data, symbol, table_style="prefix"):
    """
    Get close and adjust price columns for a symbol.
    According to rules, all stock performance calculations must use adjusted prices.
    
    Args:
        data: DataFrame with stock data
        symbol: Stock symbol
        table_style: Style of table ('prefix' or 'suffix')
        
    Returns:
        Tuple of (close_prices, adjust_prices) or (None, None) if not found
    """
    if table_style == "prefix":
        close_col = f"{symbol}_close"
        adjust_col = f"{symbol}_adjust"
    else:  # suffix
        close_col = f"close_{symbol}"
        adjust_col = f"adjust_{symbol}"
    
    # Check if columns exist
    if close_col in data.columns and adjust_col in data.columns:
        return data[close_col], data[adjust_col]
    
    # For single symbol data, try without prefix/suffix
    if 'close' in data.columns and 'adjust' in data.columns:
        return data['close'], data['adjust']
    
    return None, None
