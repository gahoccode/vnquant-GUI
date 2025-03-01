# VNQuant-GUI Usage Guide

## Overview
This document explains how to use the VNQuant-GUI library for Vietnamese stock market data visualization and analysis. VNQuant-GUI provides tools to fetch, analyze, and visualize stock data from Vietnamese exchanges with a graphical user interface.

## Installation
VNQuant-GUI can be installed directly from the GitHub source:

```bash
# Clone the repository
git clone https://github.com/gahoccode/vnquant-GUI.git

# Navigate to the vnquant-GUI directory
cd vnquant-GUI

# Install the package in development mode
python setup.py install
```

Alternatively, if you've already cloned the repository, you can simply run:

```bash
# From the vnquant-GUI directory
pip install -e .
```

This will install the package in development mode, allowing you to make changes to the code and have them immediately reflected without reinstalling.


## Basic Usage

### Importing the Library
```python
import vnquant.plot as pl
from datetime import datetime, timedelta
```

### Plotting Stock Data
The main function for plotting stock data is `vnquant_candle_stick()`. Here's how to use it:

```python
pl.vnquant_candle_stick(
    data='REE',                      # Stock symbol
    start_date='2021-08-11',         # Start date in 'YYYY-MM-DD' format
    end_date='2021-10-13',           # End date in 'YYYY-MM-DD' format
    data_source='CAFE',              # Data source ('CAFE' or 'VND')
    show_advanced=['volume', 'macd'] # Advanced indicators to display
)
```

### Function Parameters

- **data**: String or DataFrame
  - If string: Stock symbol (e.g., 'REE', 'VCB', 'TCB')
  - If DataFrame: Pre-loaded stock data with OHLCV columns

- **start_date**: String (format: 'YYYY-MM-DD')
  - The start date for fetching stock data

- **end_date**: String (format: 'YYYY-MM-DD')
  - The end date for fetching stock data

- **data_source**: String
  - 'CAFE': CafeF data source (recommended)
  - 'VND': VNDirect data source

- **show_advanced**: List of strings
  - Indicators to display. Options include:
    - 'volume': Trading volume
    - 'macd': Moving Average Convergence Divergence
    - 'rsi': Relative Strength Index

- **title**: String (optional)
  - Custom title for the chart

- **colors**: List of strings (optional)
  - Colors for up and down candles, default: ['blue', 'red']

- **width**: Integer (optional)
  - Width of the chart in pixels, default: 800

- **height**: Integer (optional)
  - Height of the chart in pixels, default: 600

## Modular Application Structure

The VNQuant-GUI Stock Analyzer application has been modularized to improve code organization, maintainability, and extensibility. The application is now divided into the following modules:

### 1. data_loader.py

This module handles all data loading operations:

```python
import data_loader as dl

# Load data from VNQuant API
data = dl.load_stock_data(
    symbols=["REE", "FMC"],
    start_date="2024-12-01",
    end_date="2025-03-01",
    data_source="cafe",
    table_style="prefix"
)

# Load data from a CSV file
data = dl.load_stock_data_from_file("path/to/data.csv")

# Get price columns for a specific symbol
close_prices, adjust_prices = dl.get_price_columns(data, "REE", table_style="prefix")
```

### 2. stock_analysis.py

This module contains all analysis and calculation functions:

```python
import stock_analysis as sa

# Calculate returns using adjusted prices
returns = sa.calculate_returns(adjust_prices)

# Calculate price statistics
stats = sa.calculate_price_statistics(close_prices, adjust_prices)

# Calculate portfolio performance
portfolio_stats = sa.calculate_portfolio_performance(
    data,
    symbols=["REE", "FMC"],
    price_columns_func=dl.get_price_columns,
    table_style="prefix",
    weights={"REE": 0.6, "FMC": 0.4}
)

# Calculate risk metrics
risk_metrics = sa.calculate_risk_metrics(adjust_prices)
```

### 3. visualization.py

This module handles all visualization functions:

```python
import visualization as viz

# Generate candlestick chart
fig = viz.generate_candlestick_chart(
    symbol="REE",
    start_date="2024-12-01",
    end_date="2025-03-01",
    data_source="cafe",
    advanced_indicators=["volume", "macd"]
)

# Create returns chart
returns_chart = viz.create_returns_chart(returns, symbol="REE")

# Create portfolio comparison chart
portfolio_chart = viz.create_portfolio_comparison_chart(portfolio_stats)
```

### 4. streamlit_components.py

This module contains all UI components for the Streamlit app:

```python
import streamlit_components as sc

# Set up page configuration
sc.setup_page_config()

# Display app header
sc.display_header()

# Get user inputs from sidebar
inputs = sc.sidebar_inputs()

# Display data
sc.display_data(data)

# Display price statistics
sc.display_price_statistics(stats)

# Display returns
sc.display_returns(returns)

# Display chart
sc.display_chart(fig)
```

### 5. streamlit_app.py

This is the main application file that connects all the modules. Run it with:

```bash
streamlit run streamlit_app.py
```

## Important Notes About the Modular Structure

1. **Separation of Concerns**: Each module has a specific responsibility, making the code easier to maintain and extend.
2. **Adjusted Prices**: All stock performance calculations use adjusted prices as required by the rules.
3. **Data Loading**: The `data_loader.py` module supports loading data from both the VNQuant API and CSV files.
4. **Visualization**: The `visualization.py` module uses the VNQuant library's `vnquant_candle_stick` function for candlestick charts and Plotly for other visualizations.
5. **Portfolio Analysis**: The application supports portfolio performance analysis with custom weights for multiple stocks.

## Examples

### Basic Stock Chart
```python
pl.vnquant_candle_stick(
    data='REE', 
    start_date='2021-08-11', 
    end_date='2021-10-13', 
    data_source='CAFE'
)
```

### Chart with Volume Indicator
```python
pl.vnquant_candle_stick(
    data='REE', 
    show_advanced=['volume'], 
    start_date='2023-10-11', 
    end_date='2023-10-13', 
    data_source='CAFE'
)
```

### Chart with Multiple Indicators
```python
pl.vnquant_candle_stick(
    data='REE', 
    show_advanced=['volume', 'macd', 'rsi'], 
    start_date='2023-04-01', 
    end_date='2023-10-13', 
    data_source='CAFE'
)
```

### Custom Title and Styling
```python
pl.vnquant_candle_stick(
    data='REE', 
    title='REE Stock Performance', 
    show_advanced=['volume', 'macd'], 
    start_date='2023-04-01', 
    end_date='2023-10-13', 
    data_source='CAFE',
    colors=['green', 'red'],
    width=1200,
    height=800
)
```

### Using Dynamic Dates
If you need to use dynamic dates (e.g., last 30 days), convert datetime objects to strings:

```python
from datetime import datetime, timedelta

today = datetime.now().strftime('%Y-%m-%d')
one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

pl.vnquant_candle_stick(
    data='REE', 
    start_date=one_month_ago, 
    end_date=today, 
    data_source='CAFE'
)
```

### Using Pre-loaded Data
You can also use a pandas DataFrame with pre-loaded data:

```python
import pandas as pd

df = pd.read_csv('path/to/stock_data.csv')
pl.vnquant_candle_stick(
    data=df, 
    show_advanced=['volume', 'macd', 'rsi']
)
```

## Important Notes

1. Always use string format ('YYYY-MM-DD') for dates, not datetime objects directly.
2. The 'CAFE' data source is generally more reliable than 'VND'.
3. Make sure your DataFrame has the correct column structure if using pre-loaded data.
4. The library requires an internet connection to fetch data from the sources.

## Troubleshooting

- If you encounter connection timeouts with 'VND' data source, try switching to 'CAFE'.
- If you get date format errors, ensure you're using string dates in 'YYYY-MM-DD' format.
- For any other issues, check the VNQuant-GUI documentation or report issues on the GitHub repository.
