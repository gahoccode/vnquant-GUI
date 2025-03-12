import unittest
import pandas as pd
import numpy as np
from stock_analysis import calculate_returns
from mpt_analysis import calculate_mpt_portfolio

class TestReturns(unittest.TestCase):
    def setUp(self):
        # Create sample price data with known returns
        dates = pd.date_range(start='2025-01-01', end='2025-01-10')
        self.prices = pd.Series([100, 102, 99, 101, 103, 102, 104, 105, 103, 106], index=dates)
        
        # Create sample portfolio data
        self.portfolio_data = pd.DataFrame({
            'Stock1': [100, 102, 99, 101, 103],
            'Stock2': [50, 51, 49, 52, 53]
        }, index=pd.date_range(start='2025-01-01', end='2025-01-05'))

    def test_calculate_returns_ordering(self):
        # Test with reversed data
        reversed_prices = self.prices.iloc[::-1]
        returns = calculate_returns(reversed_prices)
        
        # Calculate expected returns from properly ordered data
        expected_returns = calculate_returns(self.prices)
        
        # Compare results
        self.assertAlmostEqual(returns['Daily Avg'], expected_returns['Daily Avg'], places=4)
        self.assertAlmostEqual(returns['Weekly'], expected_returns['Weekly'], places=4)
        self.assertAlmostEqual(returns['Total'], expected_returns['Total'], places=4)

    def test_mpt_portfolio_ordering(self):
        # Create reversed portfolio data
        reversed_data = self.portfolio_data.iloc[::-1]
        
        def mock_price_columns(data, symbol, style):
            return None, data[symbol]
        
        # Calculate MPT metrics with both ordered and reversed data
        mpt_ordered = calculate_mpt_portfolio(
            self.portfolio_data, 
            ['Stock1', 'Stock2'], 
            mock_price_columns,
            num_port=100
        )
        
        mpt_reversed = calculate_mpt_portfolio(
            reversed_data,
            ['Stock1', 'Stock2'],
            mock_price_columns,
            num_port=100
        )
        
        # Compare results
        self.assertAlmostEqual(
            mpt_ordered['max_sharpe']['return'],
            mpt_reversed['max_sharpe']['return'],
            places=4
        )
        self.assertAlmostEqual(
            mpt_ordered['min_variance']['risk'],
            mpt_reversed['min_variance']['risk'],
            places=4
        )

if __name__ == '__main__':
    unittest.main()
