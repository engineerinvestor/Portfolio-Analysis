"""
Data loading functionality for fetching financial data.
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional, Union
from datetime import datetime


class DataLoader:
    """
    Fetch and preprocess financial data from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to fetch
    start_date : str or datetime
        Start date for historical data (YYYY-MM-DD format)
    end_date : str or datetime
        End date for historical data (YYYY-MM-DD format)

    Examples
    --------
    >>> loader = DataLoader(['VTI', 'BND'], '2020-01-01', '2023-12-31')
    >>> data = loader.fetch_data()
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self, progress: bool = True) -> pd.DataFrame:
        """
        Fetch adjusted close prices for all tickers.

        Parameters
        ----------
        progress : bool, default True
            Show download progress bar

        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and tickers as columns
        """
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=progress
        )['Adj Close']

        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.tickers[0])

        return data

    def fetch_returns(self, frequency: str = 'daily', progress: bool = True) -> pd.DataFrame:
        """
        Fetch and calculate returns.

        Parameters
        ----------
        frequency : str, default 'daily'
            Return frequency: 'daily', 'weekly', or 'monthly'
        progress : bool, default True
            Show download progress bar

        Returns
        -------
        pd.DataFrame
            DataFrame of returns
        """
        data = self.fetch_data(progress=progress)

        if frequency == 'daily':
            returns = data.pct_change().dropna()
        elif frequency == 'weekly':
            returns = data.resample('W').last().pct_change().dropna()
        elif frequency == 'monthly':
            returns = data.resample('M').last().pct_change().dropna()
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

        return returns

    @staticmethod
    def get_ticker_info(ticker: str) -> dict:
        """
        Get information about a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        dict
            Ticker information from Yahoo Finance
        """
        return yf.Ticker(ticker).info
