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
        end_date: Union[str, datetime],
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
        raw_data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=progress
        )

        # Handle yfinance column format changes across versions
        # yfinance >= 0.2.40: 'Close' is adjusted, MultiIndex is (Price, Ticker)
        # yfinance < 0.2.40: 'Adj Close' exists, MultiIndex is (Price, Ticker)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multi-ticker download with MultiIndex columns
            price_types = raw_data.columns.get_level_values(0).unique()
            if "Adj Close" in price_types:
                data = raw_data["Adj Close"]
            elif "Close" in price_types:
                data = raw_data["Close"]
            else:
                raise ValueError(
                    f"No Close or Adj Close column found. Available: {price_types.tolist()}"
                )
        else:
            # Single ticker or flat columns
            if "Adj Close" in raw_data.columns:
                data = raw_data["Adj Close"]
            elif "Close" in raw_data.columns:
                data = raw_data["Close"]
            else:
                raise ValueError(
                    f"No Close or Adj Close column found. Available: {raw_data.columns.tolist()}"
                )

        # Handle single ticker case - ensure DataFrame format
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.tickers[0])

        return data

    def fetch_returns(
        self, frequency: str = "daily", progress: bool = True
    ) -> pd.DataFrame:
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

        if frequency == "daily":
            returns = data.pct_change().dropna()
        elif frequency == "weekly":
            returns = data.resample("W").last().pct_change().dropna()
        elif frequency == "monthly":
            returns = data.resample("M").last().pct_change().dropna()
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
