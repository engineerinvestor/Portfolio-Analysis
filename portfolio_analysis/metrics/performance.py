"""
Performance metrics calculations for financial data.
"""

import numpy as np
import pandas as pd
from typing import Union


class PerformanceMetrics:
    """
    Static methods for calculating various performance metrics.

    All methods work with both Series (single asset) and DataFrame (multiple assets).

    Examples
    --------
    >>> data = loader.fetch_data()
    >>> annual_return = PerformanceMetrics.calculate_annual_return(data)
    >>> sharpe = PerformanceMetrics.calculate_sharpe_ratio(data, risk_free_rate=0.02)
    """

    TRADING_DAYS = 252

    @staticmethod
    def calculate_annual_return(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate annualized return from price data.

        Uses year-end prices to calculate annual returns, then averages.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Annualized return(s)
        """
        annual_return = data.resample("Y").last().pct_change().mean()
        return annual_return

    @staticmethod
    def calculate_cagr(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
        """
        Calculate Compound Annual Growth Rate (CAGR).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            CAGR value(s)
        """
        first_value = data.iloc[0]
        last_value = data.iloc[-1]
        years = (data.index[-1] - data.index[0]).days / 365.25

        cagr = (last_value / first_value) ** (1 / years) - 1
        return cagr

    @staticmethod
    def calculate_annual_volatility(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate annualized volatility (standard deviation of returns).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Annualized volatility
        """
        daily_returns = data.pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(
            PerformanceMetrics.TRADING_DAYS
        )
        return annual_volatility

    @staticmethod
    def calculate_sharpe_ratio(
        data: Union[pd.Series, pd.DataFrame], risk_free_rate: float = 0.02
    ) -> Union[float, pd.Series]:
        """
        Calculate Sharpe ratio.

        Sharpe Ratio = (Return - Risk Free Rate) / Volatility

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        float or pd.Series
            Sharpe ratio(s)
        """
        annual_return = PerformanceMetrics.calculate_annual_return(data)
        annual_volatility = PerformanceMetrics.calculate_annual_volatility(data)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(
        data: Union[pd.Series, pd.DataFrame], risk_free_rate: float = 0.02
    ) -> Union[float, pd.Series]:
        """
        Calculate Sortino ratio (uses downside deviation instead of volatility).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        float or pd.Series
            Sortino ratio(s)
        """
        returns = data.pct_change().dropna()

        # Downside deviation (only negative returns)
        if isinstance(returns, pd.DataFrame):
            downside_returns = returns.where(returns < 0, 0)
        else:
            downside_returns = returns[returns < 0]

        downside_deviation = downside_returns.std() * np.sqrt(
            PerformanceMetrics.TRADING_DAYS
        )
        annual_return = PerformanceMetrics.calculate_annual_return(data)

        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
        return sortino_ratio

    @staticmethod
    def calculate_max_drawdown(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate maximum drawdown.

        Maximum drawdown is the largest peak-to-trough decline.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Maximum drawdown (negative value)
        """
        cumulative_returns = (1 + data.pct_change()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        return max_drawdown

    @staticmethod
    def calculate_var(
        data: Union[pd.Series, pd.DataFrame], confidence_level: float = 0.95
    ) -> Union[float, pd.Series]:
        """
        Calculate Value at Risk (VaR) using historical method.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index
        confidence_level : float, default 0.95
            Confidence level (e.g., 0.95 for 95%)

        Returns
        -------
        float or pd.Series
            VaR value (typically negative)
        """
        returns = data.pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100, axis=0)

        if isinstance(returns, pd.DataFrame):
            return pd.Series(var, index=returns.columns)
        return var

    @staticmethod
    def calculate_calmar_ratio(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Calmar ratio(s)
        """
        cagr = PerformanceMetrics.calculate_cagr(data)
        max_dd = PerformanceMetrics.calculate_max_drawdown(data)

        # Max drawdown is negative, so we negate it
        return cagr / abs(max_dd)
