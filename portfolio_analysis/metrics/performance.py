"""
Performance metrics calculations for financial data.
"""

from typing import Union

import numpy as np
import pandas as pd

from portfolio_analysis.constants import (
    DAYS_PER_YEAR,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)


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

    TRADING_DAYS = TRADING_DAYS_PER_YEAR

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
        years = (data.index[-1] - data.index[0]).days / DAYS_PER_YEAR

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
        data: Union[pd.Series, pd.DataFrame],
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
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
        data: Union[pd.Series, pd.DataFrame],
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
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

        # Downside deviation (only negative returns, positive returns set to 0)
        # Using .where() for both Series and DataFrame ensures consistent behavior:
        # positive returns are replaced with 0, preserving the sample size
        downside_returns = returns.where(returns < 0, 0)

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
        data: Union[pd.Series, pd.DataFrame],
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
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

    @staticmethod
    def calculate_cvar(
        data: Union[pd.Series, pd.DataFrame],
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> Union[float, pd.Series]:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

        CVaR represents the expected loss given that the loss exceeds VaR.
        It is a more conservative risk measure than VaR as it considers
        the tail of the distribution.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index
        confidence_level : float, default 0.95
            Confidence level (e.g., 0.95 for 95%)

        Returns
        -------
        float or pd.Series
            CVaR value (typically negative)
        """
        returns = data.pct_change().dropna()
        var = PerformanceMetrics.calculate_var(data, confidence_level)

        if isinstance(returns, pd.DataFrame):
            cvar_values = {}
            for col in returns.columns:
                col_returns = returns[col]
                col_var = var[col]
                cvar_values[col] = col_returns[col_returns <= col_var].mean()
            return pd.Series(cvar_values)
        else:
            return returns[returns <= var].mean()

    @staticmethod
    def calculate_omega_ratio(
        data: Union[pd.Series, pd.DataFrame],
        threshold: float = 0.0,
    ) -> Union[float, pd.Series]:
        """
        Calculate Omega ratio.

        The Omega ratio compares the probability-weighted gains above a threshold
        to the probability-weighted losses below it. Higher values indicate better
        risk-adjusted performance.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index
        threshold : float, default 0.0
            Daily return threshold (0.0 for break-even)

        Returns
        -------
        float or pd.Series
            Omega ratio(s)
        """
        returns = data.pct_change().dropna()

        if isinstance(returns, pd.DataFrame):
            omega_values = {}
            for col in returns.columns:
                col_returns = returns[col]
                gains = col_returns[col_returns > threshold] - threshold
                losses = threshold - col_returns[col_returns <= threshold]
                if losses.sum() == 0:
                    omega_values[col] = np.inf
                else:
                    omega_values[col] = gains.sum() / losses.sum()
            return pd.Series(omega_values)
        else:
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            if losses.sum() == 0:
                return np.inf
            return gains.sum() / losses.sum()

    @staticmethod
    def calculate_ulcer_index(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Ulcer Index.

        The Ulcer Index measures downside volatility based on drawdowns.
        It penalizes deep and prolonged drawdowns more heavily than
        standard deviation. Lower values indicate less risk.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Ulcer Index value(s)
        """
        cumulative_returns = (1 + data.pct_change()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1

        # Ulcer Index is the quadratic mean of drawdowns
        ulcer_index = np.sqrt((drawdown**2).mean())
        return ulcer_index

    @staticmethod
    def calculate_recovery_factor(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Recovery Factor.

        Recovery Factor = Total Return / |Max Drawdown|

        A higher recovery factor indicates the portfolio generates more
        return per unit of maximum drawdown risk.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Recovery factor(s)
        """
        total_return = (data.iloc[-1] / data.iloc[0]) - 1
        max_dd = PerformanceMetrics.calculate_max_drawdown(data)

        return total_return / abs(max_dd)

    @staticmethod
    def calculate_win_rate(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Win Rate (percentage of positive return periods).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Win rate as a decimal (e.g., 0.55 for 55% win rate)
        """
        returns = data.pct_change().dropna()
        win_rate = (returns > 0).mean()
        return win_rate

    @staticmethod
    def calculate_profit_factor(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Profit Factor.

        Profit Factor = Sum of Gains / |Sum of Losses|

        A value greater than 1 indicates profitable trading.
        Higher values indicate more profitable strategies.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Profit factor(s)
        """
        returns = data.pct_change().dropna()

        if isinstance(returns, pd.DataFrame):
            profit_factors = {}
            for col in returns.columns:
                col_returns = returns[col]
                gains = col_returns[col_returns > 0].sum()
                losses = abs(col_returns[col_returns < 0].sum())
                if losses == 0:
                    profit_factors[col] = np.inf if gains > 0 else 0.0
                else:
                    profit_factors[col] = gains / losses
            return pd.Series(profit_factors)
        else:
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            if losses == 0:
                return np.inf if gains > 0 else 0.0
            return gains / losses

    @staticmethod
    def calculate_payoff_ratio(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Calculate Payoff Ratio (Average Win / Average Loss).

        Also known as the Risk/Reward ratio. Higher values indicate
        that winning trades are larger than losing trades on average.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Price data with datetime index

        Returns
        -------
        float or pd.Series
            Payoff ratio(s)
        """
        returns = data.pct_change().dropna()

        if isinstance(returns, pd.DataFrame):
            payoff_ratios = {}
            for col in returns.columns:
                col_returns = returns[col]
                avg_win = col_returns[col_returns > 0].mean()
                avg_loss = abs(col_returns[col_returns < 0].mean())
                if avg_loss == 0 or np.isnan(avg_loss):
                    payoff_ratios[col] = np.inf if avg_win > 0 else 0.0
                elif np.isnan(avg_win):
                    payoff_ratios[col] = 0.0
                else:
                    payoff_ratios[col] = avg_win / avg_loss
            return pd.Series(payoff_ratios)
        else:
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            if avg_loss == 0 or np.isnan(avg_loss):
                return np.inf if avg_win > 0 else 0.0
            if np.isnan(avg_win):
                return 0.0
            return avg_win / avg_loss

    @staticmethod
    def calculate_herfindahl_index(weights: list[float]) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration.

        HHI = sum(w_i^2) for all weights
        - HHI = 1.0 means full concentration in one asset
        - HHI = 1/n means equal weighting across n assets

        Parameters
        ----------
        weights : list of float
            Portfolio weights

        Returns
        -------
        float
            HHI value between 0 and 1
        """
        weights = np.array(weights)
        return float(np.sum(weights**2))

    @staticmethod
    def calculate_effective_n(weights: list[float]) -> float:
        """
        Calculate Effective N (effective number of assets).

        Effective N = 1 / HHI

        Represents how many equal-weighted assets would produce
        the same concentration level.

        Parameters
        ----------
        weights : list of float
            Portfolio weights

        Returns
        -------
        float
            Effective number of assets
        """
        hhi = PerformanceMetrics.calculate_herfindahl_index(weights)
        if hhi == 0:
            return float("inf")
        return 1.0 / hhi
