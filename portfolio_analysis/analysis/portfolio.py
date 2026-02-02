"""
Core portfolio analysis functionality.
"""

import numpy as np
import pandas as pd


class PortfolioAnalysis:
    """
    Analyze a weighted portfolio of assets.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data with datetime index
    weights : list of float
        Portfolio weights (must sum to 1.0)

    Examples
    --------
    >>> portfolio = PortfolioAnalysis(data, [0.6, 0.4])
    >>> print(f"Return: {portfolio.calculate_portfolio_return():.2%}")
    >>> print(f"Volatility: {portfolio.calculate_portfolio_volatility():.2%}")
    >>> print(f"Sharpe: {portfolio.calculate_portfolio_sharpe_ratio():.2f}")
    """

    TRADING_DAYS = 252

    def __init__(self, data: pd.DataFrame, weights: list[float]):
        self.data = data
        self.weights = np.array(weights)

        if len(weights) != len(data.columns):
            raise ValueError(
                f"Weights length ({len(weights)}) must match "
                f"number of assets ({len(data.columns)})"
            )

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")

    def calculate_portfolio_return(self) -> float:
        """
        Calculate annualized portfolio return.

        Returns
        -------
        float
            Annualized portfolio return
        """
        returns = self.data.pct_change().mean()
        portfolio_return = np.dot(self.weights, returns) * self.TRADING_DAYS
        return portfolio_return

    def calculate_portfolio_volatility(self) -> float:
        """
        Calculate annualized portfolio volatility.

        Uses the covariance matrix to account for asset correlations.

        Returns
        -------
        float
            Annualized portfolio volatility (standard deviation)
        """
        returns = self.data.pct_change()
        covariance_matrix = returns.cov() * self.TRADING_DAYS
        portfolio_volatility = np.sqrt(
            np.dot(self.weights.T, np.dot(covariance_matrix, self.weights))
        )
        return portfolio_volatility

    def calculate_portfolio_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate portfolio Sharpe ratio.

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        float
            Sharpe ratio
        """
        portfolio_return = self.calculate_portfolio_return()
        portfolio_volatility = self.calculate_portfolio_volatility()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return sharpe_ratio

    def calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate daily portfolio returns.

        Returns
        -------
        pd.Series
            Daily weighted portfolio returns
        """
        returns = self.data.pct_change().dropna()
        return returns.dot(self.weights)

    def calculate_cumulative_returns(self) -> pd.Series:
        """
        Calculate cumulative portfolio returns.

        Returns
        -------
        pd.Series
            Cumulative returns (growth of $1)
        """
        returns = self.calculate_portfolio_returns()
        return (1 + returns).cumprod()

    def calculate_max_drawdown(self) -> float:
        """
        Calculate portfolio maximum drawdown.

        Returns
        -------
        float
            Maximum drawdown (negative value)
        """
        cumulative = self.calculate_cumulative_returns()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown.min()

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate portfolio Sortino ratio.

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        float
            Sortino ratio
        """
        returns = self.calculate_portfolio_returns()
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.TRADING_DAYS)

        portfolio_return = self.calculate_portfolio_return()
        return (portfolio_return - risk_free_rate) / downside_deviation

    def get_summary(self, risk_free_rate: float = 0.02) -> dict:
        """
        Get all portfolio metrics as a dictionary.

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        dict
            Dictionary of portfolio metrics
        """
        return {
            "annual_return": self.calculate_portfolio_return(),
            "annual_volatility": self.calculate_portfolio_volatility(),
            "sharpe_ratio": self.calculate_portfolio_sharpe_ratio(risk_free_rate),
            "sortino_ratio": self.calculate_sortino_ratio(risk_free_rate),
            "max_drawdown": self.calculate_max_drawdown(),
        }

    def print_summary(self, risk_free_rate: float = 0.02) -> None:
        """Print a formatted summary of portfolio metrics."""
        summary = self.get_summary(risk_free_rate)

        print("\n" + "=" * 40)
        print("PORTFOLIO SUMMARY")
        print("=" * 40)
        print(f"Annual Return:     {summary['annual_return']*100:>10.2f}%")
        print(f"Annual Volatility: {summary['annual_volatility']*100:>10.2f}%")
        print(f"Sharpe Ratio:      {summary['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:     {summary['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:      {summary['max_drawdown']*100:>10.2f}%")
        print("=" * 40)
