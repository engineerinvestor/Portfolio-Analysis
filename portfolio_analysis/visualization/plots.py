"""
Portfolio visualization functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional


class PortfolioVisualization:
    """
    Static methods for visualizing portfolio performance.

    Examples
    --------
    >>> PortfolioVisualization.plot_performance(data)
    >>> PortfolioVisualization.plot_portfolio_return(data, weights)
    >>> PortfolioVisualization.plot_allocation_pie(weights, tickers)
    """

    @staticmethod
    def plot_performance(data: pd.DataFrame, figsize: tuple = (10, 6)) -> None:
        """
        Plot cumulative returns for all assets.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        figsize : tuple, default (10, 6)
            Figure size
        """
        cumulative_returns = (1 + data.pct_change()).cumprod()
        cumulative_returns.plot(figsize=figsize)
        plt.title('Cumulative Returns by Asset')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_portfolio_return(
        data: pd.DataFrame,
        weights: List[float],
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot cumulative portfolio return.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        weights : list of float
            Portfolio weights
        figsize : tuple, default (10, 6)
            Figure size
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        cumulative_portfolio_returns = (1 + weighted_returns).cumprod()

        plt.figure(figsize=figsize)
        plt.plot(cumulative_portfolio_returns, linewidth=2, color='blue')
        plt.title('Portfolio Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Growth of $1')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_allocation_pie(
        weights: List[float],
        tickers: List[str],
        figsize: tuple = (8, 8)
    ) -> None:
        """
        Plot portfolio allocation as a pie chart.

        Parameters
        ----------
        weights : list of float
            Portfolio weights
        tickers : list of str
            Ticker symbols
        figsize : tuple, default (8, 8)
            Figure size
        """
        # Filter out zero weights
        non_zero = [(t, w) for t, w in zip(tickers, weights) if w > 0.001]
        labels, sizes = zip(*non_zero) if non_zero else ([], [])

        plt.figure(figsize=figsize)
        plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired.colors
        )
        plt.title('Portfolio Allocation')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_drawdown(
        data: pd.DataFrame,
        weights: List[float],
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot portfolio drawdown over time.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        weights : list of float
            Portfolio weights
        figsize : tuple, default (12, 6)
            Figure size
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        cumulative = (1 + weighted_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1

        plt.figure(figsize=figsize)
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.5, color='red')
        plt.plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rolling_volatility(
        data: pd.DataFrame,
        weights: List[float],
        window: int = 21,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot rolling portfolio volatility.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        weights : list of float
            Portfolio weights
        window : int, default 21
            Rolling window in trading days
        figsize : tuple, default (12, 6)
            Figure size
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        rolling_vol = weighted_returns.rolling(window=window).std() * np.sqrt(252)

        plt.figure(figsize=figsize)
        plt.plot(rolling_vol.index, rolling_vol.values * 100, linewidth=1.5, color='blue')
        plt.title(f'{window}-Day Rolling Volatility (Annualized)')
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_returns_distribution(
        data: pd.DataFrame,
        weights: List[float],
        bins: int = 50,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot histogram of portfolio returns.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        weights : list of float
            Portfolio weights
        bins : int, default 50
            Number of histogram bins
        figsize : tuple, default (10, 6)
            Figure size
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)

        plt.figure(figsize=figsize)
        plt.hist(weighted_returns * 100, bins=bins, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.axvline(
            x=weighted_returns.mean() * 100,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {weighted_returns.mean()*100:.2f}%'
        )
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
