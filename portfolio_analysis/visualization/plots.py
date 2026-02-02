"""
Portfolio visualization functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_analysis.constants import TRADING_DAYS_PER_YEAR


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
    def plot_performance(
        data: pd.DataFrame, figsize: tuple = (10, 6), show: bool = True
    ) -> plt.Figure:
        """
        Plot cumulative returns for all assets.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with datetime index
        figsize : tuple, default (10, 6)
            Figure size
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        cumulative_returns = (1 + data.pct_change()).cumprod()
        cumulative_returns.plot(ax=ax)
        ax.set_title("Cumulative Returns by Asset")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_portfolio_return(
        data: pd.DataFrame,
        weights: list[float],
        figsize: tuple = (10, 6),
        show: bool = True,
    ) -> plt.Figure:
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
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        cumulative_portfolio_returns = (1 + weighted_returns).cumprod()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cumulative_portfolio_returns, linewidth=2, color="blue")
        ax.set_title("Portfolio Cumulative Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_allocation_pie(
        weights: list[float],
        tickers: list[str],
        figsize: tuple = (8, 8),
        show: bool = True,
    ) -> plt.Figure:
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
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Filter out zero weights
        non_zero = [(t, w) for t, w in zip(tickers, weights) if w > 0.001]
        labels, sizes = zip(*non_zero) if non_zero else ([], [])

        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Paired.colors,
        )
        ax.set_title("Portfolio Allocation")
        ax.axis("equal")
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_drawdown(
        data: pd.DataFrame,
        weights: list[float],
        figsize: tuple = (12, 6),
        show: bool = True,
    ) -> plt.Figure:
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
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        cumulative = (1 + weighted_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1

        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(
            drawdown.index, drawdown.values * 100, 0, alpha=0.5, color="red"
        )
        ax.plot(drawdown.index, drawdown.values * 100, color="darkred", linewidth=1)
        ax.set_title("Portfolio Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_rolling_volatility(
        data: pd.DataFrame,
        weights: list[float],
        window: int = 21,
        figsize: tuple = (12, 6),
        show: bool = True,
    ) -> plt.Figure:
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
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)
        rolling_vol = weighted_returns.rolling(window=window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            rolling_vol.index, rolling_vol.values * 100, linewidth=1.5, color="blue"
        )
        ax.set_title(f"{window}-Day Rolling Volatility (Annualized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_returns_distribution(
        data: pd.DataFrame,
        weights: list[float],
        bins: int = 50,
        figsize: tuple = (10, 6),
        show: bool = True,
    ) -> plt.Figure:
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
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        returns = data.pct_change().dropna()
        weighted_returns = returns.dot(weights)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(weighted_returns * 100, bins=bins, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax.axvline(
            x=weighted_returns.mean() * 100,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {weighted_returns.mean()*100:.2f}%",
        )
        ax.set_title("Daily Returns Distribution")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig
