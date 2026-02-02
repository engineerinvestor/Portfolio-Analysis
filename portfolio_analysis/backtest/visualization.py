"""
Visualization tools for backtest results.

This module provides plotting functions for analyzing backtest results
including equity curves, drawdowns, and rolling metrics.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_analysis.backtest.engine import BacktestResult
from portfolio_analysis.backtest.metrics import BacktestMetrics


class BacktestVisualization:
    """
    Visualization tools for backtest results.

    Provides methods for plotting equity curves, drawdowns, rolling metrics,
    and comparative analysis of multiple strategies.

    Parameters
    ----------
    result : BacktestResult
        Backtest result to visualize

    Examples
    --------
    >>> result = engine.run()
    >>> viz = BacktestVisualization(result)
    >>> viz.plot_equity_curve()
    >>> viz.plot_drawdown()
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self.metrics = BacktestMetrics(result.portfolio_value, result.trades)

    def plot_equity_curve(
        self,
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (12, 6),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.

        Parameters
        ----------
        benchmark : pd.Series, optional
            Benchmark portfolio value to compare against
        figsize : tuple, default (12, 6)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot strategy equity curve
        ax.plot(
            self.result.portfolio_value.index,
            self.result.portfolio_value.values,
            label=self.result.strategy_name,
            linewidth=2,
        )

        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize to same starting value
            normalized_benchmark = (
                benchmark / benchmark.iloc[0] *
                self.result.portfolio_value.iloc[0]
            )
            ax.plot(
                normalized_benchmark.index,
                normalized_benchmark.values,
                label="Benchmark",
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

        # Add metrics annotation
        metrics = self.result.metrics
        textstr = (
            f"CAGR: {metrics.get('cagr', 0):.1%}\n"
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Max DD: {metrics.get('max_drawdown', 0):.1%}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_title(f"Equity Curve: {self.result.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_drawdown(
        self,
        figsize: tuple = (12, 5),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot drawdown over time.

        Parameters
        ----------
        figsize : tuple, default (12, 5)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        drawdown = self.metrics.calculate_drawdown_series()

        ax.fill_between(
            drawdown.index,
            drawdown.values * 100,
            0,
            alpha=0.5,
            color="red",
        )
        ax.plot(
            drawdown.index,
            drawdown.values * 100,
            color="darkred",
            linewidth=1,
        )

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()
        ax.scatter(
            [max_dd_idx], [max_dd * 100],
            color="black", s=100, zorder=5,
            label=f"Max DD: {max_dd:.1%}"
        )

        ax.set_title(f"Drawdown: {self.result.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_rolling_sharpe(
        self,
        window: int = 252,
        figsize: tuple = (12, 5),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot rolling Sharpe ratio.

        Parameters
        ----------
        window : int, default 252
            Rolling window in trading days
        figsize : tuple, default (12, 5)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        rolling_sharpe = self.metrics.calculate_rolling_sharpe(window=window)

        ax.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            linewidth=1.5,
            color="blue",
        )
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7)

        # Add average line
        avg_sharpe = rolling_sharpe.mean()
        ax.axhline(
            y=avg_sharpe, color="green", linestyle="--", linewidth=1,
            label=f"Average: {avg_sharpe:.2f}"
        )

        ax.set_title(f"Rolling {window}-Day Sharpe Ratio: {self.result.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_monthly_returns(
        self,
        figsize: tuple = (14, 8),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap.

        Parameters
        ----------
        figsize : tuple, default (14, 8)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        monthly = self.metrics.calculate_monthly_returns()

        # Create month-year matrix
        returns_by_month = monthly.groupby(
            [monthly.index.year, monthly.index.month]
        ).first().unstack()

        # Plot heatmap
        im = ax.imshow(
            returns_by_month.values * 100,
            cmap="RdYlGn",
            aspect="auto",
            vmin=-10,
            vmax=10,
        )

        # Labels
        ax.set_xticks(range(12))
        ax.set_xticklabels([
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ])
        ax.set_yticks(range(len(returns_by_month.index)))
        ax.set_yticklabels(returns_by_month.index)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Return (%)")

        # Add text annotations
        for i in range(len(returns_by_month.index)):
            for j in range(12):
                if j < returns_by_month.shape[1]:
                    val = returns_by_month.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val*100:.1f}",
                            ha="center", va="center",
                            color="white" if abs(val) > 0.05 else "black",
                            fontsize=8,
                        )

        ax.set_title(f"Monthly Returns (%): {self.result.strategy_name}")
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_weights_over_time(
        self,
        figsize: tuple = (12, 6),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot portfolio weights over time as stacked area chart.

        Parameters
        ----------
        figsize : tuple, default (12, 6)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        weights = self.result.weights_history * 100

        ax.stackplot(
            weights.index,
            [weights[col] for col in weights.columns],
            labels=weights.columns,
            alpha=0.8,
        )

        ax.set_title(f"Portfolio Weights Over Time: {self.result.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_summary(
        self,
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (15, 12),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot comprehensive summary with multiple charts.

        Parameters
        ----------
        benchmark : pd.Series, optional
            Benchmark to compare against
        figsize : tuple, default (15, 12)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)

        # Equity curve (top)
        ax1 = fig.add_subplot(3, 2, (1, 2))
        ax1.plot(
            self.result.portfolio_value.index,
            self.result.portfolio_value.values,
            label=self.result.strategy_name,
            linewidth=2,
        )
        if benchmark is not None:
            normalized = benchmark / benchmark.iloc[0] * self.result.portfolio_value.iloc[0]
            ax1.plot(normalized.index, normalized.values, label="Benchmark", linewidth=2, linestyle="--")
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown (middle left)
        ax2 = fig.add_subplot(3, 2, 3)
        drawdown = self.metrics.calculate_drawdown_series()
        ax2.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.5, color="red")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Rolling volatility (middle right)
        ax3 = fig.add_subplot(3, 2, 4)
        rolling_vol = self.metrics.calculate_rolling_volatility(window=21)
        ax3.plot(rolling_vol.index, rolling_vol.values * 100, linewidth=1)
        ax3.set_title("21-Day Rolling Volatility (Annualized)")
        ax3.set_ylabel("Volatility (%)")
        ax3.grid(True, alpha=0.3)

        # Returns distribution (bottom left)
        ax4 = fig.add_subplot(3, 2, 5)
        ax4.hist(self.result.daily_returns * 100, bins=50, edgecolor="black", alpha=0.7)
        ax4.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax4.set_title("Daily Returns Distribution")
        ax4.set_xlabel("Return (%)")
        ax4.set_ylabel("Frequency")

        # Weights over time (bottom right)
        ax5 = fig.add_subplot(3, 2, 6)
        weights = self.result.weights_history * 100
        ax5.stackplot(
            weights.index,
            [weights[col] for col in weights.columns],
            labels=weights.columns,
            alpha=0.8,
        )
        ax5.set_title("Portfolio Weights")
        ax5.set_ylabel("Weight (%)")
        ax5.legend(loc="upper left", fontsize=8)
        ax5.set_ylim(0, 100)

        fig.suptitle(f"Backtest Summary: {self.result.strategy_name}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def compare_strategies(
        results: list[BacktestResult],
        figsize: tuple = (12, 6),
        show: bool = True,
    ) -> plt.Figure:
        """
        Compare multiple strategy equity curves.

        Parameters
        ----------
        results : list of BacktestResult
            Results to compare
        figsize : tuple, default (12, 6)
            Figure size
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        for result in results:
            # Normalize to 100
            normalized = result.portfolio_value / result.portfolio_value.iloc[0] * 100
            ax.plot(
                normalized.index,
                normalized.values,
                label=f"{result.strategy_name} ({result.metrics.get('cagr', 0):.1%} CAGR)",
                linewidth=2,
            )

        ax.set_title("Strategy Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value (Starting at 100)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig
