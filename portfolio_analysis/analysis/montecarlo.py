"""
Monte Carlo simulation for portfolio projections.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_analysis.constants import (
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_PROJECTION_DAYS,
    WEIGHT_SUM_TOLERANCE,
)
from portfolio_analysis.exceptions import ValidationError


class MonteCarloSimulation:
    """
    Monte Carlo simulation for portfolio performance projection.

    Simulates multiple future paths for a portfolio based on historical
    return distributions, accounting for asset correlations.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data for portfolio assets
    weights : array-like
        Portfolio weights (must sum to 1.0)
    num_simulations : int, default 1000
        Number of simulation paths to generate
    time_horizon : int, default 252
        Number of trading days to simulate (252 = 1 year)
    initial_investment : float, default 10000
        Starting portfolio value

    Examples
    --------
    >>> mc = MonteCarloSimulation(data, weights, num_simulations=1000, time_horizon=252)
    >>> mc.simulate()
    >>> mc.print_summary()
    >>> mc.plot_simulation()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        weights: list[float],
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        time_horizon: int = DEFAULT_PROJECTION_DAYS,
        initial_investment: float = 10000,
    ):
        self.data = data
        self.weights = np.array(weights)
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.initial_investment = initial_investment

        # Validate weights
        if len(self.weights) != len(data.columns):
            raise ValidationError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of assets ({len(data.columns)})"
            )
        if not np.isclose(sum(self.weights), 1.0, atol=WEIGHT_SUM_TOLERANCE):
            raise ValidationError(
                f"Weights must sum to 1.0, got {sum(self.weights):.6f}"
            )

        self._results = None

    def simulate(self) -> np.ndarray:
        """
        Run Monte Carlo simulation.

        Returns
        -------
        np.ndarray
            Array of shape (num_simulations, time_horizon) containing
            portfolio values for each simulation path over time.
        """
        returns = self.data.pct_change().dropna()
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        results = np.zeros((self.num_simulations, self.time_horizon))

        for i in range(self.num_simulations):
            # Generate correlated random returns for all assets
            sim_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, self.time_horizon
            )

            # Calculate weighted portfolio returns at each time step
            portfolio_returns = sim_returns @ self.weights

            # Track cumulative portfolio value over time
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            results[i, :] = self.initial_investment * cumulative_returns

        self._results = results
        return results

    def get_statistics(self, percentiles: list[int] = [5, 25, 50, 75, 95]) -> dict:
        """
        Calculate statistics across all simulation paths.

        Parameters
        ----------
        percentiles : list of int
            Percentiles to calculate

        Returns
        -------
        dict
            Dictionary containing percentiles, mean, std, and final value statistics
        """
        if self._results is None:
            self.simulate()

        results = self._results
        final_values = results[:, -1]

        return {
            "percentiles": {p: np.percentile(results, p, axis=0) for p in percentiles},
            "mean": np.mean(results, axis=0),
            "std": np.std(results, axis=0),
            "final_values": {
                "mean": np.mean(final_values),
                "median": np.median(final_values),
                "std": np.std(final_values),
                "min": np.min(final_values),
                "max": np.max(final_values),
                "percentile_5": np.percentile(final_values, 5),
                "percentile_95": np.percentile(final_values, 95),
                "prob_loss": np.mean(final_values < self.initial_investment) * 100,
            },
        }

    def plot_simulation(
        self,
        show_percentiles: bool = True,
        num_paths: int = 100,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """
        Plot Monte Carlo simulation results with percentile bands.

        Parameters
        ----------
        show_percentiles : bool, default True
            Whether to show percentile bands
        num_paths : int, default 100
            Number of individual paths to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        show : bool, default True
            Whether to display the plot. Set to False for automated/server contexts.
            Only applies when ax is None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if self._results is None:
            self.simulate()

        results = self._results
        stats = self.get_statistics()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
            created_fig = True

        # Plot a subset of individual paths
        paths_to_plot = min(num_paths, self.num_simulations)
        for i in range(paths_to_plot):
            ax.plot(results[i, :], color="lightblue", alpha=0.3, linewidth=0.5)

        if show_percentiles:
            days = np.arange(self.time_horizon)

            # Plot percentile bands
            ax.fill_between(
                days,
                stats["percentiles"][5],
                stats["percentiles"][95],
                color="blue",
                alpha=0.2,
                label="5th-95th percentile",
            )
            ax.fill_between(
                days,
                stats["percentiles"][25],
                stats["percentiles"][75],
                color="blue",
                alpha=0.3,
                label="25th-75th percentile",
            )

            # Plot median
            ax.plot(
                stats["percentiles"][50],
                color="darkblue",
                linewidth=2,
                label="Median (50th percentile)",
            )

        # Plot initial investment line
        ax.axhline(
            y=self.initial_investment,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Initial: ${self.initial_investment:,.0f}",
        )

        ax.set_title(
            f"Monte Carlo Simulation ({self.num_simulations:,} paths, {self.time_horizon} days)"
        )
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add summary statistics text box
        final_stats = stats["final_values"]
        textstr = "Final Value Statistics:\n"
        textstr += f"Median: ${final_stats['median']:,.0f}\n"
        textstr += f"5th %: ${final_stats['percentile_5']:,.0f}\n"
        textstr += f"95th %: ${final_stats['percentile_95']:,.0f}\n"
        textstr += f"Prob. of Loss: {final_stats['prob_loss']:.1f}%"

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.98,
            0.02,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        if created_fig and show:
            plt.show()

        return ax

    def print_summary(self) -> None:
        """Print a summary of simulation results."""
        if self._results is None:
            self.simulate()

        stats = self.get_statistics()
        final = stats["final_values"]

        print("\n" + "=" * 50)
        print("MONTE CARLO SIMULATION SUMMARY")
        print("=" * 50)
        print(f"Initial Investment: ${self.initial_investment:,.0f}")
        print(f"Time Horizon: {self.time_horizon} trading days")
        print(f"Number of Simulations: {self.num_simulations:,}")
        print("-" * 50)
        print("Final Portfolio Value Statistics:")
        print(f"  Mean:     ${final['mean']:,.0f}")
        print(f"  Median:   ${final['median']:,.0f}")
        print(f"  Std Dev:  ${final['std']:,.0f}")
        print(f"  Min:      ${final['min']:,.0f}")
        print(f"  Max:      ${final['max']:,.0f}")
        print("-" * 50)
        print("Percentile Projections:")
        print(f"  5th percentile:  ${final['percentile_5']:,.0f}")
        print(f"  95th percentile: ${final['percentile_95']:,.0f}")
        print("-" * 50)
        print(f"Probability of Loss: {final['prob_loss']:.1f}%")
        print("=" * 50)
