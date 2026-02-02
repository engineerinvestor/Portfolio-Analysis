"""
Portfolio optimization functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Portfolio optimization using various strategies.

    Supports Maximum Sharpe Ratio, Minimum Volatility, Risk Parity,
    and Target Return optimization strategies.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data with datetime index
    risk_free_rate : float, default 0.02
        Annual risk-free rate

    Examples
    --------
    >>> optimizer = PortfolioOptimizer(data, risk_free_rate=0.02)
    >>> optimal = optimizer.optimize_max_sharpe()
    >>> print(f"Optimal weights: {optimal['weights']}")
    >>> optimizer.plot_efficient_frontier()
    """

    TRADING_DAYS = 252

    def __init__(self, data: pd.DataFrame, risk_free_rate: float = 0.02):
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(data.columns)
        self.tickers = list(data.columns)

        # Calculate returns and covariance
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * self.TRADING_DAYS
        self.cov_matrix = self.returns.cov() * self.TRADING_DAYS

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, self.mean_returns)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe for minimization."""
        return -self._portfolio_sharpe(weights)

    def optimize_max_sharpe(self, weight_bounds: Tuple[float, float] = (0, 1)) -> Dict:
        """
        Find the portfolio with maximum Sharpe ratio.

        Parameters
        ----------
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        dict
            Optimal weights, return, volatility, and Sharpe ratio
        """
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            self._neg_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
        }

    def optimize_min_volatility(
        self, weight_bounds: Tuple[float, float] = (0, 1)
    ) -> Dict:
        """
        Find the minimum volatility portfolio.

        Parameters
        ----------
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        dict
            Optimal weights, return, volatility, and Sharpe ratio
        """
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            self._portfolio_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
        }

    def optimize_target_return(
        self, target_return: float, weight_bounds: Tuple[float, float] = (0, 1)
    ) -> Dict:
        """
        Find minimum volatility portfolio for a target return.

        Parameters
        ----------
        target_return : float
            Target annual return
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        dict
            Optimal weights, return, volatility, and Sharpe ratio
        """
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: self._portfolio_return(x) - target_return},
        ]
        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            self._portfolio_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
        }

    def optimize_risk_parity(self) -> Dict:
        """
        Find the risk parity portfolio (equal risk contribution).

        Each asset contributes equally to total portfolio risk.

        Returns
        -------
        dict
            Optimal weights, return, volatility, and Sharpe ratio
        """

        def risk_contribution(weights):
            """Calculate risk contribution of each asset."""
            port_vol = self._portfolio_volatility(weights)
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            return risk_contrib

        def risk_parity_objective(weights):
            """Objective: minimize deviation from equal risk contribution."""
            rc = risk_contribution(weights)
            target_rc = np.mean(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 1) for _ in range(self.n_assets))  # Min 1% per asset
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
            "risk_contributions": dict(
                zip(self.tickers, risk_contribution(optimal_weights))
            ),
        }

    def generate_efficient_frontier(
        self, n_points: int = 50, weight_bounds: Tuple[float, float] = (0, 1)
    ) -> pd.DataFrame:
        """
        Generate points on the efficient frontier.

        Parameters
        ----------
        n_points : int, default 50
            Number of points to generate
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        pd.DataFrame
            DataFrame with return, volatility, and Sharpe for each point
        """
        # Find return range
        min_vol = self.optimize_min_volatility(weight_bounds)
        max_sharpe = self.optimize_max_sharpe(weight_bounds)

        min_ret = min_vol["return"]
        max_ret = max(self.mean_returns)

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            try:
                portfolio = self.optimize_target_return(target, weight_bounds)
                frontier.append(
                    {
                        "return": portfolio["return"],
                        "volatility": portfolio["volatility"],
                        "sharpe_ratio": portfolio["sharpe_ratio"],
                    }
                )
            except:
                continue

        return pd.DataFrame(frontier)

    def plot_efficient_frontier(
        self,
        n_points: int = 50,
        show_assets: bool = True,
        show_optimal: bool = True,
        weight_bounds: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        Plot the efficient frontier with optimal portfolios marked.

        Parameters
        ----------
        n_points : int, default 50
            Number of points on the frontier
        show_assets : bool, default True
            Show individual assets
        show_optimal : bool, default True
            Mark optimal portfolios (max Sharpe, min vol)
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset
        """
        frontier = self.generate_efficient_frontier(n_points, weight_bounds)

        plt.figure(figsize=(12, 8))

        # Plot efficient frontier
        plt.plot(
            frontier["volatility"] * 100,
            frontier["return"] * 100,
            "b-",
            linewidth=2,
            label="Efficient Frontier",
        )

        if show_assets:
            # Plot individual assets
            for i, ticker in enumerate(self.tickers):
                plt.scatter(
                    np.sqrt(self.cov_matrix.iloc[i, i]) * 100,
                    self.mean_returns.iloc[i] * 100,
                    s=100,
                    marker="o",
                    label=ticker,
                )

        if show_optimal:
            # Mark max Sharpe portfolio
            max_sharpe = self.optimize_max_sharpe(weight_bounds)
            plt.scatter(
                max_sharpe["volatility"] * 100,
                max_sharpe["return"] * 100,
                s=200,
                marker="*",
                c="red",
                label="Max Sharpe",
            )

            # Mark min volatility portfolio
            min_vol = self.optimize_min_volatility(weight_bounds)
            plt.scatter(
                min_vol["volatility"] * 100,
                min_vol["return"] * 100,
                s=200,
                marker="*",
                c="green",
                label="Min Volatility",
            )

        plt.xlabel("Volatility (%)")
        plt.ylabel("Expected Return (%)")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_comparison(self, weight_bounds: Tuple[float, float] = (0, 1)) -> None:
        """Print comparison of optimization strategies."""
        equal_weight = np.array([1 / self.n_assets] * self.n_assets)
        max_sharpe = self.optimize_max_sharpe(weight_bounds)
        min_vol = self.optimize_min_volatility(weight_bounds)
        risk_parity = self.optimize_risk_parity()

        print("\n" + "=" * 70)
        print("PORTFOLIO OPTIMIZATION COMPARISON")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Return':>12} {'Volatility':>12} {'Sharpe':>10}")
        print("-" * 70)

        # Equal weight
        print(
            f"{'Equal Weight':<20} "
            f"{self._portfolio_return(equal_weight)*100:>11.2f}% "
            f"{self._portfolio_volatility(equal_weight)*100:>11.2f}% "
            f"{self._portfolio_sharpe(equal_weight):>10.2f}"
        )

        # Max Sharpe
        print(
            f"{'Max Sharpe':<20} "
            f"{max_sharpe['return']*100:>11.2f}% "
            f"{max_sharpe['volatility']*100:>11.2f}% "
            f"{max_sharpe['sharpe_ratio']:>10.2f}"
        )

        # Min Volatility
        print(
            f"{'Min Volatility':<20} "
            f"{min_vol['return']*100:>11.2f}% "
            f"{min_vol['volatility']*100:>11.2f}% "
            f"{min_vol['sharpe_ratio']:>10.2f}"
        )

        # Risk Parity
        print(
            f"{'Risk Parity':<20} "
            f"{risk_parity['return']*100:>11.2f}% "
            f"{risk_parity['volatility']*100:>11.2f}% "
            f"{risk_parity['sharpe_ratio']:>10.2f}"
        )

        print("=" * 70)

        print("\nOptimal Weights (Max Sharpe):")
        for ticker, weight in max_sharpe["weights"].items():
            if weight > 0.01:
                print(f"  {ticker}: {weight*100:.1f}%")
