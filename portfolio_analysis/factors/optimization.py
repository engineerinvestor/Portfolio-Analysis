"""
Factor-aware portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize

from portfolio_analysis.factors.data import align_returns_with_factors
from portfolio_analysis.factors.models import FactorModel


class FactorOptimizer:
    """
    Factor-aware portfolio optimization.

    Optimize portfolios to achieve target factor exposures, minimize
    factor exposure, or generate factor-efficient frontiers.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data with datetime index and tickers as columns
    factor_data : pd.DataFrame
        Factor data from FactorDataLoader
    risk_free_rate : float, default 0.02
        Annual risk-free rate for Sharpe calculations

    Examples
    --------
    >>> from portfolio_analysis.factors import FactorOptimizer, FactorDataLoader
    >>> factor_loader = FactorDataLoader()
    >>> ff3 = factor_loader.get_ff3_factors('2015-01-01', '2023-12-31')
    >>> optimizer = FactorOptimizer(price_data, ff3)
    >>> result = optimizer.optimize_target_exposures(
    ...     target_betas={'Mkt-RF': 1.0, 'SMB': 0.3, 'HML': 0.2}
    ... )
    >>> print(result['weights'])
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        price_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ):
        self.price_data = price_data
        self.tickers = list(price_data.columns)
        self.n_assets = len(self.tickers)
        self.risk_free_rate = risk_free_rate

        # Calculate returns
        self.returns = price_data.pct_change().dropna()

        # Align with factor data
        common_dates = self.returns.index.intersection(factor_data.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between price data and factor data")

        self.returns = self.returns.loc[common_dates]
        self.factor_data = factor_data.loc[common_dates]

        # Calculate excess returns for each asset
        self.excess_returns = self.returns.sub(self.factor_data["RF"], axis=0)

        # Pre-compute individual asset betas for all factors
        self._asset_betas = self._compute_asset_betas()

        # Annualized statistics
        self.mean_returns = self.returns.mean() * self.TRADING_DAYS
        self.cov_matrix = self.returns.cov() * self.TRADING_DAYS

    def _compute_asset_betas(self) -> pd.DataFrame:
        """Compute factor betas for each individual asset."""
        factors = [
            f
            for f in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
            if f in self.factor_data.columns
        ]

        betas = {}
        for ticker in self.tickers:
            y = self.excess_returns[ticker].values
            X = self.factor_data[factors].values

            # Add constant
            X_const = np.column_stack([np.ones(len(X)), X])

            # OLS
            coeffs = np.linalg.lstsq(X_const, y, rcond=None)[0]
            betas[ticker] = dict(zip(factors, coeffs[1:]))

        return pd.DataFrame(betas).T

    def _portfolio_betas(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio-level factor betas given weights."""
        portfolio_betas = {}
        for factor in self._asset_betas.columns:
            portfolio_betas[factor] = float(np.dot(weights, self._asset_betas[factor]))
        return portfolio_betas

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return."""
        return float(np.dot(weights, self.mean_returns))

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0

    def optimize_target_exposures(
        self,
        target_betas: Dict[str, float],
        weight_bounds: Tuple[float, float] = (0, 1),
        tolerance: float = 0.1,
    ) -> Dict:
        """
        Optimize portfolio to achieve target factor exposures.

        Minimizes tracking error to target betas while maximizing Sharpe ratio.

        Parameters
        ----------
        target_betas : dict
            Target factor exposures (e.g., {'Mkt-RF': 1.0, 'SMB': 0.3})
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset
        tolerance : float, default 0.1
            Allowed deviation from target betas

        Returns
        -------
        dict
            Optimal weights, achieved betas, return, volatility, and Sharpe ratio
        """
        # Validate factors exist
        for factor in target_betas:
            if factor not in self._asset_betas.columns:
                raise ValueError(
                    f"Factor '{factor}' not available. "
                    f"Available: {self._asset_betas.columns.tolist()}"
                )

        def objective(weights):
            # Maximize Sharpe (minimize negative Sharpe)
            return -self._portfolio_sharpe(weights)

        def beta_constraint(weights, factor, target):
            """Constraint: achieved beta should be close to target."""
            achieved = np.dot(weights, self._asset_betas[factor])
            return tolerance - abs(achieved - target)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        # Add beta constraints
        for factor, target in target_betas.items():
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, f=factor, t=target: beta_constraint(x, f, t),
                }
            )

        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        optimal_weights = result.x

        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "achieved_betas": self._portfolio_betas(optimal_weights),
            "target_betas": target_betas,
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
            "success": result.success,
        }

    def optimize_factor_neutral(
        self,
        factors: List[str],
        weight_bounds: Tuple[float, float] = (0, 1),
        tolerance: float = 0.05,
    ) -> Dict:
        """
        Optimize portfolio to be neutral to specified factors.

        Parameters
        ----------
        factors : list of str
            Factors to neutralize (e.g., ['SMB', 'HML'])
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset
        tolerance : float, default 0.05
            Maximum allowed absolute beta for neutral factors

        Returns
        -------
        dict
            Optimal weights with near-zero exposure to specified factors
        """
        # Validate factors
        for factor in factors:
            if factor not in self._asset_betas.columns:
                raise ValueError(f"Factor '{factor}' not available")

        def objective(weights):
            return -self._portfolio_sharpe(weights)

        def neutrality_constraint(weights, factor):
            """Beta should be close to zero."""
            beta = np.dot(weights, self._asset_betas[factor])
            return tolerance - abs(beta)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        for factor in factors:
            constraints.append(
                {"type": "ineq", "fun": lambda x, f=factor: neutrality_constraint(x, f)}
            )

        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        optimal_weights = result.x

        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "achieved_betas": self._portfolio_betas(optimal_weights),
            "neutralized_factors": factors,
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
            "success": result.success,
        }

    def optimize_max_alpha(
        self,
        model: Union[str, FactorModel] = "ff3",
        weight_bounds: Tuple[float, float] = (0, 1),
    ) -> Dict:
        """
        Optimize portfolio to maximize expected alpha.

        Uses pre-computed asset alphas to find the highest-alpha portfolio.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model for alpha calculation
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        dict
            Portfolio weights maximizing expected alpha
        """
        # Get model factors
        if isinstance(model, str):
            model_factors = {
                "capm": ["Mkt-RF"],
                "ff3": ["Mkt-RF", "SMB", "HML"],
                "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                "carhart": ["Mkt-RF", "SMB", "HML", "MOM"],
            }.get(model.lower(), ["Mkt-RF", "SMB", "HML"])
        else:
            model_factors = model.value

        factors = [f for f in model_factors if f in self.factor_data.columns]

        # Compute alpha for each asset
        alphas = []
        for ticker in self.tickers:
            y = self.excess_returns[ticker].values
            X = self.factor_data[factors].values
            X_const = np.column_stack([np.ones(len(X)), X])
            coeffs = np.linalg.lstsq(X_const, y, rcond=None)[0]
            alphas.append(coeffs[0] * self.TRADING_DAYS)  # Annualized alpha

        alphas = np.array(alphas)

        def objective(weights):
            # Negative alpha (minimize)
            return -np.dot(weights, alphas)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple(weight_bounds for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x

        return {
            "weights": dict(zip(self.tickers, optimal_weights)),
            "expected_alpha": float(np.dot(optimal_weights, alphas)),
            "asset_alphas": dict(zip(self.tickers, alphas)),
            "achieved_betas": self._portfolio_betas(optimal_weights),
            "return": self._portfolio_return(optimal_weights),
            "volatility": self._portfolio_volatility(optimal_weights),
            "sharpe_ratio": self._portfolio_sharpe(optimal_weights),
            "success": result.success,
        }

    def generate_factor_frontier(
        self,
        factor: str,
        n_points: int = 20,
        weight_bounds: Tuple[float, float] = (0, 1),
    ) -> pd.DataFrame:
        """
        Generate efficient frontier varying one factor's exposure.

        Parameters
        ----------
        factor : str
            Factor to vary (e.g., 'SMB', 'HML')
        n_points : int, default 20
            Number of points on the frontier
        weight_bounds : tuple, default (0, 1)
            Min and max weight for each asset

        Returns
        -------
        pd.DataFrame
            Frontier with columns: factor_beta, return, volatility, sharpe_ratio
        """
        if factor not in self._asset_betas.columns:
            raise ValueError(f"Factor '{factor}' not available")

        # Find beta range
        asset_betas = self._asset_betas[factor].values
        min_beta = asset_betas.min()
        max_beta = asset_betas.max()

        target_betas = np.linspace(min_beta, max_beta, n_points)

        frontier = []
        for target in target_betas:
            try:
                result = self.optimize_target_exposures(
                    target_betas={factor: target},
                    weight_bounds=weight_bounds,
                    tolerance=0.05,
                )
                if result["success"]:
                    frontier.append(
                        {
                            f"{factor}_beta": result["achieved_betas"][factor],
                            "return": result["return"],
                            "volatility": result["volatility"],
                            "sharpe_ratio": result["sharpe_ratio"],
                        }
                    )
            except Exception:
                continue

        return pd.DataFrame(frontier)

    def get_asset_betas(self) -> pd.DataFrame:
        """
        Get factor betas for all individual assets.

        Returns
        -------
        pd.DataFrame
            DataFrame with assets as rows and factors as columns
        """
        return self._asset_betas.copy()

    def summary(self) -> str:
        """Generate a summary of optimization capabilities and asset betas."""
        lines = [
            f"\n{'=' * 60}",
            "Factor Optimizer Summary",
            f"{'=' * 60}",
            f"Assets: {len(self.tickers)}",
            f"Observations: {len(self.returns)}",
            f"Available factors: {self._asset_betas.columns.tolist()}",
            "",
            "Asset Factor Betas:",
            f"{'-' * 60}",
        ]

        # Format asset betas table
        beta_str = self._asset_betas.to_string()
        lines.append(beta_str)

        lines.append("=" * 60)
        return "\n".join(lines)
