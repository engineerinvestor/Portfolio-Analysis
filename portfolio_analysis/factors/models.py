"""
Factor regression models for portfolio analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

from portfolio_analysis.factors.data import align_returns_with_factors


class FactorModel(Enum):
    """
    Supported factor models.

    Attributes
    ----------
    CAPM : Single-factor market model
    FF3 : Fama-French 3-factor model (Mkt-RF, SMB, HML)
    FF5 : Fama-French 5-factor model (+ RMW, CMA)
    CARHART : Carhart 4-factor model (FF3 + MOM)
    """
    CAPM = ['Mkt-RF']
    FF3 = ['Mkt-RF', 'SMB', 'HML']
    FF5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    CARHART = ['Mkt-RF', 'SMB', 'HML', 'MOM']


@dataclass
class RegressionResults:
    """
    Results from a factor regression.

    Attributes
    ----------
    alpha : float
        Jensen's alpha (annualized intercept)
    alpha_pvalue : float
        P-value for alpha significance test
    betas : dict
        Factor loadings (sensitivities)
    beta_pvalues : dict
        P-values for each beta
    beta_tstats : dict
        T-statistics for each beta
    r_squared : float
        R-squared (explained variance)
    adj_r_squared : float
        Adjusted R-squared
    residual_std : float
        Standard deviation of residuals (annualized)
    n_observations : int
        Number of observations used
    model : str
        Model name used for regression
    factors : list
        Factor names used in the model
    """
    alpha: float
    alpha_pvalue: float
    betas: Dict[str, float]
    beta_pvalues: Dict[str, float]
    beta_tstats: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_std: float
    n_observations: int
    model: str
    factors: List[str]

    def summary(self) -> str:
        """Generate a text summary of regression results."""
        lines = [
            f"\n{'=' * 60}",
            f"Factor Regression Results: {self.model}",
            f"{'=' * 60}",
            f"Observations: {self.n_observations}",
            f"R-squared: {self.r_squared:.4f}",
            f"Adj R-squared: {self.adj_r_squared:.4f}",
            f"Residual Std: {self.residual_std * 100:.2f}% (annualized)",
            f"\n{'Coefficient':<12} {'Value':>10} {'T-stat':>10} {'P-value':>10}",
            f"{'-' * 42}",
            f"{'Alpha':<12} {self.alpha * 100:>9.2f}% {self._alpha_tstat():>10.2f} {self.alpha_pvalue:>10.4f}",
        ]
        for factor in self.factors:
            lines.append(
                f"{factor:<12} {self.betas[factor]:>10.3f} "
                f"{self.beta_tstats[factor]:>10.2f} {self.beta_pvalues[factor]:>10.4f}"
            )
        lines.append('=' * 60)
        return '\n'.join(lines)

    def _alpha_tstat(self) -> float:
        """Approximate t-stat for alpha from p-value."""
        from scipy import stats
        if self.alpha_pvalue >= 1.0:
            return 0.0
        if self.alpha_pvalue <= 0.0:
            return np.inf if self.alpha > 0 else -np.inf
        return stats.norm.ppf(1 - self.alpha_pvalue / 2) * np.sign(self.alpha)

    def __repr__(self) -> str:
        return (
            f"RegressionResults(model='{self.model}', alpha={self.alpha:.4f}, "
            f"r_squared={self.r_squared:.4f})"
        )


class FactorRegression:
    """
    Run factor regressions on portfolio or asset returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio or asset returns with datetime index
    factor_data : pd.DataFrame
        Factor data from FactorDataLoader
    annualization_factor : int, default 252
        Number of periods per year (252 for daily, 12 for monthly)

    Examples
    --------
    >>> from portfolio_analysis.factors import FactorDataLoader, FactorRegression
    >>> factor_loader = FactorDataLoader()
    >>> ff3 = factor_loader.get_ff3_factors('2015-01-01', '2023-12-31')
    >>> regression = FactorRegression(portfolio_returns, ff3)
    >>> results = regression.run_regression('ff3')
    >>> print(results.summary())
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,
        annualization_factor: Optional[int] = None
    ):
        self.raw_returns = returns
        self.raw_factor_data = factor_data

        # Align data
        self.excess_returns, self.factor_data = align_returns_with_factors(
            returns, factor_data, compute_excess=True
        )

        # Auto-detect frequency if not specified
        if annualization_factor is None:
            # Check average days between observations
            if len(self.excess_returns) > 1:
                avg_days = (
                    self.excess_returns.index[-1] - self.excess_returns.index[0]
                ).days / len(self.excess_returns)
                if avg_days > 20:  # Monthly
                    annualization_factor = 12
                else:  # Daily
                    annualization_factor = self.TRADING_DAYS
            else:
                annualization_factor = self.TRADING_DAYS

        self.annualization_factor = annualization_factor

    def _get_model_factors(self, model: Union[str, FactorModel]) -> List[str]:
        """Get factor names for a model."""
        if isinstance(model, str):
            model = model.lower()
            if model == 'capm':
                return FactorModel.CAPM.value
            elif model == 'ff3':
                return FactorModel.FF3.value
            elif model == 'ff5':
                return FactorModel.FF5.value
            elif model == 'carhart':
                return FactorModel.CARHART.value
            else:
                raise ValueError(f"Unknown model: {model}. Use 'capm', 'ff3', 'ff5', or 'carhart'")
        return model.value

    def _ols_regression(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> tuple:
        """
        Run OLS regression with statistical inference.

        Returns coefficients, t-stats, p-values, and fit statistics.
        """
        from scipy import stats

        n = len(y)
        k = X.shape[1]

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(n), X])

        # OLS: (X'X)^-1 X'y
        XtX = X_with_const.T @ X_with_const
        XtX_inv = np.linalg.inv(XtX)
        coeffs = XtX_inv @ X_with_const.T @ y

        # Residuals and variance
        y_hat = X_with_const @ coeffs
        residuals = y - y_hat
        sse = residuals @ residuals
        dof = n - k - 1  # degrees of freedom

        if dof <= 0:
            raise ValueError("Not enough observations for regression")

        mse = sse / dof
        residual_std = np.sqrt(mse)

        # Coefficient standard errors
        se = np.sqrt(np.diag(XtX_inv) * mse)

        # T-statistics and p-values
        t_stats = coeffs / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        # R-squared
        ss_total = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - sse / ss_total if ss_total > 0 else 0

        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof

        return {
            'coeffs': coeffs,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'residual_std': residual_std,
            'n': n
        }

    def run_regression(
        self,
        model: Union[str, FactorModel] = 'ff3'
    ) -> RegressionResults:
        """
        Run a factor regression.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use: 'capm', 'ff3', 'ff5', or 'carhart'

        Returns
        -------
        RegressionResults
            Regression results with alpha, betas, and statistics
        """
        factors = self._get_model_factors(model)

        # Validate factors exist in data
        missing = [f for f in factors if f not in self.factor_data.columns]
        if missing:
            raise ValueError(f"Factors not in data: {missing}. Available: {self.factor_data.columns.tolist()}")

        # Prepare data
        y = self.excess_returns.values
        X = self.factor_data[factors].values

        # Run regression
        results = self._ols_regression(y, X)

        # Extract results
        alpha = results['coeffs'][0]
        betas = dict(zip(factors, results['coeffs'][1:]))
        beta_tstats = dict(zip(factors, results['t_stats'][1:]))
        beta_pvalues = dict(zip(factors, results['p_values'][1:]))

        # Annualize alpha and residual std
        alpha_annual = alpha * self.annualization_factor
        residual_std_annual = results['residual_std'] * np.sqrt(self.annualization_factor)

        model_name = model.name if isinstance(model, FactorModel) else model.upper()

        return RegressionResults(
            alpha=alpha_annual,
            alpha_pvalue=results['p_values'][0],
            betas=betas,
            beta_pvalues=beta_pvalues,
            beta_tstats=beta_tstats,
            r_squared=results['r_squared'],
            adj_r_squared=results['adj_r_squared'],
            residual_std=residual_std_annual,
            n_observations=results['n'],
            model=model_name,
            factors=factors
        )

    def run_rolling_regression(
        self,
        model: Union[str, FactorModel] = 'ff3',
        window: int = 60
    ) -> pd.DataFrame:
        """
        Run rolling factor regressions.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use
        window : int, default 60
            Rolling window size (number of periods)

        Returns
        -------
        pd.DataFrame
            DataFrame with rolling alpha and betas, indexed by date
        """
        factors = self._get_model_factors(model)
        y = self.excess_returns
        X = self.factor_data[factors]

        results = []
        dates = []

        for i in range(window, len(y) + 1):
            y_window = y.iloc[i - window:i].values
            X_window = X.iloc[i - window:i].values

            try:
                reg = self._ols_regression(y_window, X_window)
                result = {'alpha': reg['coeffs'][0] * self.annualization_factor}
                for j, factor in enumerate(factors):
                    result[factor] = reg['coeffs'][j + 1]
                result['r_squared'] = reg['r_squared']
                results.append(result)
                dates.append(y.index[i - 1])
            except Exception:
                continue

        return pd.DataFrame(results, index=pd.DatetimeIndex(dates))

    def compare_models(self) -> pd.DataFrame:
        """
        Compare different factor models.

        Returns
        -------
        pd.DataFrame
            Comparison table with alpha, R-squared, and key betas for each model
        """
        models = ['capm', 'ff3']

        # Add ff5 if factors available
        if all(f in self.factor_data.columns for f in FactorModel.FF5.value):
            models.append('ff5')

        # Add carhart if momentum available
        if 'MOM' in self.factor_data.columns:
            models.append('carhart')

        results = []
        for model in models:
            try:
                reg = self.run_regression(model)
                result = {
                    'Model': reg.model,
                    'Alpha (%)': reg.alpha * 100,
                    'Alpha p-value': reg.alpha_pvalue,
                    'R-squared': reg.r_squared,
                    'Adj R-squared': reg.adj_r_squared,
                    'Mkt Beta': reg.betas.get('Mkt-RF', np.nan),
                }
                # Add other betas if available
                for factor in ['SMB', 'HML', 'RMW', 'CMA', 'MOM']:
                    if factor in reg.betas:
                        result[factor] = reg.betas[factor]
                results.append(result)
            except Exception:
                continue

        return pd.DataFrame(results)
