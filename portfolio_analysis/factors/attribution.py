"""
Factor-based return and risk attribution for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from portfolio_analysis.factors.data import align_returns_with_factors
from portfolio_analysis.factors.models import FactorModel, FactorRegression


class FactorAttribution:
    """
    Decompose portfolio returns and risk into factor contributions.

    This class uses factor regression to attribute portfolio performance
    to systematic factors and idiosyncratic (alpha) components.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns with datetime index
    factor_data : pd.DataFrame
        Factor data from FactorDataLoader
    annualization_factor : int, default 252
        Number of periods per year (252 for daily, 12 for monthly)

    Examples
    --------
    >>> from portfolio_analysis.factors import FactorAttribution, FactorDataLoader
    >>> factor_loader = FactorDataLoader()
    >>> ff3 = factor_loader.get_ff3_factors('2015-01-01', '2023-12-31')
    >>> attribution = FactorAttribution(portfolio_returns, ff3)
    >>> decomp = attribution.decompose_returns()
    >>> print(f"Market contribution: {decomp['Mkt-RF']:.2%}")
    >>> print(f"Alpha: {decomp['alpha']:.2%}")
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,
        annualization_factor: Optional[int] = None,
    ):
        self.raw_returns = returns
        self.raw_factor_data = factor_data

        # Align data
        self.excess_returns, self.factor_data = align_returns_with_factors(
            returns, factor_data, compute_excess=True
        )

        # Auto-detect frequency
        if annualization_factor is None:
            if len(self.excess_returns) > 1:
                avg_days = (
                    self.excess_returns.index[-1] - self.excess_returns.index[0]
                ).days / len(self.excess_returns)
                annualization_factor = 12 if avg_days > 20 else self.TRADING_DAYS
            else:
                annualization_factor = self.TRADING_DAYS

        self.annualization_factor = annualization_factor

        # Create regression object for analysis
        self._regression = FactorRegression(returns, factor_data, annualization_factor)

    def decompose_returns(
        self, model: Union[str, FactorModel] = "ff3"
    ) -> Dict[str, float]:
        """
        Decompose total returns into factor contributions.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use for decomposition

        Returns
        -------
        dict
            Dictionary with:
            - 'total': Total annualized return
            - 'risk_free': Risk-free contribution
            - One key per factor with its return contribution
            - 'alpha': Idiosyncratic return (Jensen's alpha)
        """
        # Run regression to get betas
        reg_results = self._regression.run_regression(model)
        factors = reg_results.factors

        # Calculate average factor returns (annualized)
        avg_factor_returns = (
            self.factor_data[factors].mean() * self.annualization_factor
        )

        # Risk-free rate contribution
        rf_return = self.factor_data["RF"].mean() * self.annualization_factor

        # Total return
        total_return = (
            self.raw_returns.loc[self.excess_returns.index].mean()
            * self.annualization_factor
        )

        # Factor contributions = beta * average factor return
        contributions = {}
        contributions["total"] = total_return
        contributions["risk_free"] = rf_return

        for factor in factors:
            contributions[factor] = (
                reg_results.betas[factor] * avg_factor_returns[factor]
            )

        # Alpha is the residual
        contributions["alpha"] = reg_results.alpha

        return contributions

    def decompose_risk(
        self, model: Union[str, FactorModel] = "ff3"
    ) -> Dict[str, float]:
        """
        Decompose portfolio variance into factor contributions.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use for decomposition

        Returns
        -------
        dict
            Dictionary with:
            - 'total': Total annualized variance
            - One key per factor with its variance contribution
            - 'idiosyncratic': Residual (unexplained) variance
            - 'r_squared': Fraction explained by factors
        """
        # Run regression
        reg_results = self._regression.run_regression(model)
        factors = reg_results.factors

        # Factor covariance matrix (annualized)
        factor_cov = self.factor_data[factors].cov() * self.annualization_factor

        # Total variance (annualized)
        total_variance = self.excess_returns.var() * self.annualization_factor

        # Systematic variance = beta' * Cov(factors) * beta
        betas = np.array([reg_results.betas[f] for f in factors])
        systematic_variance = betas @ factor_cov.values @ betas

        # Individual factor contributions (marginal)
        contributions = {"total": total_variance}

        for i, factor in enumerate(factors):
            # Factor contribution = beta_i^2 * var(factor_i)
            factor_var = self.factor_data[factor].var() * self.annualization_factor
            contributions[factor] = reg_results.betas[factor] ** 2 * factor_var

        # Idiosyncratic variance
        contributions["idiosyncratic"] = total_variance - systematic_variance
        contributions["r_squared"] = reg_results.r_squared

        return contributions

    def get_rolling_attribution(
        self, model: Union[str, FactorModel] = "ff3", window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling return attribution over time.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use
        window : int, default 60
            Rolling window size (number of periods)

        Returns
        -------
        pd.DataFrame
            DataFrame with rolling factor contributions, indexed by date
        """
        # Get factors for the model
        if isinstance(model, str):
            model_enum = {
                "capm": FactorModel.CAPM,
                "ff3": FactorModel.FF3,
                "ff5": FactorModel.FF5,
                "carhart": FactorModel.CARHART,
            }.get(model.lower())
            factors = model_enum.value if model_enum else FactorModel.FF3.value
        else:
            factors = model.value

        # Get rolling betas
        rolling_betas = self._regression.run_rolling_regression(model, window)

        # Calculate rolling factor returns
        results = []
        for date in rolling_betas.index:
            # Get window data
            loc = self.factor_data.index.get_loc(date)
            start_loc = max(0, loc - window + 1)
            factor_window = self.factor_data.iloc[start_loc : loc + 1]

            # Average factor returns (annualized)
            avg_returns = factor_window[factors].mean() * self.annualization_factor

            # Contributions
            row = {"date": date}
            for factor in factors:
                beta = rolling_betas.loc[date, factor]
                row[f"{factor}_contrib"] = beta * avg_returns[factor]
            row["alpha"] = rolling_betas.loc[date, "alpha"]

            results.append(row)

        df = pd.DataFrame(results)
        if "date" in df.columns:
            df = df.set_index("date")

        return df

    def get_attribution_summary(
        self, model: Union[str, FactorModel] = "ff3"
    ) -> pd.DataFrame:
        """
        Get a summary table of return and risk attribution.

        Parameters
        ----------
        model : str or FactorModel, default 'ff3'
            Factor model to use

        Returns
        -------
        pd.DataFrame
            Summary table with return and risk contributions
        """
        return_decomp = self.decompose_returns(model)
        risk_decomp = self.decompose_risk(model)

        # Build summary
        rows = []

        # Total
        rows.append(
            {
                "Component": "Total",
                "Return (%)": return_decomp["total"] * 100,
                "Variance": risk_decomp["total"],
                "Std Dev (%)": np.sqrt(risk_decomp["total"]) * 100,
            }
        )

        # Risk-free
        rows.append(
            {
                "Component": "Risk-Free",
                "Return (%)": return_decomp["risk_free"] * 100,
                "Variance": 0,
                "Std Dev (%)": 0,
            }
        )

        # Factors
        reg_results = self._regression.run_regression(model)
        for factor in reg_results.factors:
            rows.append(
                {
                    "Component": factor,
                    "Return (%)": return_decomp[factor] * 100,
                    "Variance": risk_decomp.get(factor, 0),
                    "Std Dev (%)": np.sqrt(risk_decomp.get(factor, 0)) * 100,
                }
            )

        # Alpha / Idiosyncratic
        rows.append(
            {
                "Component": "Alpha (Idiosyncratic)",
                "Return (%)": return_decomp["alpha"] * 100,
                "Variance": risk_decomp["idiosyncratic"],
                "Std Dev (%)": np.sqrt(max(0, risk_decomp["idiosyncratic"])) * 100,
            }
        )

        return pd.DataFrame(rows)

    def summary(self, model: Union[str, FactorModel] = "ff3") -> str:
        """Generate a text summary of factor attribution."""
        return_decomp = self.decompose_returns(model)
        risk_decomp = self.decompose_risk(model)

        model_name = model.name if isinstance(model, FactorModel) else model.upper()

        lines = [
            f"\n{'=' * 60}",
            f"Factor Attribution Summary: {model_name}",
            f"{'=' * 60}",
            "",
            "RETURN ATTRIBUTION",
            f"{'-' * 40}",
            f"{'Component':<20} {'Return':>12}",
            f"{'-' * 40}",
            f"{'Total':<20} {return_decomp['total']*100:>11.2f}%",
            f"{'Risk-Free':<20} {return_decomp['risk_free']*100:>11.2f}%",
        ]

        # Factor contributions
        reg_results = self._regression.run_regression(model)
        for factor in reg_results.factors:
            lines.append(f"{factor:<20} {return_decomp[factor]*100:>11.2f}%")
        lines.append(f"{'Alpha':<20} {return_decomp['alpha']*100:>11.2f}%")

        lines.extend(
            [
                "",
                "RISK ATTRIBUTION",
                f"{'-' * 40}",
                f"{'Component':<20} {'Variance':>12} {'% of Total':>12}",
                f"{'-' * 40}",
            ]
        )

        total_var = risk_decomp["total"]
        for factor in reg_results.factors:
            pct = risk_decomp[factor] / total_var * 100 if total_var > 0 else 0
            lines.append(f"{factor:<20} {risk_decomp[factor]:>12.6f} {pct:>11.1f}%")

        idio_pct = (
            risk_decomp["idiosyncratic"] / total_var * 100 if total_var > 0 else 0
        )
        lines.append(
            f"{'Idiosyncratic':<20} {risk_decomp['idiosyncratic']:>12.6f} {idio_pct:>11.1f}%"
        )
        lines.append(f"\nR-squared: {risk_decomp['r_squared']:.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)
