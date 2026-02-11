"""
Composite global factor regression for multi-asset, multi-region portfolios.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from portfolio_analysis.factors.data import FactorDataLoader
from portfolio_analysis.factors.models import (
    FactorModel,
    FactorRegression,
    RegressionResults,
)


@dataclass
class CompositeRegressionResults:
    """
    Results from a composite multi-region factor regression.

    Attributes
    ----------
    weighted_betas : dict[str, float]
        Weighted-average factor loadings across all constituents.
    weighted_alpha : float
        Weighted-average annualized alpha.
    constituent_results : dict[str, RegressionResults]
        Per-ticker regression results.
    portfolio_weights : dict[str, float]
        Portfolio weights used ({ticker: weight}).
    region_map : dict[str, str]
        Region assignment used ({ticker: region}).
    coverage : float
        Fraction of total portfolio weight successfully analyzed.
    """

    weighted_betas: dict[str, float]
    weighted_alpha: float
    constituent_results: dict[str, RegressionResults]
    portfolio_weights: dict[str, float]
    region_map: dict[str, str]
    coverage: float

    def summary(self) -> str:
        """Generate a text summary of composite regression results."""
        lines = [
            f"\n{'=' * 60}",
            "Composite Global Factor Regression Results",
            f"{'=' * 60}",
            f"Constituents analyzed: {len(self.constituent_results)}",
            f"Portfolio coverage: {self.coverage:.1%}",
            "",
            f"{'Factor':<12} {'Weighted Beta':>14}",
            f"{'-' * 28}",
        ]

        for factor, beta in self.weighted_betas.items():
            lines.append(f"{factor:<12} {beta:>14.4f}")

        lines.append(f"{'Alpha':<12} {self.weighted_alpha * 100:>13.2f}%")
        lines.append("")

        lines.append(
            f"{'Ticker':<8} {'Weight':>8} {'Region':<18} {'R²':>6} {'Alpha':>8}"
        )
        lines.append(f"{'-' * 52}")
        for ticker, result in self.constituent_results.items():
            weight = self.portfolio_weights.get(ticker, 0)
            region = self.region_map.get(ticker, "us")
            lines.append(
                f"{ticker:<8} {weight:>7.1%} {region:<18} "
                f"{result.r_squared:>6.4f} {result.alpha * 100:>7.2f}%"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def comparison_table(self) -> pd.DataFrame:
        """
        Build a per-ETF comparison table.

        Returns
        -------
        pd.DataFrame
            Columns: Ticker, Weight, Region, R², Alpha (%), plus one column
            per factor beta.
        """
        rows = []
        for ticker, result in self.constituent_results.items():
            row = {
                "Ticker": ticker,
                "Weight": self.portfolio_weights.get(ticker, 0),
                "Region": self.region_map.get(ticker, "us"),
                "R²": result.r_squared,
                "Alpha (%)": result.alpha * 100,
            }
            for factor in result.factors:
                row[factor] = result.betas[factor]
            rows.append(row)

        return pd.DataFrame(rows).set_index("Ticker")


class CompositeFactorRegression:
    """
    Run factor regressions on a multi-asset portfolio using per-asset regional factors.

    Parameters
    ----------
    returns : pd.DataFrame
        Multi-column daily returns (columns = tickers).
    portfolio_weights : dict[str, float]
        {ticker: weight}, should sum to ~1.0.
    region_map : dict[str, str]
        {ticker: region}. Tickers not in the map default to ``'us'``.
    factor_loader : FactorDataLoader, optional
        Reusable loader instance. Created if not provided.
    start_date : str, optional
        Start date for factor data. Inferred from returns if not provided.
    end_date : str, optional
        End date for factor data. Inferred from returns if not provided.

    Examples
    --------
    >>> composite = CompositeFactorRegression(
    ...     returns=returns_df,
    ...     portfolio_weights={'AVUS': 0.42, 'AVDE': 0.11, 'AVEM': 0.07},
    ...     region_map={'AVDE': 'developed_ex_us', 'AVEM': 'emerging'},
    ... )
    >>> results = composite.run_composite_regression('ff5')
    >>> print(results.summary())
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        portfolio_weights: dict[str, float],
        region_map: Optional[dict[str, str]] = None,
        factor_loader: Optional[FactorDataLoader] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.returns = returns
        self.portfolio_weights = portfolio_weights
        self.region_map = region_map or {}
        self.factor_loader = factor_loader or FactorDataLoader()

        # Infer date range from returns if not provided
        self.start_date = start_date or str(returns.index.min().date())
        self.end_date = end_date or str(returns.index.max().date())

        # Pre-fetch factor data for each unique region
        regions_needed = set(self.region_map.values()) | {"us"}
        self._factor_cache: dict[str, pd.DataFrame] = {}
        for region in regions_needed:
            self._factor_cache[region] = self._fetch_factors_for_region(region)

    def _fetch_factors_for_region(self, region: str) -> pd.DataFrame:
        """
        Fetch FF5 + MOM factor data for a region.

        For regions that only have monthly data (e.g. emerging), fetches monthly.
        For others, fetches daily. Returns combined DataFrame with MOM column.
        """
        frequency = self._get_available_frequency(region)

        ff5 = self.factor_loader.get_ff5_factors(
            self.start_date, self.end_date, frequency=frequency, region=region
        )

        try:
            mom = self.factor_loader.get_momentum_factor(
                self.start_date, self.end_date, frequency=frequency, region=region
            )
            common = ff5.index.intersection(mom.index)
            factors = ff5.loc[common].copy()
            factors["MOM"] = mom.loc[common]
        except (ValueError, KeyError):
            factors = ff5.copy()

        return factors

    @staticmethod
    def _get_available_frequency(region: str) -> str:
        """Return the best available frequency for a region."""
        region_datasets = FactorDataLoader.REGIONS.get(region, {})
        if region_datasets.get("ff5_daily") is not None:
            return "daily"
        return "monthly"

    def _prepare_returns(self, ticker: str, region: str) -> pd.Series:
        """
        Prepare asset returns, resampling to monthly if needed for the region.
        """
        etf_ret = self.returns[ticker].dropna()
        frequency = self._get_available_frequency(region)

        if frequency == "monthly":
            # Resample daily returns to monthly
            monthly_ret = (1 + etf_ret).resample("ME").prod() - 1
            return monthly_ret.dropna()

        return etf_ret

    def run_composite_regression(
        self, model: Union[str, FactorModel] = "ff5"
    ) -> CompositeRegressionResults:
        """
        Run per-asset regressions using regional factors and aggregate.

        Parameters
        ----------
        model : str or FactorModel, default 'ff5'
            Factor model to use: 'capm', 'ff3', 'ff5', or 'carhart'.

        Returns
        -------
        CompositeRegressionResults
            Aggregated results with weighted betas and alpha.
        """
        constituent_results: dict[str, RegressionResults] = {}
        analyzed_weight = 0.0

        for ticker, weight in self.portfolio_weights.items():
            if ticker not in self.returns.columns:
                continue

            region = self.region_map.get(ticker, "us")
            factor_data = self._factor_cache.get(region)

            if factor_data is None:
                continue

            etf_ret = self._prepare_returns(ticker, region)

            if len(etf_ret) < 12:
                continue

            try:
                reg = FactorRegression(etf_ret, factor_data)
                result = reg.run_regression(model)
                constituent_results[ticker] = result
                analyzed_weight += weight
            except (ValueError, np.linalg.LinAlgError):
                continue

        if analyzed_weight == 0:
            raise ValueError(
                "No assets could be analyzed. Check returns and factor data."
            )

        # Aggregate weighted betas and alpha
        scale = 1.0 / analyzed_weight
        weighted_betas: dict[str, float] = {}
        weighted_alpha = 0.0

        for ticker, result in constituent_results.items():
            w = self.portfolio_weights[ticker]
            weighted_alpha += result.alpha * w
            for factor in result.factors:
                weighted_betas[factor] = (
                    weighted_betas.get(factor, 0) + result.betas[factor] * w
                )

        weighted_alpha *= scale
        weighted_betas = {k: v * scale for k, v in weighted_betas.items()}

        # Build effective region map (only analyzed tickers)
        effective_region_map = {
            t: self.region_map.get(t, "us") for t in constituent_results
        }

        return CompositeRegressionResults(
            weighted_betas=weighted_betas,
            weighted_alpha=weighted_alpha,
            constituent_results=constituent_results,
            portfolio_weights=self.portfolio_weights,
            region_map=effective_region_map,
            coverage=analyzed_weight,
        )

    def compare_us_vs_regional(
        self, model: Union[str, FactorModel] = "ff5"
    ) -> pd.DataFrame:
        """
        Compare US-only vs regional factor regressions for each asset.

        Parameters
        ----------
        model : str or FactorModel, default 'ff5'
            Factor model to use.

        Returns
        -------
        pd.DataFrame
            Comparison table with US R², Regional R², alpha values, and changes.
        """
        us_factors = self._factor_cache.get("us")
        rows = []

        for ticker, weight in self.portfolio_weights.items():
            if ticker not in self.returns.columns:
                continue

            region = self.region_map.get(ticker, "us")
            regional_factors = self._factor_cache.get(region)

            if us_factors is None or regional_factors is None:
                continue

            # US regression (always daily)
            us_ret = self.returns[ticker].dropna()
            if len(us_ret) < 12:
                continue

            try:
                us_reg = FactorRegression(us_ret, us_factors)
                us_result = us_reg.run_regression(model)
            except (ValueError, np.linalg.LinAlgError):
                continue

            # Regional regression
            reg_ret = self._prepare_returns(ticker, region)
            if len(reg_ret) < 12:
                continue

            try:
                reg_reg = FactorRegression(reg_ret, regional_factors)
                reg_result = reg_reg.run_regression(model)
            except (ValueError, np.linalg.LinAlgError):
                continue

            rows.append(
                {
                    "Ticker": ticker,
                    "Weight": weight,
                    "Region": region,
                    "US R²": round(us_result.r_squared, 4),
                    "Regional R²": round(reg_result.r_squared, 4),
                    "R² Change": round(reg_result.r_squared - us_result.r_squared, 4),
                    "US Alpha (%)": round(us_result.alpha * 100, 2),
                    "Regional Alpha (%)": round(reg_result.alpha * 100, 2),
                    "Alpha Change (pp)": round(
                        (reg_result.alpha - us_result.alpha) * 100, 2
                    ),
                }
            )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("Ticker")
