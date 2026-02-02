"""
Characteristic-based factor exposures for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import yfinance as yf


class FactorExposures:
    """
    Calculate characteristic-based factor exposures for a portfolio.

    This class estimates factor tilts based on security characteristics
    (market cap, valuation ratios, momentum, etc.) rather than regression.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols in the portfolio
    weights : list of float
        Portfolio weights for each ticker (must sum to 1.0)

    Examples
    --------
    >>> exposures = FactorExposures(['VTI', 'VBR', 'VTV'], [0.5, 0.25, 0.25])
    >>> tilts = exposures.get_all_tilts()
    >>> print(f"Size tilt: {tilts['size']:.2f}")
    >>> print(f"Value tilt: {tilts['value']:.2f}")
    """

    # Market cap thresholds (in billions)
    LARGE_CAP_THRESHOLD = 10.0
    SMALL_CAP_THRESHOLD = 2.0

    def __init__(self, tickers: List[str], weights: List[float]):
        if len(tickers) != len(weights):
            raise ValueError("Number of tickers must match number of weights")

        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        self.tickers = tickers
        self.weights = np.array(weights)
        self._characteristics: Optional[pd.DataFrame] = None

    def _fetch_characteristics(self) -> pd.DataFrame:
        """Fetch fundamental characteristics for all tickers."""
        if self._characteristics is not None:
            return self._characteristics

        data = []
        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                data.append(
                    {
                        "ticker": ticker,
                        "market_cap": info.get("marketCap", None),
                        "pe_ratio": info.get("trailingPE", info.get("forwardPE", None)),
                        "pb_ratio": info.get("priceToBook", None),
                        "dividend_yield": info.get("dividendYield", 0) or 0,
                        "beta": info.get("beta", None),
                        "profit_margin": info.get("profitMargins", None),
                        "roe": info.get("returnOnEquity", None),
                        "debt_to_equity": info.get("debtToEquity", None),
                        "revenue_growth": info.get("revenueGrowth", None),
                        "earnings_growth": info.get("earningsGrowth", None),
                    }
                )
            except Exception:
                # Use defaults for ETFs or failed lookups
                data.append(
                    {
                        "ticker": ticker,
                        "market_cap": None,
                        "pe_ratio": None,
                        "pb_ratio": None,
                        "dividend_yield": 0,
                        "beta": 1.0,
                        "profit_margin": None,
                        "roe": None,
                        "debt_to_equity": None,
                        "revenue_growth": None,
                        "earnings_growth": None,
                    }
                )

        self._characteristics = pd.DataFrame(data).set_index("ticker")
        return self._characteristics

    def _calculate_momentum(self, lookback_months: int = 12) -> pd.Series:
        """Calculate momentum based on historical returns."""
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30 + 30)

        try:
            prices = yf.download(
                self.tickers, start=start_date, end=end_date, progress=False
            )

            # Handle column format
            if isinstance(prices.columns, pd.MultiIndex):
                if "Adj Close" in prices.columns.get_level_values(0):
                    prices = prices["Adj Close"]
                else:
                    prices = prices["Close"]

            # Calculate momentum (skip most recent month)
            if len(prices) > 21:
                momentum = (prices.iloc[-22] / prices.iloc[0]) - 1
            else:
                momentum = pd.Series(0, index=self.tickers)

            return momentum

        except Exception:
            return pd.Series(0, index=self.tickers)

    def calculate_size_tilt(self) -> float:
        """
        Calculate portfolio size tilt (SMB exposure).

        Returns
        -------
        float
            Size tilt from -1 (large cap) to +1 (small cap)
            0 indicates market-neutral size exposure
        """
        chars = self._fetch_characteristics()

        # Convert market cap to billions
        market_caps = chars["market_cap"].fillna(chars["market_cap"].median())
        market_caps_b = market_caps / 1e9

        # Score each holding: -1 for large, 0 for mid, +1 for small
        scores = []
        for mc in market_caps_b:
            if mc is None or pd.isna(mc):
                scores.append(0)
            elif mc > self.LARGE_CAP_THRESHOLD:
                scores.append(-1)
            elif mc < self.SMALL_CAP_THRESHOLD:
                scores.append(1)
            else:
                # Linear interpolation for mid-cap
                scores.append(
                    (self.LARGE_CAP_THRESHOLD - mc)
                    / (self.LARGE_CAP_THRESHOLD - self.SMALL_CAP_THRESHOLD)
                    * 2
                    - 1
                )

        return float(np.dot(scores, self.weights))

    def calculate_value_tilt(self) -> float:
        """
        Calculate portfolio value tilt (HML exposure).

        Returns
        -------
        float
            Value tilt from -1 (growth) to +1 (value)
            0 indicates market-neutral value exposure
        """
        chars = self._fetch_characteristics()

        # Use P/B ratio primarily, P/E as backup
        pb_ratios = chars["pb_ratio"].fillna(chars["pb_ratio"].median())
        pe_ratios = chars["pe_ratio"].fillna(chars["pe_ratio"].median())

        # Score based on valuation: low P/B = value (+1), high P/B = growth (-1)
        # Typical P/B ranges: <1 deep value, 1-3 neutral, >3 growth
        scores = []
        for i, (pb, pe) in enumerate(zip(pb_ratios, pe_ratios)):
            # Use P/B if available, else P/E
            if pd.notna(pb) and pb > 0:
                if pb < 1.5:
                    score = 1.0
                elif pb > 4.0:
                    score = -1.0
                else:
                    # Linear interpolation
                    score = 1.0 - (pb - 1.5) / 2.5 * 2
            elif pd.notna(pe) and pe > 0:
                # P/E based scoring: <15 value, >25 growth
                if pe < 15:
                    score = 1.0
                elif pe > 25:
                    score = -1.0
                else:
                    score = 1.0 - (pe - 15) / 10 * 2
            else:
                score = 0.0
            scores.append(score)

        return float(np.dot(scores, self.weights))

    def calculate_momentum_tilt(self) -> float:
        """
        Calculate portfolio momentum tilt (MOM exposure).

        Returns
        -------
        float
            Momentum tilt from -1 (low momentum) to +1 (high momentum)
        """
        momentum = self._calculate_momentum(lookback_months=12)

        # Normalize momentum scores
        # Market average is roughly 10% annual, winners >20%, losers <0%
        scores = []
        for ticker in self.tickers:
            mom = momentum.get(ticker, 0)
            if mom > 0.2:
                score = 1.0
            elif mom < 0.0:
                score = -1.0
            else:
                # Linear interpolation
                score = mom / 0.2 * 2 - 1
            scores.append(score)

        return float(np.dot(scores, self.weights))

    def calculate_quality_tilt(self) -> float:
        """
        Calculate portfolio quality tilt (RMW-like exposure).

        Quality is based on profitability and financial health.

        Returns
        -------
        float
            Quality tilt from -1 (low quality) to +1 (high quality)
        """
        chars = self._fetch_characteristics()

        scores = []
        for i, ticker in enumerate(self.tickers):
            row = chars.loc[ticker]

            # Score components
            profit_score = 0
            roe_score = 0
            debt_score = 0

            # Profit margin (>15% good, <5% poor)
            pm = row.get("profit_margin")
            if pd.notna(pm):
                if pm > 0.15:
                    profit_score = 1
                elif pm < 0.05:
                    profit_score = -1
                else:
                    profit_score = (pm - 0.05) / 0.10 * 2 - 1

            # ROE (>15% good, <8% poor)
            roe = row.get("roe")
            if pd.notna(roe):
                if roe > 0.15:
                    roe_score = 1
                elif roe < 0.08:
                    roe_score = -1
                else:
                    roe_score = (roe - 0.08) / 0.07 * 2 - 1

            # Debt/Equity (low is better: <50% good, >150% poor)
            de = row.get("debt_to_equity")
            if pd.notna(de):
                de_ratio = de / 100  # Often reported as percentage
                if de_ratio < 0.5:
                    debt_score = 1
                elif de_ratio > 1.5:
                    debt_score = -1
                else:
                    debt_score = 1 - (de_ratio - 0.5) / 1.0 * 2

            # Average available scores
            available = [s for s in [profit_score, roe_score, debt_score] if s != 0]
            scores.append(np.mean(available) if available else 0)

        return float(np.dot(scores, self.weights))

    def calculate_investment_tilt(self) -> float:
        """
        Calculate portfolio investment tilt (CMA-like exposure).

        Conservative investment (low asset growth) vs aggressive.

        Returns
        -------
        float
            Investment tilt from -1 (aggressive) to +1 (conservative)
        """
        chars = self._fetch_characteristics()

        scores = []
        for ticker in self.tickers:
            row = chars.loc[ticker]

            # Use revenue growth as proxy for investment aggressiveness
            rev_growth = row.get("revenue_growth")
            if pd.notna(rev_growth):
                # High growth = aggressive (-1), low growth = conservative (+1)
                if rev_growth > 0.20:
                    score = -1.0
                elif rev_growth < 0.05:
                    score = 1.0
                else:
                    score = 1.0 - (rev_growth - 0.05) / 0.15 * 2
            else:
                score = 0.0
            scores.append(score)

        return float(np.dot(scores, self.weights))

    def get_all_tilts(self) -> Dict[str, float]:
        """
        Calculate all factor tilts for the portfolio.

        Returns
        -------
        dict
            Dictionary with all factor tilts:
            - size: SMB-like exposure
            - value: HML-like exposure
            - momentum: MOM-like exposure
            - quality: RMW-like exposure
            - investment: CMA-like exposure
        """
        return {
            "size": self.calculate_size_tilt(),
            "value": self.calculate_value_tilt(),
            "momentum": self.calculate_momentum_tilt(),
            "quality": self.calculate_quality_tilt(),
            "investment": self.calculate_investment_tilt(),
        }

    def get_characteristics_table(self) -> pd.DataFrame:
        """
        Get a table of fundamental characteristics for all holdings.

        Returns
        -------
        pd.DataFrame
            Characteristics for each holding with portfolio weight
        """
        chars = self._fetch_characteristics().copy()
        chars["weight"] = self.weights

        # Reorder columns
        cols = [
            "weight",
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "dividend_yield",
            "beta",
            "profit_margin",
            "roe",
            "debt_to_equity",
            "revenue_growth",
            "earnings_growth",
        ]
        available_cols = [c for c in cols if c in chars.columns]

        return chars[available_cols]

    def summary(self) -> str:
        """Generate a text summary of factor exposures."""
        tilts = self.get_all_tilts()

        lines = [
            f"\n{'=' * 50}",
            "Portfolio Factor Exposures (Characteristic-Based)",
            f"{'=' * 50}",
            f"{'Factor':<15} {'Tilt':>10} {'Interpretation':<25}",
            f"{'-' * 50}",
        ]

        interpretations = {
            "size": lambda x: (
                "Small Cap" if x > 0.3 else ("Large Cap" if x < -0.3 else "Neutral")
            ),
            "value": lambda x: (
                "Value" if x > 0.3 else ("Growth" if x < -0.3 else "Blend")
            ),
            "momentum": lambda x: (
                "High Mom" if x > 0.3 else ("Low Mom" if x < -0.3 else "Neutral")
            ),
            "quality": lambda x: (
                "High Quality"
                if x > 0.3
                else ("Low Quality" if x < -0.3 else "Neutral")
            ),
            "investment": lambda x: (
                "Conservative" if x > 0.3 else ("Aggressive" if x < -0.3 else "Neutral")
            ),
        }

        for factor, tilt in tilts.items():
            interp = interpretations[factor](tilt)
            lines.append(f"{factor.capitalize():<15} {tilt:>10.2f} {interp:<25}")

        lines.append("=" * 50)
        return "\n".join(lines)
