"""
Risk metrics section for portfolio tear sheet.
"""

from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template
from scipy import stats

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting.sections.base import ReportSection


class RiskSection(ReportSection):
    """
    Risk metrics section with comprehensive statistics tables.

    Parameters
    ----------
    portfolio : PortfolioAnalysis
        The portfolio analysis object
    template : jinja2.Template
        The Jinja2 template for this section
    risk_free_rate : float, default 0.02
        Annual risk-free rate for calculations
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        portfolio: PortfolioAnalysis,
        template: Template,
        risk_free_rate: float = 0.02,
    ):
        super().__init__(template)
        self.portfolio = portfolio
        self.risk_free_rate = risk_free_rate

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def compute_data(self) -> dict[str, Any]:
        """Compute risk metrics section data."""
        returns = self.portfolio.calculate_portfolio_returns()
        cumulative = self.portfolio.calculate_cumulative_returns()

        # Calculate years
        years = (
            self.portfolio.data.index[-1] - self.portfolio.data.index[0]
        ).days / 365.25

        # CAGR
        cagr = (cumulative.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

        # Basic stats
        summary = self.portfolio.get_summary(self.risk_free_rate)

        # Distribution stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # VaR and CVaR
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)

        # Calmar ratio
        max_dd = abs(summary["max_drawdown"])
        calmar = cagr / max_dd if max_dd > 0 else np.inf

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.TRADING_DAYS)

        # Average gain/loss
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0

        # Gain/loss ratio
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf

        # Risk metrics table
        risk_metrics = [
            {
                "name": "Annual Volatility",
                "value": f"{summary['annual_volatility'] * 100:.2f}%",
            },
            {"name": "Max Drawdown", "value": f"{summary['max_drawdown'] * 100:.2f}%"},
            {"name": "VaR (95%)", "value": f"{var_95 * 100:.2f}%"},
            {"name": "CVaR (95%)", "value": f"{cvar_95 * 100:.2f}%"},
            {"name": "VaR (99%)", "value": f"{var_99 * 100:.2f}%"},
            {"name": "Downside Deviation", "value": f"{downside_deviation * 100:.2f}%"},
        ]

        # Return metrics table
        return_metrics = [
            {
                "name": "Total Return",
                "value": f"{(cumulative.iloc[-1] - 1) * 100:.2f}%",
            },
            {"name": "CAGR", "value": f"{cagr * 100:.2f}%"},
            {
                "name": "Annual Return",
                "value": f"{summary['annual_return'] * 100:.2f}%",
            },
            {"name": "Avg Daily Return", "value": f"{returns.mean() * 100:.4f}%"},
            {"name": "Avg Gain (daily)", "value": f"{avg_gain * 100:.4f}%"},
            {"name": "Avg Loss (daily)", "value": f"{avg_loss * 100:.4f}%"},
        ]

        # Ratio metrics table
        ratio_metrics = [
            {"name": "Sharpe Ratio", "value": f"{summary['sharpe_ratio']:.2f}"},
            {"name": "Sortino Ratio", "value": f"{summary['sortino_ratio']:.2f}"},
            {"name": "Calmar Ratio", "value": f"{calmar:.2f}"},
            {"name": "Gain/Loss Ratio", "value": f"{gain_loss_ratio:.2f}"},
            {"name": "Skewness", "value": f"{skewness:.2f}"},
            {"name": "Kurtosis", "value": f"{kurtosis:.2f}"},
        ]

        return {
            "risk_metrics": risk_metrics,
            "return_metrics": return_metrics,
            "ratio_metrics": ratio_metrics,
        }
