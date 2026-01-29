"""
Performance section for portfolio tear sheet.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting.chart_utils import fig_to_base64, create_figure
from portfolio_analysis.reporting.sections.base import ReportSection


class PerformanceSection(ReportSection):
    """
    Performance section with cumulative returns chart.

    Parameters
    ----------
    portfolio : PortfolioAnalysis
        The portfolio analysis object
    template : jinja2.Template
        The Jinja2 template for this section
    """

    def __init__(self, portfolio: PortfolioAnalysis, template: Template):
        super().__init__(template)
        self.portfolio = portfolio

    def _create_cumulative_returns_chart(self) -> str:
        """Create cumulative returns chart and return as base64."""
        cumulative = self.portfolio.calculate_cumulative_returns()

        fig, ax = create_figure(figsize=(12, 5))

        ax.plot(cumulative.index, cumulative.values, linewidth=1.5, color="#1f77b4")
        ax.fill_between(cumulative.index, 1, cumulative.values,
                        alpha=0.2, color="#1f77b4")

        ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title("Cumulative Returns (Growth of $1)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))

        fig.tight_layout()
        return fig_to_base64(fig)

    def compute_data(self) -> Dict[str, Any]:
        """Compute performance section data."""
        returns = self.portfolio.calculate_portfolio_returns()
        cumulative = self.portfolio.calculate_cumulative_returns()

        # Calculate best and worst days
        best_day = returns.max()
        best_day_date = returns.idxmax()
        worst_day = returns.min()
        worst_day_date = returns.idxmin()

        # Total return
        total_return = cumulative.iloc[-1] - 1

        return {
            "chart": self._create_cumulative_returns_chart(),
            "total_return": total_return * 100,
            "best_day": best_day * 100,
            "best_day_date": best_day_date.strftime("%Y-%m-%d"),
            "worst_day": worst_day * 100,
            "worst_day_date": worst_day_date.strftime("%Y-%m-%d"),
            "positive_days": (returns > 0).sum(),
            "negative_days": (returns < 0).sum(),
            "positive_pct": (returns > 0).mean() * 100,
        }
