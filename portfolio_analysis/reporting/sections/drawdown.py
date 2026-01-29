"""
Drawdown section for portfolio tear sheet.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting.chart_utils import fig_to_base64, create_figure
from portfolio_analysis.reporting.sections.base import ReportSection


class DrawdownSection(ReportSection):
    """
    Drawdown section with underwater chart and worst periods table.

    Parameters
    ----------
    portfolio : PortfolioAnalysis
        The portfolio analysis object
    template : jinja2.Template
        The Jinja2 template for this section
    top_n : int, default 5
        Number of worst drawdowns to display
    """

    def __init__(
        self,
        portfolio: PortfolioAnalysis,
        template: Template,
        top_n: int = 5
    ):
        super().__init__(template)
        self.portfolio = portfolio
        self.top_n = top_n

    def _calculate_drawdown_series(self) -> pd.Series:
        """Calculate the drawdown series."""
        cumulative = self.portfolio.calculate_cumulative_returns()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown

    def _find_worst_drawdowns(self) -> List[Dict[str, Any]]:
        """Find the worst drawdown periods."""
        cumulative = self.portfolio.calculate_cumulative_returns()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1

        # Find drawdown periods
        is_dd = drawdown < 0
        starts = is_dd & ~is_dd.shift(1, fill_value=False)
        ends = ~is_dd & is_dd.shift(1, fill_value=False)

        start_dates = cumulative.index[starts].tolist()
        end_dates = cumulative.index[ends].tolist()

        # Handle ongoing drawdown
        if len(start_dates) > len(end_dates):
            end_dates.append(cumulative.index[-1])

        periods = []
        for start, end in zip(start_dates, end_dates):
            period_dd = drawdown[start:end]
            if len(period_dd) == 0:
                continue

            trough_date = period_dd.idxmin()
            trough_value = period_dd.min()

            # Find recovery date (if recovered)
            recovery_mask = drawdown[trough_date:] >= 0
            if recovery_mask.any():
                recovery_date = drawdown[trough_date:][recovery_mask].index[0]
                recovery_days = (recovery_date - trough_date).days
            else:
                recovery_date = None
                recovery_days = None

            periods.append({
                "start": start.strftime("%Y-%m-%d"),
                "trough": trough_date.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d") if recovery_date else "Ongoing",
                "drawdown": trough_value * 100,
                "days_to_trough": (trough_date - start).days,
                "recovery_days": recovery_days,
            })

        # Sort by drawdown magnitude and take top N
        periods.sort(key=lambda x: x["drawdown"])
        return periods[:self.top_n]

    def _create_drawdown_chart(self) -> str:
        """Create underwater/drawdown chart and return as base64."""
        drawdown = self._calculate_drawdown_series()

        fig, ax = create_figure(figsize=(12, 4))

        ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                        color="#d62728", alpha=0.6)
        ax.plot(drawdown.index, drawdown.values * 100,
                color="#8b0000", linewidth=0.8)

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)

        ax.set_title("Underwater Plot (Drawdown)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        min_dd = drawdown.min() * 100
        ax.set_ylim(min_dd * 1.1, 5)

        fig.tight_layout()
        return fig_to_base64(fig)

    def compute_data(self) -> Dict[str, Any]:
        """Compute drawdown section data."""
        drawdown = self._calculate_drawdown_series()
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        return {
            "chart": self._create_drawdown_chart(),
            "max_drawdown": max_dd * 100,
            "max_drawdown_date": max_dd_date.strftime("%Y-%m-%d"),
            "current_drawdown": drawdown.iloc[-1] * 100,
            "worst_periods": self._find_worst_drawdowns(),
        }
