"""
Returns section with monthly heatmap for portfolio tear sheet.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting.chart_utils import fig_to_base64, create_figure
from portfolio_analysis.reporting.sections.base import ReportSection


class ReturnsSection(ReportSection):
    """
    Returns section with monthly returns heatmap.

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

    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns as a year x month matrix."""
        returns = self.portfolio.calculate_portfolio_returns()

        # Resample to monthly returns (use 'M' for pandas < 2.2 compatibility)
        monthly = (1 + returns).resample("M").prod() - 1

        # Create year-month matrix
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values
        })

        # Pivot to matrix format
        matrix = monthly_df.pivot(index="year", columns="month", values="return")

        # Rename columns to month abbreviations
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        matrix.columns = [month_names[m-1] for m in matrix.columns]

        return matrix

    def _create_heatmap_chart(self) -> str:
        """Create monthly returns heatmap and return as base64."""
        matrix = self._calculate_monthly_returns()

        # Determine figure size based on data
        n_years = len(matrix.index)
        fig_height = max(4, min(n_years * 0.5 + 1, 10))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Create heatmap
        data = matrix.values * 100
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                       vmin=-10, vmax=10)

        # Set ticks
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticklabels(matrix.index)

        # Add text annotations
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = data[i, j]
                if not np.isnan(value):
                    text_color = "white" if abs(value) > 5 else "black"
                    ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                            color=text_color, fontsize=8)

        ax.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Return (%)", fontsize=10)

        fig.tight_layout()
        return fig_to_base64(fig)

    def compute_data(self) -> Dict[str, Any]:
        """Compute returns section data."""
        returns = self.portfolio.calculate_portfolio_returns()
        # Use 'M' and 'Y' for pandas < 2.2 compatibility
        monthly = (1 + returns).resample("M").prod() - 1

        # Calculate annual returns
        annual = (1 + returns).resample("Y").prod() - 1

        best_month = monthly.max()
        best_month_date = monthly.idxmax()
        worst_month = monthly.min()
        worst_month_date = monthly.idxmin()

        # Calculate win rate
        positive_months = (monthly > 0).sum()
        total_months = len(monthly)
        win_rate = positive_months / total_months * 100 if total_months > 0 else 0

        # Average monthly return
        avg_monthly = monthly.mean()

        return {
            "chart": self._create_heatmap_chart(),
            "best_month": best_month * 100,
            "best_month_date": best_month_date.strftime("%b %Y"),
            "worst_month": worst_month * 100,
            "worst_month_date": worst_month_date.strftime("%b %Y"),
            "avg_monthly_return": avg_monthly * 100,
            "positive_months": positive_months,
            "total_months": total_months,
            "win_rate": win_rate,
            "annual_returns": [
                {"year": idx.year, "return": val * 100}
                for idx, val in annual.items()
            ],
        }
