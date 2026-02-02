"""
Header section for portfolio tear sheet.
"""

from typing import Any, Optional

from jinja2 import Template

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting.sections.base import ReportSection


class HeaderSection(ReportSection):
    """
    Header section containing portfolio overview and key metrics.

    Parameters
    ----------
    portfolio : PortfolioAnalysis
        The portfolio analysis object
    template : jinja2.Template
        The Jinja2 template for this section
    title : str, optional
        Report title
    tickers : list of str, optional
        Ticker symbols for allocation display
    """

    def __init__(
        self,
        portfolio: PortfolioAnalysis,
        template: Template,
        title: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ):
        super().__init__(template)
        self.portfolio = portfolio
        self.title = title or "Portfolio Analysis"
        self.tickers = tickers or list(portfolio.data.columns)

    def compute_data(self) -> dict[str, Any]:
        """Compute header section data."""
        summary = self.portfolio.get_summary()
        cumulative = self.portfolio.calculate_cumulative_returns()

        # Calculate CAGR
        years = (
            self.portfolio.data.index[-1] - self.portfolio.data.index[0]
        ).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

        # Calculate total return
        total_return = cumulative.iloc[-1] - 1

        # Build allocation display
        allocation = [
            {"ticker": ticker, "weight": weight * 100}
            for ticker, weight in zip(self.tickers, self.portfolio.weights)
            if weight > 0.001
        ]

        return {
            "title": self.title,
            "start_date": self.portfolio.data.index[0].strftime("%Y-%m-%d"),
            "end_date": self.portfolio.data.index[-1].strftime("%Y-%m-%d"),
            "allocation": allocation,
            "metrics": {
                "total_return": total_return * 100,
                "cagr": cagr * 100,
                "annual_volatility": summary["annual_volatility"] * 100,
                "sharpe_ratio": summary["sharpe_ratio"],
                "sortino_ratio": summary["sortino_ratio"],
                "max_drawdown": summary["max_drawdown"] * 100,
            },
        }
