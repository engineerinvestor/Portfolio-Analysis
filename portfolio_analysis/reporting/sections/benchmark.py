"""
Benchmark comparison section for portfolio tear sheet.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from portfolio_analysis.metrics.benchmark import BenchmarkComparison
from portfolio_analysis.reporting.chart_utils import create_figure, fig_to_base64
from portfolio_analysis.reporting.sections.base import ReportSection


class BenchmarkSection(ReportSection):
    """
    Benchmark comparison section with performance chart and metrics.

    Parameters
    ----------
    benchmark : BenchmarkComparison
        The benchmark comparison object
    template : jinja2.Template
        The Jinja2 template for this section
    """

    def __init__(self, benchmark: BenchmarkComparison, template: Template):
        super().__init__(template)
        self.benchmark = benchmark

    def _create_comparison_chart(self) -> str:
        """Create portfolio vs benchmark chart and return as base64."""
        portfolio_cum = (1 + self.benchmark.portfolio_returns).cumprod()
        benchmark_cum = (1 + self.benchmark.benchmark_returns).cumprod()

        fig, ax = create_figure(figsize=(12, 5))

        ax.plot(
            portfolio_cum.index,
            portfolio_cum.values,
            linewidth=1.5,
            color="#1f77b4",
            label="Portfolio",
        )
        ax.plot(
            benchmark_cum.index,
            benchmark_cum.values,
            linewidth=1.5,
            color="#ff7f0e",
            label=f"Benchmark ({self.benchmark.benchmark_ticker})",
        )

        ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title("Portfolio vs Benchmark", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Growth of $1", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))

        fig.tight_layout()
        return fig_to_base64(fig)

    def _create_rolling_alpha_beta_chart(self, window: int = 252) -> str:
        """Create rolling alpha and beta chart."""
        rolling_beta = []
        rolling_alpha = []
        dates = []

        portfolio_returns = self.benchmark.portfolio_returns
        benchmark_returns = self.benchmark.benchmark_returns
        rf_daily = self.benchmark.risk_free_rate / 252

        for i in range(window, len(portfolio_returns)):
            port_window = portfolio_returns.iloc[i - window : i]
            bench_window = benchmark_returns.iloc[i - window : i]

            cov = np.cov(port_window, bench_window)[0, 1]
            var = np.var(bench_window)
            beta = cov / var if var > 0 else 0

            alpha = (
                port_window.mean()
                - (rf_daily + beta * (bench_window.mean() - rf_daily))
            ) * 252

            rolling_beta.append(beta)
            rolling_alpha.append(alpha)
            dates.append(portfolio_returns.index[i])

        if len(dates) < 2:
            # Not enough data for rolling window
            return ""

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(dates, rolling_beta, linewidth=1.2, color="#1f77b4")
        axes[0].axhline(y=1.0, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
        axes[0].set_ylabel("Beta", fontsize=10)
        axes[0].set_title(
            f"Rolling {window}-Day Beta and Alpha", fontsize=12, fontweight="bold"
        )
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            dates, [a * 100 for a in rolling_alpha], linewidth=1.2, color="#2ca02c"
        )
        axes[1].axhline(y=0, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
        axes[1].set_ylabel("Alpha (%)", fontsize=10)
        axes[1].set_xlabel("Date", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig_to_base64(fig)

    def compute_data(self) -> dict[str, Any]:
        """Compute benchmark comparison section data."""
        metrics = self.benchmark.get_metrics()

        # Get benchmark name
        benchmark_name = self.benchmark.BENCHMARKS.get(
            self.benchmark.benchmark_ticker, self.benchmark.benchmark_ticker
        )

        # Performance difference
        performance_diff = metrics["portfolio_return"] - metrics["benchmark_return"]

        # Rolling chart
        rolling_chart = self._create_rolling_alpha_beta_chart()

        # Build metrics tables
        return_metrics = [
            {
                "name": "Portfolio Annual Return",
                "value": f"{metrics['portfolio_return'] * 100:.2f}%",
            },
            {
                "name": "Benchmark Annual Return",
                "value": f"{metrics['benchmark_return'] * 100:.2f}%",
            },
            {"name": "Outperformance", "value": f"{performance_diff * 100:+.2f}%"},
        ]

        risk_metrics = [
            {
                "name": "Portfolio Volatility",
                "value": f"{metrics['portfolio_volatility'] * 100:.2f}%",
            },
            {
                "name": "Benchmark Volatility",
                "value": f"{metrics['benchmark_volatility'] * 100:.2f}%",
            },
            {
                "name": "Tracking Error",
                "value": f"{metrics['tracking_error'] * 100:.2f}%",
            },
        ]

        capm_metrics = [
            {"name": "Beta", "value": f"{metrics['beta']:.3f}"},
            {"name": "Alpha (annualized)", "value": f"{metrics['alpha'] * 100:.2f}%"},
            {"name": "R-squared", "value": f"{metrics['r_squared']:.3f}"},
            {"name": "Correlation", "value": f"{metrics['correlation']:.3f}"},
        ]

        performance_metrics = [
            {
                "name": "Information Ratio",
                "value": f"{metrics['information_ratio']:.3f}",
            },
            {"name": "Up Capture", "value": f"{metrics['up_capture']:.1f}%"},
            {"name": "Down Capture", "value": f"{metrics['down_capture']:.1f}%"},
        ]

        return {
            "benchmark_ticker": self.benchmark.benchmark_ticker,
            "benchmark_name": benchmark_name,
            "chart": self._create_comparison_chart(),
            "rolling_chart": rolling_chart,
            "return_metrics": return_metrics,
            "risk_metrics": risk_metrics,
            "capm_metrics": capm_metrics,
            "performance_metrics": performance_metrics,
        }
