"""
Report sections for HTML tear sheet generation.

Each section class generates a specific part of the portfolio report.
"""

from portfolio_analysis.reporting.sections.base import ReportSection
from portfolio_analysis.reporting.sections.header import HeaderSection
from portfolio_analysis.reporting.sections.performance import PerformanceSection
from portfolio_analysis.reporting.sections.drawdown import DrawdownSection
from portfolio_analysis.reporting.sections.returns import ReturnsSection
from portfolio_analysis.reporting.sections.risk import RiskSection
from portfolio_analysis.reporting.sections.benchmark import BenchmarkSection

__all__ = [
    "ReportSection",
    "HeaderSection",
    "PerformanceSection",
    "DrawdownSection",
    "ReturnsSection",
    "RiskSection",
    "BenchmarkSection",
]
