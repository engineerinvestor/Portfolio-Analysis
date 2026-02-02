"""
Portfolio Analysis - Professional portfolio analysis tools for DIY investors.

An open-source Python package for analyzing investment portfolios,
running Monte Carlo simulations, and comparing against benchmarks.
"""

__version__ = "0.1.0"
__author__ = "Engineer Investor"

# Core classes - easy imports
from portfolio_analysis.data.loader import DataLoader
from portfolio_analysis.metrics.performance import PerformanceMetrics
from portfolio_analysis.metrics.benchmark import BenchmarkComparison
from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation
from portfolio_analysis.visualization.plots import PortfolioVisualization

# Optional: optimization (requires scipy)
try:
    from portfolio_analysis.analysis.optimization import PortfolioOptimizer
except ImportError:
    PortfolioOptimizer = None

# Optional: interactive widgets (requires ipywidgets)
try:
    from portfolio_analysis.visualization.interactive import (
        InteractivePortfolioAnalyzer,
    )
except ImportError:
    InteractivePortfolioAnalyzer = None

# Optional: HTML reporting (requires jinja2)
try:
    from portfolio_analysis.reporting import ReportBuilder
except ImportError:
    ReportBuilder = None

# Optional: Factor analysis (requires pandas-datareader)
try:
    from portfolio_analysis.factors import (
        FactorDataLoader,
        FactorModel,
        RegressionResults,
        FactorRegression,
        FactorExposures,
        FactorAttribution,
        FactorOptimizer,
        FactorVisualization,
    )
except ImportError:
    FactorDataLoader = None
    FactorModel = None
    RegressionResults = None
    FactorRegression = None
    FactorExposures = None
    FactorAttribution = None
    FactorOptimizer = None
    FactorVisualization = None

__all__ = [
    "DataLoader",
    "PerformanceMetrics",
    "BenchmarkComparison",
    "PortfolioAnalysis",
    "MonteCarloSimulation",
    "PortfolioVisualization",
    "PortfolioOptimizer",
    "InteractivePortfolioAnalyzer",
    "ReportBuilder",
    "FactorDataLoader",
    "FactorModel",
    "RegressionResults",
    "FactorRegression",
    "FactorExposures",
    "FactorAttribution",
    "FactorOptimizer",
    "FactorVisualization",
]
