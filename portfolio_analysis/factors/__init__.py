"""
Factor Analysis Module - Fama-French factor analysis for portfolio evaluation.

This module provides tools for:
- Loading Fama-French factor data (FF3, FF5, Carhart)
- Running factor regressions on portfolio returns
- Calculating characteristic-based factor exposures
- Decomposing returns into factor contributions
- Factor-aware portfolio optimization
"""

from portfolio_analysis.factors.attribution import FactorAttribution
from portfolio_analysis.factors.composite import (
    CompositeFactorRegression,
    CompositeRegressionResults,
)
from portfolio_analysis.factors.data import FactorDataLoader
from portfolio_analysis.factors.exposures import FactorExposures
from portfolio_analysis.factors.models import (
    FactorModel,
    FactorRegression,
    RegressionResults,
)
from portfolio_analysis.factors.optimization import FactorOptimizer
from portfolio_analysis.factors.visualization import FactorVisualization

__all__ = [
    "FactorDataLoader",
    "FactorModel",
    "RegressionResults",
    "FactorRegression",
    "FactorExposures",
    "FactorAttribution",
    "FactorOptimizer",
    "FactorVisualization",
    "CompositeFactorRegression",
    "CompositeRegressionResults",
]
