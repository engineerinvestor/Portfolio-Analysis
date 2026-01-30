# API Reference

Complete API documentation for Portfolio Analysis.

## Core Modules

| Module | Description |
|--------|-------------|
| [`portfolio_analysis.data`](data.md) | Data loading from Yahoo Finance |
| [`portfolio_analysis.metrics`](metrics.md) | Performance and benchmark metrics |
| [`portfolio_analysis.analysis`](analysis.md) | Portfolio and Monte Carlo analysis |
| [`portfolio_analysis.factors`](factors.md) | Factor models and attribution |
| [`portfolio_analysis.optimization`](optimization.md) | Portfolio optimization |
| [`portfolio_analysis.visualization`](visualization.md) | Plotting functions |

## Quick Import

```python
# Core classes
from portfolio_analysis import (
    DataLoader,
    PortfolioAnalysis,
    PerformanceMetrics,
    BenchmarkComparison,
    MonteCarloSimulation,
    PortfolioOptimizer,
    PortfolioVisualization,
)

# Factor analysis
from portfolio_analysis.factors import (
    FactorDataLoader,
    FactorRegression,
    FactorAttribution,
    FactorExposures,
    FactorOptimizer,
    FactorVisualization,
)

# Reporting
from portfolio_analysis.reporting import ReportBuilder
```
