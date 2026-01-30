# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Factor analysis module with Fama-French models
- `FactorDataLoader` for fetching factor data from Kenneth French Data Library
- `FactorRegression` for CAPM, FF3, FF5, and Carhart regressions
- `FactorAttribution` for return and risk decomposition
- `FactorExposures` for characteristic-based factor tilts
- `FactorOptimizer` for factor-aware portfolio optimization
- `FactorVisualization` for factor analysis plots
- Factor Analysis Demo notebook
- MkDocs documentation site
- GitHub Actions CI/CD pipeline

## [0.1.0] - 2024-01-01

### Added
- Initial release
- `DataLoader` for fetching data from Yahoo Finance
- `PortfolioAnalysis` for portfolio metrics
- `PerformanceMetrics` for individual asset metrics
- `BenchmarkComparison` for alpha, beta, and capture ratios
- `MonteCarloSimulation` for portfolio projections
- `PortfolioOptimizer` for mean-variance optimization
- `PortfolioVisualization` for plotting
- `ReportBuilder` for HTML tear sheets
- Streamlit web application
- Jupyter notebook tutorials
