# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-source portfolio analysis tools for DIY investors. The repository consists of:
- **Python package** (`portfolio_analysis/`) - Installable via pip
- **Jupyter notebooks** - For Google Colab tutorials
- **Streamlit web app** (`streamlit_app/`) - Interactive web interface

## Setup

### For Development
```bash
pip install -e ".[all]"
```

### For Notebooks
```bash
pip install -r requirements.txt
```

### For Streamlit App
```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app/app.py
```

## Repository Structure

```
Portfolio-Analysis/
├── portfolio_analysis/          # Python package
│   ├── __init__.py             # Package exports
│   ├── data/
│   │   └── loader.py           # DataLoader class
│   ├── metrics/
│   │   ├── performance.py      # PerformanceMetrics class
│   │   └── benchmark.py        # BenchmarkComparison class
│   ├── analysis/
│   │   ├── portfolio.py        # PortfolioAnalysis class
│   │   ├── montecarlo.py       # MonteCarloSimulation class
│   │   └── optimization.py     # PortfolioOptimizer class
│   ├── visualization/
│   │   ├── plots.py            # PortfolioVisualization class
│   │   └── interactive.py      # InteractivePortfolioAnalyzer class
│   └── utils/
│       └── helpers.py          # Utility functions
├── streamlit_app/
│   └── app.py                  # Main Streamlit application
├── tests/
│   └── test_portfolio.py       # pytest test suite
├── Tutorials/                  # Additional tutorial notebooks
├── Visualization/              # Visualization-focused notebooks
├── Basic_Portfolio_Analysis.ipynb    # Main tutorial notebook
├── Interactive_Portfolio_Analysis.ipynb  # Widget-based notebook
├── pyproject.toml              # Package configuration
├── setup.py                    # Legacy setup support
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
└── requirements-streamlit.txt  # Streamlit dependencies
```

## Core Classes

### Data Loading
- **DataLoader** (`data/loader.py`): Fetches historical price data via yfinance

### Performance Metrics
- **PerformanceMetrics** (`metrics/performance.py`): Static methods for annual return, volatility, Sharpe ratio, Sortino ratio, max drawdown, VaR, CAGR, Calmar ratio
- **BenchmarkComparison** (`metrics/benchmark.py`): Alpha, beta, tracking error, information ratio, capture ratios

### Analysis
- **PortfolioAnalysis** (`analysis/portfolio.py`): Weighted portfolio return, volatility, Sharpe ratio
- **MonteCarloSimulation** (`analysis/montecarlo.py`): Portfolio projections with percentile bands
- **PortfolioOptimizer** (`analysis/optimization.py`): Max Sharpe, min volatility, risk parity, efficient frontier

### Visualization
- **PortfolioVisualization** (`visualization/plots.py`): Cumulative returns, allocation pie, drawdown plots
- **InteractivePortfolioAnalyzer** (`visualization/interactive.py`): ipywidgets-based interface

## Key Patterns

- Portfolios are defined as dictionaries mapping ticker symbols to weights (must sum to 1.0)
- Data is fetched using `yf.download()` for adjusted close prices
- Returns are calculated using `pct_change()` on price data
- Annual volatility uses 252 trading days: `daily_std * sqrt(252)`
- All classes support both single assets (Series) and multiple assets (DataFrame)

## Testing

```bash
pytest tests/ -v
```

## Commands

```bash
# Install in development mode
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Run Streamlit app
streamlit run streamlit_app/app.py

# Format code
black portfolio_analysis/

# Lint code
ruff check portfolio_analysis/
```

## Coding Standards

- Use type hints for function parameters and returns
- Include docstrings with Parameters/Returns sections (NumPy style)
- Validate inputs (weights sum to 1.0, matching array lengths)
- Handle single-ticker edge cases (Series vs DataFrame)
- Use constants for magic numbers (TRADING_DAYS = 252)

## Git Commit Attribution

All commits must be attributed to:
- **Name:** Engineer Investor
- **Email:** egr.investor@gmail.com
