# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from PyPI

The simplest way to install Portfolio Analysis:

```bash
pip install engineer-investor-portfolio
```

## Optional Dependencies

Portfolio Analysis has optional features that require additional packages:

### Factor Analysis

For Fama-French factor models and return attribution:

```bash
pip install "engineer-investor-portfolio[factors]"
```

This installs `pandas-datareader` for fetching factor data from Kenneth French's Data Library.

### Portfolio Optimization

For mean-variance optimization and efficient frontier:

```bash
pip install "engineer-investor-portfolio[optimization]"
```

This installs `scipy` for numerical optimization.

### Interactive Widgets

For Jupyter notebook widgets:

```bash
pip install "engineer-investor-portfolio[interactive]"
```

This installs `ipywidgets` for interactive portfolio analysis.

### Streamlit App

For running the web application locally:

```bash
pip install "engineer-investor-portfolio[streamlit]"
```

### Full Installation

Install everything:

```bash
pip install "engineer-investor-portfolio[all]"
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/engineerinvestor/Portfolio-Analysis.git
cd Portfolio-Analysis
pip install -e ".[all]"
```

## Verify Installation

```python
import portfolio_analysis
print(portfolio_analysis.__version__)
```

## Troubleshooting

### yfinance Issues

If you encounter issues with `yfinance`, try upgrading:

```bash
pip install --upgrade yfinance
```

### pandas-datareader Issues

For factor data loading issues:

```bash
pip install --upgrade pandas-datareader
```

### Apple Silicon (M1/M2)

On Apple Silicon Macs, you may need:

```bash
pip install --upgrade numpy scipy
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [Tutorials](tutorials.md) - Step-by-step examples
