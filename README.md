# Portfolio-Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://engineer-investor-portfolio-analysis.streamlit.app/)
[![PyPI version](https://badge.fury.io/py/engineer-investor-portfolio.svg)](https://badge.fury.io/py/engineer-investor-portfolio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source portfolio analysis tools for DIY investors and finance enthusiasts. This repository aims to provide a comprehensive suite of tools to analyze and optimize investment portfolios, with an emphasis on transparency, flexibility, and extensibility.

## Quick Start

### Try Online (No Installation)

**Google Colab Notebooks:**
- [Basic Portfolio Analysis](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Basic_Portfolio_Analysis.ipynb) - Core analysis tutorial
- [Interactive Portfolio Analysis](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Interactive_Portfolio_Analysis.ipynb) - Widget-based interface

### Install as Python Package

```bash
pip install engineer-investor-portfolio
```

Or install from source with all features:

```bash
git clone https://github.com/engineerinvestor/Portfolio-Analysis.git
cd Portfolio-Analysis
pip install -e ".[all]"
```

### Run the Streamlit Web App

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app/app.py
```

### Or Use the Live App
[**Launch Streamlit App**](https://engineer-investor-portfolio-analysis.streamlit.app/)


## Features

### Core Analysis
- **Performance Metrics**: Annual return, volatility, Sharpe ratio, Sortino ratio, max drawdown, VaR
- **Portfolio Analysis**: Weighted returns, covariance-based volatility, cumulative returns
- **Monte Carlo Simulation**: Project future portfolio values with confidence intervals
- **Benchmark Comparison**: Alpha, beta, tracking error, information ratio, capture ratios

### Optimization
- Maximum Sharpe Ratio portfolio
- Minimum Volatility portfolio
- Risk Parity (equal risk contribution)
- Target Return optimization
- Efficient Frontier visualization

### HTML Tear Sheet Reports
- Professional portfolio reports inspired by [QuantStats](https://github.com/ranaroussi/quantstats)
- Embedded charts (cumulative returns, drawdowns, monthly heatmap)
- Comprehensive metrics tables (risk, return, ratios)
- Optional benchmark comparison section
- Self-contained HTML files with no external dependencies

### Interactive Tools
- Jupyter widgets for Colab/notebook analysis
- Streamlit web application
- Preset portfolios (60/40, Three-Fund, All-Weather, etc.)

## Usage Examples

### Python Package

```python
from portfolio_analysis import DataLoader, PortfolioAnalysis, MonteCarloSimulation

# Load data
loader = DataLoader(['VTI', 'VXUS', 'BND'], '2018-01-01', '2024-01-01')
data = loader.fetch_data()

# Analyze portfolio
portfolio = PortfolioAnalysis(data, weights=[0.4, 0.2, 0.4])
portfolio.print_summary()

# Run Monte Carlo simulation
mc = MonteCarloSimulation(data, weights=[0.4, 0.2, 0.4], num_simulations=1000)
mc.print_summary()
mc.plot_simulation()
```

### Optimization

```python
from portfolio_analysis import PortfolioOptimizer

optimizer = PortfolioOptimizer(data, risk_free_rate=0.04)
optimal = optimizer.optimize_max_sharpe()
print(f"Optimal weights: {optimal['weights']}")

# Visualize efficient frontier
optimizer.plot_efficient_frontier()
```

### HTML Tear Sheet

```python
from portfolio_analysis import DataLoader, PortfolioAnalysis, BenchmarkComparison
from portfolio_analysis.reporting import ReportBuilder

# Load data and create portfolio
loader = DataLoader(['VTI', 'BND'], '2020-01-01', '2024-01-01')
data = loader.fetch_data()
portfolio = PortfolioAnalysis(data, [0.6, 0.4])

# Generate basic report
report = ReportBuilder(portfolio, title="60/40 Portfolio")
report.generate("tearsheet.html")

# With benchmark comparison
benchmark = BenchmarkComparison(data, [0.6, 0.4], benchmark_ticker='SPY')
report = ReportBuilder(portfolio, benchmark=benchmark, title="60/40 Portfolio")
report.generate("tearsheet_with_benchmark.html")
```

## Repository Structure

```
Portfolio-Analysis/
├── portfolio_analysis/          # Python package
│   ├── data/                    # Data loading
│   ├── metrics/                 # Performance & benchmark metrics
│   ├── analysis/                # Portfolio & Monte Carlo analysis
│   ├── visualization/           # Plotting & interactive widgets
│   ├── reporting/               # HTML tear sheet generation
│   └── utils/                   # Helper functions
├── streamlit_app/               # Web application
│   └── app.py                   # Main Streamlit app
├── tests/                       # pytest test suite
├── Tutorials/                   # Additional notebooks
├── Visualization/               # Visualization notebooks
├── Basic_Portfolio_Analysis.ipynb
├── Interactive_Portfolio_Analysis.ipynb
├── pyproject.toml               # Package configuration
└── requirements*.txt            # Dependencies
```

## Installation Options

### Minimal (Core Analysis)
```bash
pip install engineer-investor-portfolio
```

### With Optimization
```bash
pip install "engineer-investor-portfolio[optimization]"
```

### With Interactive Widgets
```bash
pip install "engineer-investor-portfolio[interactive]"
```

### For Streamlit App
```bash
pip install "engineer-investor-portfolio[streamlit]"
```

### Full Development
```bash
pip install "engineer-investor-portfolio[all]"
```

## Contributing

We welcome contributions from the community. Please read the following guidelines:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a Pull Request

See also: [Beginner's Guide to Contributing](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/Tutorials/Beginner's_Guide_to_Contributing_to_the_Portfolio_Analysis_Repository.ipynb)

## Roadmap

- [x] Core portfolio analysis classes
- [x] Monte Carlo simulation (fixed)
- [x] Benchmark comparison
- [x] Interactive widgets for Colab
- [x] Python package (PyPI-ready)
- [x] Portfolio optimization
- [x] Streamlit web application
- [x] HTML tear sheet reports
- [ ] Factor analysis (Fama-French, momentum, quality)
- [ ] Time-varying risk-free rate (T-bill data)
- [ ] Tax-loss harvesting tools
- [ ] Comprehensive test coverage

## Related Resources

- [Awesome Quant](https://github.com/HugoDelatte/awesome-quant) - Curated quant finance resources
- [QuantStats](https://github.com/ranaroussi/quantstats) - Portfolio analytics
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - Portfolio optimization
- [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) - Portfolio optimization and risk management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This is not investment advice.** These tools are for educational purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting a qualified financial advisor.

## Contact

- Twitter: [@egr_investor](https://twitter.com/egr_investor)
- GitHub: [engineerinvestor](https://github.com/engineerinvestor)
- Email: egr.investor (gmail)
