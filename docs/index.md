# Portfolio Analysis

<p align="center">
  <strong>Professional portfolio analysis tools for systematic factor investing</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/engineer-investor-portfolio/"><img src="https://badge.fury.io/py/engineer-investor-portfolio.svg" alt="PyPI version"></a>
  <a href="https://github.com/engineerinvestor/Portfolio-Analysis/actions"><img src="https://github.com/engineerinvestor/Portfolio-Analysis/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/engineerinvestor/Portfolio-Analysis"><img src="https://codecov.io/gh/engineerinvestor/Portfolio-Analysis/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

## What is Portfolio Analysis?

**Portfolio Analysis** is an open-source Python package that provides institutional-quality tools for:

- **Performance Measurement** - Sharpe, Sortino, max drawdown, VaR, and more
- **Factor Analysis** - Fama-French regressions, return attribution, factor tilts
- **Portfolio Optimization** - Mean-variance, risk parity, factor-aware optimization
- **Professional Reporting** - HTML tear sheets, interactive dashboards

Whether you're a DIY investor analyzing your 401(k) or a quant building factor strategies, this package has you covered.

---

## Quick Example

```python
from portfolio_analysis import DataLoader, PortfolioAnalysis
from portfolio_analysis.factors import FactorDataLoader, FactorRegression

# Load your portfolio
loader = DataLoader(['VTI', 'VXUS', 'BND'], '2019-01-01', '2024-01-01')
data = loader.fetch_data()

# Analyze performance
portfolio = PortfolioAnalysis(data, weights=[0.5, 0.2, 0.3])
portfolio.print_summary()

# Run factor analysis
factor_loader = FactorDataLoader()
ff3 = factor_loader.get_ff3_factors('2019-01-01', '2024-01-01')

regression = FactorRegression(portfolio.calculate_portfolio_returns(), ff3)
results = regression.run_regression('ff3')
print(results.summary())
```

---

## Key Features

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Performance Metrics**

    ---

    Calculate 20+ performance metrics including Sharpe ratio, Sortino ratio,
    max drawdown, VaR, CAGR, and capture ratios.

    [:octicons-arrow-right-24: Learn more](user-guide/portfolio-analysis.md)

-   :material-scale-balance:{ .lg .middle } **Factor Analysis**

    ---

    Decompose returns using CAPM, Fama-French 3/5 factor, and Carhart models.
    Understand your true alpha vs systematic factor exposure.

    [:octicons-arrow-right-24: Learn more](user-guide/factor-models.md)

-   :material-target:{ .lg .middle } **Portfolio Optimization**

    ---

    Find optimal portfolios using max Sharpe, min volatility, risk parity,
    or custom factor targets.

    [:octicons-arrow-right-24: Learn more](user-guide/optimization.md)

-   :material-file-document:{ .lg .middle } **Professional Reports**

    ---

    Generate beautiful HTML tear sheets with embedded charts,
    comprehensive metrics, and benchmark comparisons.

    [:octicons-arrow-right-24: Learn more](user-guide/reporting.md)

</div>

---

## Installation

=== "pip"

    ```bash
    pip install engineer-investor-portfolio
    ```

=== "With factor analysis"

    ```bash
    pip install "engineer-investor-portfolio[factors]"
    ```

=== "Full installation"

    ```bash
    pip install "engineer-investor-portfolio[all]"
    ```

=== "From source"

    ```bash
    git clone https://github.com/engineerinvestor/Portfolio-Analysis.git
    cd Portfolio-Analysis
    pip install -e ".[all]"
    ```

---

## Why Portfolio Analysis?

| Feature | QuantStats | PyPortfolioOpt | Riskfolio | **Portfolio Analysis** |
|---------|:----------:|:--------------:|:---------:|:----------------------:|
| Return factor regressions (CAPM/FF/Carhart) + attribution | ❌ | ❌ | ❌ | ✅ |
| HTML Tear Sheets | ✅ | ❌ | ❌ | ✅ |
| Optimization | ❌ | ✅ | ✅ | ✅ |
| Interactive Widgets | ❌ | ❌ | ❌ | ✅ |
| Streamlit App | ❌ | ❌ | ❌ | ✅ |
| Beginner Friendly | ⚠️ | ⚠️ | ❌ | ✅ |

---

## Try It Now

No installation required! Open in Google Colab:

- [Basic Portfolio Analysis](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Basic_Portfolio_Analysis.ipynb)
- [Factor Analysis Demo](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Factor_Analysis_Demo.ipynb)
- [Interactive Analysis](https://colab.research.google.com/github/engineerinvestor/Portfolio-Analysis/blob/main/Interactive_Portfolio_Analysis.ipynb)

Or try the [**Streamlit Web App**](https://engineer-investor-portfolio-analysis.streamlit.app/) →

---

## Community

- **GitHub**: [engineerinvestor/Portfolio-Analysis](https://github.com/engineerinvestor/Portfolio-Analysis)
- **Twitter**: [@egr_investor](https://twitter.com/egr_investor)
- **Issues**: [Report bugs or request features](https://github.com/engineerinvestor/Portfolio-Analysis/issues)

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/engineerinvestor/Portfolio-Analysis/blob/main/LICENSE) for details.

!!! warning "Disclaimer"
    This is not investment advice. These tools are for educational purposes only.
    Past performance does not guarantee future results.
