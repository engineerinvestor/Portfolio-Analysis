# Reporting

Generate professional HTML tear sheet reports.

## Basic Report

```python
from portfolio_analysis import PortfolioAnalysis
from portfolio_analysis.reporting import ReportBuilder

portfolio = PortfolioAnalysis(prices, weights)

report = ReportBuilder(portfolio, title="My Portfolio")
report.generate("tearsheet.html")
```

## With Benchmark

```python
from portfolio_analysis import BenchmarkComparison

benchmark = BenchmarkComparison(prices, weights, benchmark_ticker='SPY')

report = ReportBuilder(
    portfolio,
    benchmark=benchmark,
    title="Portfolio vs S&P 500"
)
report.generate("tearsheet_benchmark.html")
```

## Report Contents

The generated HTML includes:

- **Header**: Title, date range, key metrics
- **Performance Chart**: Cumulative returns vs benchmark
- **Drawdown Chart**: Underwater plot
- **Monthly Returns Heatmap**: Year x Month grid
- **Metrics Tables**: Risk, return, and ratio statistics
- **Benchmark Comparison**: Alpha, beta, capture ratios

## Customization

Reports are self-contained HTML files with embedded CSS and base64-encoded charts. No external dependencies required.
