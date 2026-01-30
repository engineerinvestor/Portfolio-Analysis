# Quick Start

Get up and running with Portfolio Analysis in 5 minutes.

## Basic Portfolio Analysis

```python
from portfolio_analysis import DataLoader, PortfolioAnalysis

# Define your portfolio
tickers = ['VTI', 'VXUS', 'BND']  # Total US, International, Bonds
weights = [0.5, 0.2, 0.3]         # 50/20/30 allocation

# Load historical data
loader = DataLoader(tickers, '2019-01-01', '2024-01-01')
data = loader.fetch_data()

# Create portfolio and analyze
portfolio = PortfolioAnalysis(data, weights)
portfolio.print_summary()
```

**Output:**
```
============================================================
PORTFOLIO SUMMARY
============================================================
Annual Return:           8.45%
Annual Volatility:      12.34%
Sharpe Ratio:            0.52
Sortino Ratio:           0.78
Max Drawdown:          -23.45%
============================================================
```

## Visualize Performance

```python
from portfolio_analysis import PortfolioVisualization

# Cumulative returns chart
PortfolioVisualization.plot_portfolio_return(data, weights)

# Drawdown chart
PortfolioVisualization.plot_drawdown(data, weights)

# Asset allocation pie chart
PortfolioVisualization.plot_allocation_pie(weights, tickers)
```

## Compare to Benchmark

```python
from portfolio_analysis import BenchmarkComparison

# Compare to S&P 500
benchmark = BenchmarkComparison(data, weights, benchmark_ticker='SPY')
benchmark.print_comparison()
```

**Output:**
```
============================================================
BENCHMARK COMPARISON (vs SPY)
============================================================
                      Portfolio    Benchmark
Annual Return:           8.45%       10.23%
Annual Volatility:      12.34%       18.56%
Sharpe Ratio:            0.52         0.45
Max Drawdown:          -23.45%      -33.92%

Alpha:                   2.12%
Beta:                    0.65
Tracking Error:          8.23%
Information Ratio:       0.26
============================================================
```

## Run Monte Carlo Simulation

```python
from portfolio_analysis import MonteCarloSimulation

# Simulate 1000 scenarios over 252 trading days
mc = MonteCarloSimulation(
    data,
    weights,
    num_simulations=1000,
    time_horizon=252,
    initial_investment=100000
)

# View results
mc.print_summary()
mc.plot_simulation()
```

## Factor Analysis

```python
from portfolio_analysis.factors import FactorDataLoader, FactorRegression

# Load Fama-French factors
factor_loader = FactorDataLoader()
ff3 = factor_loader.get_ff3_factors('2019-01-01', '2024-01-01')

# Run regression
returns = portfolio.calculate_portfolio_returns()
regression = FactorRegression(returns, ff3)
results = regression.run_regression('ff3')

print(results.summary())
```

**Output:**
```
============================================================
Factor Regression Results: FF3
============================================================
Observations: 1257
R-squared: 0.9234
Adj R-squared: 0.9231

Coefficient      Value     T-stat    P-value
------------------------------------------
Alpha            0.02%       0.45     0.6521
Mkt-RF           0.682      45.23     0.0000
SMB              0.123       4.56     0.0000
HML              0.087       3.21     0.0014
============================================================
```

## Portfolio Optimization

```python
from portfolio_analysis import PortfolioOptimizer

# Find optimal portfolio
optimizer = PortfolioOptimizer(data, risk_free_rate=0.04)

# Maximum Sharpe ratio
optimal = optimizer.optimize_max_sharpe()
print(f"Optimal weights: {optimal['weights']}")
print(f"Expected Sharpe: {optimal['sharpe_ratio']:.2f}")

# Visualize efficient frontier
optimizer.plot_efficient_frontier()
```

## Generate Report

```python
from portfolio_analysis.reporting import ReportBuilder

# Create HTML tear sheet
report = ReportBuilder(
    portfolio,
    benchmark=benchmark,
    title="My Portfolio Analysis"
)
report.generate("tearsheet.html")
```

## Next Steps

- [Factor Models Guide](../user-guide/factor-models.md) - Deep dive into factor analysis
- [Optimization Guide](../user-guide/optimization.md) - Advanced optimization techniques
- [API Reference](../api/index.md) - Complete API documentation
