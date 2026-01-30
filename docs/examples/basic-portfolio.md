# Basic Portfolio Example

A complete example analyzing a three-fund portfolio.

## Setup

```python
from portfolio_analysis import (
    DataLoader,
    PortfolioAnalysis,
    BenchmarkComparison,
    MonteCarloSimulation,
    PortfolioVisualization
)

# Three-fund portfolio
tickers = ['VTI', 'VXUS', 'BND']
weights = [0.50, 0.20, 0.30]

# Load 5 years of data
loader = DataLoader(tickers, '2019-01-01', '2024-01-01')
prices = loader.fetch_data()
```

## Performance Analysis

```python
portfolio = PortfolioAnalysis(prices, weights)
portfolio.print_summary()
```

## Benchmark Comparison

```python
benchmark = BenchmarkComparison(prices, weights, benchmark_ticker='SPY')
benchmark.print_comparison()
```

## Monte Carlo Projection

```python
mc = MonteCarloSimulation(
    prices, weights,
    num_simulations=10000,
    time_horizon=252 * 5,  # 5 years
    initial_investment=100000
)
mc.print_summary()
mc.plot_simulation()
```

## Visualization

```python
# Performance
PortfolioVisualization.plot_portfolio_return(prices, weights)

# Risk
PortfolioVisualization.plot_drawdown(prices, weights)

# Allocation
PortfolioVisualization.plot_allocation_pie(weights, tickers)
```
