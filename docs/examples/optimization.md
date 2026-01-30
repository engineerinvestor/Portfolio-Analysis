# Optimization Example

Find optimal portfolio weights.

## Setup

```python
from portfolio_analysis import DataLoader, PortfolioOptimizer

tickers = ['VTI', 'VEA', 'VWO', 'BND', 'VNQ', 'GLD']
loader = DataLoader(tickers, '2019-01-01', '2024-01-01')
prices = loader.fetch_data()

optimizer = PortfolioOptimizer(prices, risk_free_rate=0.04)
```

## Find Optimal Portfolios

```python
# Maximum Sharpe
max_sharpe = optimizer.optimize_max_sharpe()
print("Max Sharpe Portfolio:")
for ticker, weight in max_sharpe['weights'].items():
    if weight > 0.01:
        print(f"  {ticker}: {weight:.1%}")

# Minimum Volatility
min_vol = optimizer.optimize_min_volatility()

# Risk Parity
risk_parity = optimizer.optimize_risk_parity()
```

## Efficient Frontier

```python
optimizer.plot_efficient_frontier(
    show_assets=True,
    show_optimal=True
)
```

## Compare Strategies

```python
optimizer.print_comparison()
```
