# Factor Analysis Example

Decompose portfolio returns using Fama-French factors.

## Setup

```python
from portfolio_analysis import DataLoader, PortfolioAnalysis
from portfolio_analysis.factors import (
    FactorDataLoader,
    FactorRegression,
    FactorAttribution,
    FactorVisualization
)

# Factor-tilted portfolio
tickers = ['VTI', 'VBR', 'VTV', 'MTUM', 'BND']
weights = [0.25, 0.15, 0.15, 0.15, 0.30]

loader = DataLoader(tickers, '2019-01-01', '2024-01-01')
prices = loader.fetch_data()

portfolio = PortfolioAnalysis(prices, weights)
returns = portfolio.calculate_portfolio_returns()
```

## Load Factor Data

```python
factor_loader = FactorDataLoader()
ff3 = factor_loader.get_ff3_factors('2019-01-01', '2024-01-01')
carhart = factor_loader.get_carhart_factors('2019-01-01', '2024-01-01')
```

## Factor Regression

```python
regression = FactorRegression(returns, carhart)

# Compare models
print(regression.compare_models())

# Detailed FF3 results
results = regression.run_regression('ff3')
print(results.summary())
```

## Return Attribution

```python
attribution = FactorAttribution(returns, ff3)

# Return decomposition
decomp = attribution.decompose_returns('ff3')
print(f"Total: {decomp['total']:.2%}")
print(f"Market: {decomp['Mkt-RF']:.2%}")
print(f"Size: {decomp['SMB']:.2%}")
print(f"Value: {decomp['HML']:.2%}")
print(f"Alpha: {decomp['alpha']:.2%}")
```

## Visualization

```python
# Factor exposures
FactorVisualization.plot_factor_exposures(results)

# Return attribution waterfall
FactorVisualization.plot_return_attribution(decomp)

# Rolling betas
rolling = regression.run_rolling_regression('ff3', window=60)
FactorVisualization.plot_rolling_betas(rolling)
```
