# Portfolio Optimization

Find optimal portfolio weights using various strategies.

## Basic Usage

```python
from portfolio_analysis import PortfolioOptimizer

optimizer = PortfolioOptimizer(prices, risk_free_rate=0.04)
```

## Optimization Strategies

### Maximum Sharpe Ratio

Find the portfolio with the highest risk-adjusted return:

```python
result = optimizer.optimize_max_sharpe()
print(f"Weights: {result['weights']}")
print(f"Expected Return: {result['return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

### Minimum Volatility

Find the least risky portfolio:

```python
result = optimizer.optimize_min_volatility()
```

### Target Return

Find minimum volatility for a given return target:

```python
result = optimizer.optimize_target_return(target_return=0.10)
```

### Risk Parity

Equal risk contribution from each asset:

```python
result = optimizer.optimize_risk_parity()
print(f"Risk contributions: {result['risk_contributions']}")
```

## Constraints

```python
# Long-only with max 40% per asset
result = optimizer.optimize_max_sharpe(weight_bounds=(0, 0.4))

# Allow short selling
result = optimizer.optimize_max_sharpe(weight_bounds=(-0.3, 1.0))
```

## Efficient Frontier

```python
# Generate frontier points
frontier = optimizer.generate_efficient_frontier(n_points=50)

# Visualize
optimizer.plot_efficient_frontier(show_assets=True, show_optimal=True)
```

## Factor-Aware Optimization

```python
from portfolio_analysis.factors import FactorOptimizer

factor_opt = FactorOptimizer(prices, factor_data)

# Target specific factor exposures
result = factor_opt.optimize_target_exposures(
    target_betas={'Mkt-RF': 1.0, 'SMB': 0.3, 'HML': 0.2}
)

# Neutralize factors
result = factor_opt.optimize_factor_neutral(factors=['SMB', 'HML'])

# Maximize alpha
result = factor_opt.optimize_max_alpha(model='ff3')
```
