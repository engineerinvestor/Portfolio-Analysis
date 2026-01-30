# Portfolio Analysis

This guide covers the core portfolio analysis capabilities.

## Loading Data

```python
from portfolio_analysis import DataLoader

# Define your portfolio
tickers = ['VTI', 'VXUS', 'BND', 'VNQ']
loader = DataLoader(tickers, '2019-01-01', '2024-01-01')

# Fetch adjusted close prices
prices = loader.fetch_data()

# Or fetch returns directly
returns = loader.fetch_returns(frequency='daily')  # or 'weekly', 'monthly'
```

## Portfolio Analysis

```python
from portfolio_analysis import PortfolioAnalysis

weights = [0.4, 0.2, 0.3, 0.1]
portfolio = PortfolioAnalysis(prices, weights)

# Individual metrics
annual_return = portfolio.calculate_portfolio_return()
annual_vol = portfolio.calculate_portfolio_volatility()
sharpe = portfolio.calculate_portfolio_sharpe_ratio(risk_free_rate=0.04)
max_dd = portfolio.calculate_max_drawdown()

# Get all metrics at once
summary = portfolio.get_summary()
portfolio.print_summary()
```

## Performance Metrics

```python
from portfolio_analysis import PerformanceMetrics

# Calculate for all assets
annual_returns = PerformanceMetrics.calculate_annual_return(prices)
annual_vols = PerformanceMetrics.calculate_annual_volatility(prices)
sharpe_ratios = PerformanceMetrics.calculate_sharpe_ratio(prices)
sortino_ratios = PerformanceMetrics.calculate_sortino_ratio(prices)
max_drawdowns = PerformanceMetrics.calculate_max_drawdown(prices)
var_95 = PerformanceMetrics.calculate_var(prices, confidence_level=0.95)
cagr = PerformanceMetrics.calculate_cagr(prices)
calmar = PerformanceMetrics.calculate_calmar_ratio(prices)
```

## Benchmark Comparison

```python
from portfolio_analysis import BenchmarkComparison

benchmark = BenchmarkComparison(prices, weights, benchmark_ticker='SPY')

# Key metrics
alpha = benchmark.calculate_alpha()
beta = benchmark.calculate_beta()
tracking_error = benchmark.calculate_tracking_error()
info_ratio = benchmark.calculate_information_ratio()
up_capture = benchmark.calculate_up_capture()
down_capture = benchmark.calculate_down_capture()

benchmark.print_comparison()
```

## Monte Carlo Simulation

```python
from portfolio_analysis import MonteCarloSimulation

mc = MonteCarloSimulation(
    prices,
    weights,
    num_simulations=10000,
    time_horizon=252,  # 1 year
    initial_investment=100000
)

results = mc.simulate()
stats = mc.get_statistics()

print(f"Expected final value: ${stats['final_values']['mean']:,.0f}")
print(f"5th percentile: ${stats['final_values']['percentile_5']:,.0f}")
print(f"Probability of loss: {stats['final_values']['prob_loss']:.1f}%")

mc.plot_simulation()
```

## Visualization

```python
from portfolio_analysis import PortfolioVisualization

# Cumulative returns
PortfolioVisualization.plot_performance(prices)
PortfolioVisualization.plot_portfolio_return(prices, weights)

# Risk visualization
PortfolioVisualization.plot_drawdown(prices, weights)
PortfolioVisualization.plot_rolling_volatility(prices, weights, window=21)
PortfolioVisualization.plot_returns_distribution(prices, weights)

# Allocation
PortfolioVisualization.plot_allocation_pie(weights, tickers)
```
