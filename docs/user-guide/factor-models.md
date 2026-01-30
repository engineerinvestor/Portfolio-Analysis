# Factor Models

Understanding what drives your portfolio returns is crucial for informed investing. Factor models decompose returns into systematic components, helping you distinguish between alpha (skill) and beta (market exposure).

## What Are Factor Models?

Factor models explain asset returns as a linear combination of common risk factors plus an idiosyncratic component:

$$R_i - R_f = \alpha_i + \beta_1 F_1 + \beta_2 F_2 + ... + \epsilon_i$$

Where:

- $R_i - R_f$ = Excess return of asset $i$
- $\alpha_i$ = Alpha (unexplained return, often attributed to skill)
- $\beta_k$ = Sensitivity to factor $k$
- $F_k$ = Factor return
- $\epsilon_i$ = Idiosyncratic (asset-specific) return

## Available Models

### CAPM (Capital Asset Pricing Model)

The simplest model with one factor: the market.

```python
from portfolio_analysis.factors import FactorRegression

results = regression.run_regression('capm')
print(f"Market Beta: {results.betas['Mkt-RF']:.3f}")
```

**Factors:** Market excess return (Mkt-RF)

### Fama-French 3-Factor Model

Adds size and value factors to CAPM.

```python
results = regression.run_regression('ff3')
```

**Factors:**

| Factor | Description | Interpretation |
|--------|-------------|----------------|
| Mkt-RF | Market excess return | Equity market exposure |
| SMB | Small Minus Big | Small-cap tilt (+ = small, - = large) |
| HML | High Minus Low | Value tilt (+ = value, - = growth) |

### Fama-French 5-Factor Model

Adds profitability and investment factors.

```python
# Requires FF5 data
ff5 = factor_loader.get_ff5_factors(start, end)
regression = FactorRegression(returns, ff5)
results = regression.run_regression('ff5')
```

**Additional Factors:**

| Factor | Description | Interpretation |
|--------|-------------|----------------|
| RMW | Robust Minus Weak | Quality/profitability tilt |
| CMA | Conservative Minus Aggressive | Investment tilt |

### Carhart 4-Factor Model

FF3 plus momentum.

```python
carhart = factor_loader.get_carhart_factors(start, end)
regression = FactorRegression(returns, carhart)
results = regression.run_regression('carhart')
```

**Additional Factor:**

| Factor | Description | Interpretation |
|--------|-------------|----------------|
| MOM | Momentum | Winners minus losers (12-1 month) |

## Loading Factor Data

Factor data is automatically fetched from Kenneth French's Data Library:

```python
from portfolio_analysis.factors import FactorDataLoader

loader = FactorDataLoader()

# Daily or monthly frequency
ff3_daily = loader.get_ff3_factors('2020-01-01', '2024-01-01', frequency='daily')
ff3_monthly = loader.get_ff3_factors('2020-01-01', '2024-01-01', frequency='monthly')

# Data is cached locally for 7 days
```

## Running Regressions

### Basic Regression

```python
from portfolio_analysis.factors import FactorRegression

regression = FactorRegression(portfolio_returns, factor_data)
results = regression.run_regression('ff3')

# Access results
print(f"Alpha (annual): {results.alpha:.2%}")
print(f"Alpha p-value: {results.alpha_pvalue:.4f}")
print(f"R-squared: {results.r_squared:.4f}")

# Factor betas
for factor, beta in results.betas.items():
    pval = results.beta_pvalues[factor]
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else "")
    print(f"{factor}: {beta:.3f}{sig}")
```

### Interpreting Results

```python
print(results.summary())
```

```
============================================================
Factor Regression Results: FF3
============================================================
Observations: 1257
R-squared: 0.9234
Adj R-squared: 0.9231
Residual Std: 5.23% (annualized)

Coefficient      Value     T-stat    P-value
--------------------------------------------
Alpha            0.02%       0.45     0.6521
Mkt-RF           0.682      45.23     0.0000
SMB              0.123       4.56     0.0000
HML              0.087       3.21     0.0014
============================================================
```

**How to read this:**

- **R-squared = 0.92**: Factors explain 92% of return variance
- **Alpha = 0.02% (p=0.65)**: No statistically significant alpha
- **Mkt-RF = 0.68**: Defensive positioning (less than 1.0)
- **SMB = 0.12**: Slight small-cap tilt
- **HML = 0.09**: Slight value tilt

### Rolling Regressions

Track how factor exposures change over time:

```python
# 60-day rolling window
rolling = regression.run_rolling_regression('ff3', window=60)

# Plot results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, factor in zip(axes, ['Mkt-RF', 'SMB', 'HML']):
    ax.plot(rolling.index, rolling[factor])
    ax.axhline(y=rolling[factor].mean(), linestyle='--', color='red')
    ax.set_ylabel(factor)
plt.show()
```

### Model Comparison

Compare how well different models explain your returns:

```python
comparison = regression.compare_models()
print(comparison)
```

```
  Model  Alpha (%)  Alpha p-value  R-squared  Mkt Beta   SMB   HML   MOM
  CAPM      0.15           0.42       0.89      0.72   NaN   NaN   NaN
   FF3      0.02           0.65       0.92      0.68  0.12  0.09   NaN
CARHART     0.01           0.78       0.93      0.67  0.11  0.08  0.05
```

## Return Attribution

Decompose your total return into factor contributions:

```python
from portfolio_analysis.factors import FactorAttribution

attribution = FactorAttribution(returns, factor_data)
decomp = attribution.decompose_returns('ff3')

print(f"Total Return: {decomp['total']:.2%}")
print(f"  Risk-Free:  {decomp['risk_free']:.2%}")
print(f"  Market:     {decomp['Mkt-RF']:.2%}")
print(f"  Size (SMB): {decomp['SMB']:.2%}")
print(f"  Value (HML):{decomp['HML']:.2%}")
print(f"  Alpha:      {decomp['alpha']:.2%}")
```

## Risk Attribution

Understand where your portfolio risk comes from:

```python
risk_decomp = attribution.decompose_risk('ff3')

print(f"Total Variance: {risk_decomp['total']:.6f}")
print(f"  Market:       {risk_decomp['Mkt-RF']/risk_decomp['total']:.1%}")
print(f"  Size:         {risk_decomp['SMB']/risk_decomp['total']:.1%}")
print(f"  Value:        {risk_decomp['HML']/risk_decomp['total']:.1%}")
print(f"  Idiosyncratic:{risk_decomp['idiosyncratic']/risk_decomp['total']:.1%}")
```

## Characteristic-Based Tilts

Estimate factor exposures from portfolio characteristics (no regression needed):

```python
from portfolio_analysis.factors import FactorExposures

exposures = FactorExposures(
    tickers=['VTI', 'VBR', 'VTV', 'VUG', 'BND'],
    weights=[0.3, 0.15, 0.2, 0.2, 0.15]
)

tilts = exposures.get_all_tilts()
print(f"Size tilt:       {tilts['size']:.2f}")  # -1=large, +1=small
print(f"Value tilt:      {tilts['value']:.2f}") # -1=growth, +1=value
print(f"Momentum tilt:   {tilts['momentum']:.2f}")
print(f"Quality tilt:    {tilts['quality']:.2f}")
print(f"Investment tilt: {tilts['investment']:.2f}")
```

## Visualization

```python
from portfolio_analysis.factors import FactorVisualization

# Factor exposure bar chart
FactorVisualization.plot_factor_exposures(results)

# Rolling betas over time
FactorVisualization.plot_rolling_betas(rolling)

# Return attribution waterfall
FactorVisualization.plot_return_attribution(decomp)

# Factor tilts radar chart
FactorVisualization.plot_factor_tilts(tilts)
```

## Best Practices

### 1. Use Sufficient Data

- Daily data: At least 1 year (252+ observations)
- Monthly data: At least 3 years (36+ observations)

### 2. Check Statistical Significance

Don't over-interpret insignificant betas:

```python
for factor in results.factors:
    if results.beta_pvalues[factor] < 0.05:
        print(f"{factor}: Significant at 5% level")
```

### 3. Consider Multiple Models

No single model is "correct". Compare models and use domain knowledge.

### 4. Watch for Regime Changes

Use rolling regressions to detect when factor exposures shift.

### 5. Distinguish Alpha from Factor Timing

"Alpha" in a static regression may actually be factor timing (changing betas over time).

## Further Reading

- Fama & French (1993): "Common Risk Factors in Stock and Bond Returns"
- Carhart (1997): "On Persistence in Mutual Fund Performance"
- Fama & French (2015): "A Five-Factor Asset Pricing Model"
