"""
Centralized constants for the portfolio_analysis package.

These constants ensure consistency across all modules and make it easy
to adjust default values in one place.
"""

# Time-related constants
TRADING_DAYS_PER_YEAR: int = 252
"""Number of trading days per year (US markets)."""

DAYS_PER_YEAR: float = 365.25
"""Average days per year including leap years (for CAGR calculations)."""

# Default financial parameters
DEFAULT_RISK_FREE_RATE: float = 0.02
"""Default annual risk-free rate (2%)."""

DEFAULT_CONFIDENCE_LEVEL: float = 0.95
"""Default confidence level for VaR calculations (95%)."""

# Default benchmark
DEFAULT_BENCHMARK: str = "SPY"
"""Default benchmark ticker symbol (S&P 500 ETF)."""

# Optimization constraints
DEFAULT_MIN_WEIGHT: float = 0.0
"""Default minimum weight per asset in optimization."""

DEFAULT_MAX_WEIGHT: float = 1.0
"""Default maximum weight per asset in optimization."""

# Monte Carlo defaults
DEFAULT_NUM_SIMULATIONS: int = 1000
"""Default number of Monte Carlo simulations."""

DEFAULT_PROJECTION_DAYS: int = 252
"""Default projection period in trading days (1 year)."""

# Tolerance values
WEIGHT_SUM_TOLERANCE: float = 1e-6
"""Tolerance for validating that weights sum to 1.0."""
