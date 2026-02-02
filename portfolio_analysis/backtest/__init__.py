"""
Backtesting framework for portfolio strategies.

This module provides tools for backtesting portfolio strategies
with realistic transaction costs, rebalancing rules, and performance analysis.
"""

from portfolio_analysis.backtest.engine import BacktestEngine, BacktestResult
from portfolio_analysis.backtest.metrics import BacktestMetrics
from portfolio_analysis.backtest.strategy import (
    BuyAndHoldStrategy,
    MomentumStrategy,
    RebalanceStrategy,
    Strategy,
)
from portfolio_analysis.backtest.visualization import BacktestVisualization

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestMetrics",
    "Strategy",
    "BuyAndHoldStrategy",
    "RebalanceStrategy",
    "MomentumStrategy",
    "BacktestVisualization",
]
