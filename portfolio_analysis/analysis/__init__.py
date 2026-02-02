"""Portfolio analysis modules."""

from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation
from portfolio_analysis.analysis.portfolio import PortfolioAnalysis

# Optional optimization (requires scipy)
try:
    from portfolio_analysis.analysis.optimization import PortfolioOptimizer

    __all__ = ["PortfolioAnalysis", "MonteCarloSimulation", "PortfolioOptimizer"]
except ImportError:
    __all__ = ["PortfolioAnalysis", "MonteCarloSimulation"]
