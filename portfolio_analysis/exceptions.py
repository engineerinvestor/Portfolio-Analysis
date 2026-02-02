"""
Custom exceptions for the portfolio_analysis package.

This module provides a hierarchy of exceptions for better error handling
and more informative error messages.
"""


class PortfolioAnalysisError(Exception):
    """
    Base exception for all portfolio_analysis errors.

    All custom exceptions in this package inherit from this class,
    making it easy to catch any package-specific error.
    """

    pass


class ValidationError(PortfolioAnalysisError):
    """
    Raised when input validation fails.

    Examples include:
    - Weights don't sum to 1.0
    - Negative weights when not allowed
    - Mismatched array lengths
    - Invalid date ranges
    """

    pass


class DataError(PortfolioAnalysisError):
    """
    Raised when there are issues with data.

    Examples include:
    - Empty DataFrames
    - Missing columns
    - Insufficient data points
    - Failed data downloads
    """

    pass


class OptimizationError(PortfolioAnalysisError):
    """
    Raised when portfolio optimization fails.

    Examples include:
    - Optimizer fails to converge
    - Infeasible constraints
    - Target return unachievable
    """

    pass


class ConfigurationError(PortfolioAnalysisError):
    """
    Raised when there are configuration issues.

    Examples include:
    - Invalid parameter combinations
    - Missing required configuration
    """

    pass
