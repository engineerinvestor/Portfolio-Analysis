"""
Utility functions for portfolio analysis.
"""

import numpy as np


def validate_weights(weights: list[float], tolerance: float = 0.001) -> bool:
    """
    Validate that portfolio weights sum to 1.0.

    Parameters
    ----------
    weights : list of float
        Portfolio weights
    tolerance : float, default 0.001
        Allowed deviation from 1.0

    Returns
    -------
    bool
        True if weights are valid
    """
    return np.isclose(sum(weights), 1.0, atol=tolerance)


def normalize_weights(weights: list[float]) -> np.ndarray:
    """
    Normalize weights to sum to 1.0.

    Parameters
    ----------
    weights : list of float
        Portfolio weights

    Returns
    -------
    np.ndarray
        Normalized weights
    """
    weights = np.array(weights)
    return weights / weights.sum()


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.

    Parameters
    ----------
    value : float
        Value to format (e.g., 0.05 for 5%)
    decimals : int, default 2
        Number of decimal places

    Returns
    -------
    str
        Formatted percentage (e.g., "5.00%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$") -> str:
    """
    Format a value as currency.

    Parameters
    ----------
    value : float
        Value to format
    symbol : str, default "$"
        Currency symbol

    Returns
    -------
    str
        Formatted currency (e.g., "$10,000.00")
    """
    return f"{symbol}{value:,.2f}"
