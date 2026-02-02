"""
Strategy classes for backtesting.

This module provides base strategy class and common strategy implementations
for use with the BacktestEngine.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for portfolio strategies.

    All strategies must implement the generate_weights method which
    returns target weights for each rebalancing period.

    Parameters
    ----------
    initial_weights : dict
        Initial portfolio weights mapping tickers to weights
    name : str, optional
        Strategy name for reporting

    Examples
    --------
    >>> class MyStrategy(Strategy):
    ...     def generate_weights(self, data, current_weights, current_date):
    ...         return self.initial_weights
    """

    def __init__(
        self,
        initial_weights: dict[str, float],
        name: Optional[str] = None,
    ):
        self.initial_weights = initial_weights
        self.name = name or self.__class__.__name__

        # Validate weights
        if not np.isclose(sum(initial_weights.values()), 1.0):
            raise ValueError("Initial weights must sum to 1.0")

    @abstractmethod
    def generate_weights(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Generate target weights for the portfolio.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data up to current_date
        current_weights : dict
            Current portfolio weights
        current_date : pd.Timestamp
            Current date in the backtest

        Returns
        -------
        dict
            Target portfolio weights
        """
        pass

    def should_rebalance(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Override this method to implement custom rebalancing logic.
        Default implementation always returns True.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data up to current_date
        current_weights : dict
            Current portfolio weights
        target_weights : dict
            Target portfolio weights
        current_date : pd.Timestamp
            Current date

        Returns
        -------
        bool
            Whether to rebalance
        """
        return True


class BuyAndHoldStrategy(Strategy):
    """
    Buy and hold strategy with no rebalancing.

    Maintains initial allocation and lets weights drift with market movements.

    Parameters
    ----------
    initial_weights : dict
        Initial portfolio weights
    name : str, optional
        Strategy name

    Examples
    --------
    >>> strategy = BuyAndHoldStrategy({'SPY': 0.6, 'AGG': 0.4})
    >>> engine = BacktestEngine(data, strategy)
    >>> result = engine.run()
    """

    def __init__(
        self,
        initial_weights: dict[str, float],
        name: Optional[str] = None,
    ):
        super().__init__(initial_weights, name or "Buy and Hold")
        self._initialized = False

    def generate_weights(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Return initial weights only on first call, then current weights."""
        if not self._initialized:
            self._initialized = True
            return self.initial_weights.copy()
        return current_weights.copy()

    def should_rebalance(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> bool:
        """Never rebalance after initial allocation."""
        return not self._initialized


class RebalanceStrategy(Strategy):
    """
    Periodic rebalancing strategy.

    Rebalances portfolio back to target weights at specified intervals.

    Parameters
    ----------
    initial_weights : dict
        Target portfolio weights
    rebalance_frequency : str, default 'Q'
        Rebalancing frequency: 'D' (daily), 'W' (weekly),
        'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    drift_threshold : float, optional
        Only rebalance if any weight drifts more than this amount from target.
        If None, always rebalance at the specified frequency.
    name : str, optional
        Strategy name

    Examples
    --------
    >>> strategy = RebalanceStrategy(
    ...     {'SPY': 0.6, 'AGG': 0.4},
    ...     rebalance_frequency='M',
    ...     drift_threshold=0.05
    ... )
    """

    FREQUENCY_MAP = {
        "D": 1,
        "W": 5,
        "M": 21,
        "Q": 63,
        "Y": 252,
    }

    def __init__(
        self,
        initial_weights: dict[str, float],
        rebalance_frequency: str = "Q",
        drift_threshold: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            initial_weights, name or f"Rebalance ({rebalance_frequency})"
        )
        self.rebalance_frequency = rebalance_frequency
        self.drift_threshold = drift_threshold
        self._last_rebalance_date: Optional[pd.Timestamp] = None
        self._days_between_rebalance = self.FREQUENCY_MAP.get(
            rebalance_frequency, 63
        )

    def generate_weights(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Return target weights."""
        return self.initial_weights.copy()

    def should_rebalance(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> bool:
        """Check if rebalancing is needed based on frequency and drift."""
        # First time - always rebalance
        if self._last_rebalance_date is None:
            self._last_rebalance_date = current_date
            return True

        # Check frequency
        days_since_last = (current_date - self._last_rebalance_date).days
        if days_since_last < self._days_between_rebalance:
            return False

        # Check drift threshold if specified
        if self.drift_threshold is not None:
            max_drift = max(
                abs(current_weights.get(k, 0) - target_weights.get(k, 0))
                for k in set(current_weights) | set(target_weights)
            )
            if max_drift < self.drift_threshold:
                return False

        self._last_rebalance_date = current_date
        return True


class MomentumStrategy(Strategy):
    """
    Momentum-based strategy.

    Allocates more weight to assets with higher recent returns.

    Parameters
    ----------
    initial_weights : dict
        Base weights (used when momentum is neutral)
    lookback_period : int, default 21
        Number of days for momentum calculation
    momentum_tilt : float, default 0.5
        How much to tilt towards momentum (0 = ignore momentum, 1 = full momentum)
    rebalance_frequency : str, default 'M'
        How often to update weights
    name : str, optional
        Strategy name

    Examples
    --------
    >>> strategy = MomentumStrategy(
    ...     {'SPY': 0.5, 'QQQ': 0.5},
    ...     lookback_period=21,
    ...     momentum_tilt=0.3
    ... )
    """

    FREQUENCY_MAP = {
        "D": 1,
        "W": 5,
        "M": 21,
        "Q": 63,
        "Y": 252,
    }

    def __init__(
        self,
        initial_weights: dict[str, float],
        lookback_period: int = 21,
        momentum_tilt: float = 0.5,
        rebalance_frequency: str = "M",
        name: Optional[str] = None,
    ):
        super().__init__(
            initial_weights, name or f"Momentum ({lookback_period}d)"
        )
        self.lookback_period = lookback_period
        self.momentum_tilt = momentum_tilt
        self.rebalance_frequency = rebalance_frequency
        self._last_rebalance_date: Optional[pd.Timestamp] = None
        self._days_between_rebalance = self.FREQUENCY_MAP.get(
            rebalance_frequency, 21
        )

    def generate_weights(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Generate weights based on momentum."""
        tickers = list(self.initial_weights.keys())

        # Get available data up to current date
        available_data = data.loc[:current_date]

        # Need enough history for lookback
        if len(available_data) < self.lookback_period:
            return self.initial_weights.copy()

        # Calculate momentum (recent returns)
        recent_data = available_data.iloc[-self.lookback_period:]
        momentum = recent_data.iloc[-1] / recent_data.iloc[0] - 1

        # Only consider tickers in our universe
        momentum = momentum[tickers]

        # Convert momentum to weights using softmax-style approach
        if momentum.std() > 0:
            # Normalize momentum
            mom_normalized = (momentum - momentum.mean()) / momentum.std()

            # Create momentum weights (higher momentum = higher weight)
            mom_weights = np.exp(mom_normalized)
            mom_weights = mom_weights / mom_weights.sum()

            # Blend with base weights
            base_weights = pd.Series(self.initial_weights)
            final_weights = (
                (1 - self.momentum_tilt) * base_weights
                + self.momentum_tilt * mom_weights
            )
        else:
            final_weights = pd.Series(self.initial_weights)

        return dict(final_weights)

    def should_rebalance(
        self,
        data: pd.DataFrame,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: pd.Timestamp,
    ) -> bool:
        """Rebalance at specified frequency."""
        if self._last_rebalance_date is None:
            self._last_rebalance_date = current_date
            return True

        days_since_last = (current_date - self._last_rebalance_date).days
        if days_since_last >= self._days_between_rebalance:
            self._last_rebalance_date = current_date
            return True

        return False
