"""
Factor data loading functionality for Fama-French factor models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from datetime import datetime
import os
import hashlib


class FactorDataLoader:
    """
    Fetch Fama-French factor data from Kenneth French Data Library.

    Uses pandas-datareader to download factor data and provides local caching
    to avoid repeated downloads.

    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching factor data. If None, uses a temp directory.

    Examples
    --------
    >>> loader = FactorDataLoader()
    >>> ff3 = loader.get_ff3_factors('2015-01-01', '2023-12-31')
    >>> print(ff3.columns.tolist())
    ['Mkt-RF', 'SMB', 'HML', 'RF']
    """

    # French data library dataset names
    FF3_DAILY = "F-F_Research_Data_Factors_daily"
    FF3_MONTHLY = "F-F_Research_Data_Factors"
    FF5_DAILY = "F-F_Research_Data_5_Factors_2x3_daily"
    FF5_MONTHLY = "F-F_Research_Data_5_Factors_2x3"
    MOM_DAILY = "F-F_Momentum_Factor_daily"
    MOM_MONTHLY = "F-F_Momentum_Factor"

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            import tempfile

            cache_dir = os.path.join(tempfile.gettempdir(), "ff_factors_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, dataset: str) -> str:
        """Generate cache file path for a dataset."""
        return os.path.join(self.cache_dir, f"{dataset.replace('-', '_')}.parquet")

    def _load_from_cache(
        self, dataset: str, max_age_days: int = 7
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not stale."""
        cache_path = self._get_cache_path(dataset)
        if not os.path.exists(cache_path):
            return None

        # Check age
        mtime = os.path.getmtime(cache_path)
        age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
        if age_days > max_age_days:
            return None

        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None

    def _save_to_cache(self, data: pd.DataFrame, dataset: str) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(dataset)
        try:
            data.to_parquet(cache_path)
        except Exception:
            pass  # Silently fail if caching doesn't work

    def _fetch_french_data(self, dataset: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch data from Kenneth French Data Library.

        Parameters
        ----------
        dataset : str
            Dataset name from French library
        use_cache : bool, default True
            Whether to use local cache

        Returns
        -------
        pd.DataFrame
            Factor data with datetime index
        """
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(dataset)
            if cached is not None:
                return cached

        try:
            import pandas_datareader.data as web
        except ImportError:
            raise ImportError(
                "pandas-datareader is required for factor data. "
                "Install with: pip install pandas-datareader"
            )

        # Fetch from French library
        data = web.DataReader(dataset, "famafrench", start="1900-01-01")

        # web.DataReader returns a dict with multiple tables
        # First table (index 0) is typically the main data
        df = data[0]

        # Convert from percentage to decimal
        df = df / 100.0

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str))

        # Cache the result
        if use_cache:
            self._save_to_cache(df, dataset)

        return df

    def _filter_dates(
        self,
        data: pd.DataFrame,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Filter data to date range."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return data[(data.index >= start) & (data.index <= end)]

    def get_ff3_factors(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Get Fama-French 3-factor data.

        Parameters
        ----------
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        frequency : str, default 'daily'
            Data frequency: 'daily' or 'monthly'

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Mkt-RF, SMB, HML, RF
        """
        if frequency == "daily":
            dataset = self.FF3_DAILY
        elif frequency == "monthly":
            dataset = self.FF3_MONTHLY
        else:
            raise ValueError(
                f"Frequency must be 'daily' or 'monthly', got: {frequency}"
            )

        data = self._fetch_french_data(dataset)
        return self._filter_dates(data, start_date, end_date)

    def get_ff5_factors(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Get Fama-French 5-factor data.

        Parameters
        ----------
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        frequency : str, default 'daily'
            Data frequency: 'daily' or 'monthly'

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
        """
        if frequency == "daily":
            dataset = self.FF5_DAILY
        elif frequency == "monthly":
            dataset = self.FF5_MONTHLY
        else:
            raise ValueError(
                f"Frequency must be 'daily' or 'monthly', got: {frequency}"
            )

        data = self._fetch_french_data(dataset)
        return self._filter_dates(data, start_date, end_date)

    def get_momentum_factor(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily",
    ) -> pd.Series:
        """
        Get momentum factor data.

        Parameters
        ----------
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        frequency : str, default 'daily'
            Data frequency: 'daily' or 'monthly'

        Returns
        -------
        pd.Series
            Momentum factor (MOM or WML)
        """
        if frequency == "daily":
            dataset = self.MOM_DAILY
        elif frequency == "monthly":
            dataset = self.MOM_MONTHLY
        else:
            raise ValueError(
                f"Frequency must be 'daily' or 'monthly', got: {frequency}"
            )

        data = self._fetch_french_data(dataset)
        filtered = self._filter_dates(data, start_date, end_date)

        # Return as Series, column name varies
        if "Mom" in filtered.columns:
            return filtered["Mom"]
        elif "WML" in filtered.columns:
            return filtered["WML"]
        else:
            return filtered.iloc[:, 0]

    def get_carhart_factors(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Get Carhart 4-factor data (FF3 + Momentum).

        Parameters
        ----------
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        frequency : str, default 'daily'
            Data frequency: 'daily' or 'monthly'

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Mkt-RF, SMB, HML, MOM, RF
        """
        ff3 = self.get_ff3_factors(start_date, end_date, frequency)
        mom = self.get_momentum_factor(start_date, end_date, frequency)

        # Align dates
        common_dates = ff3.index.intersection(mom.index)
        result = ff3.loc[common_dates].copy()
        result["MOM"] = mom.loc[common_dates]

        # Reorder columns to put MOM before RF
        cols = ["Mkt-RF", "SMB", "HML", "MOM", "RF"]
        return result[cols]


def align_returns_with_factors(
    returns: pd.Series, factor_data: pd.DataFrame, compute_excess: bool = True
) -> tuple:
    """
    Align portfolio returns with factor data and compute excess returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio or asset returns with datetime index
    factor_data : pd.DataFrame
        Factor data from FactorDataLoader
    compute_excess : bool, default True
        Whether to subtract risk-free rate from returns

    Returns
    -------
    tuple
        (aligned_excess_returns, aligned_factors) both as pandas objects
    """
    # Find common dates
    common_dates = returns.index.intersection(factor_data.index)

    if len(common_dates) == 0:
        raise ValueError(
            "No overlapping dates between returns and factor data. "
            "Check that date ranges match and frequency is compatible."
        )

    aligned_returns = returns.loc[common_dates]
    aligned_factors = factor_data.loc[common_dates]

    if compute_excess and "RF" in aligned_factors.columns:
        excess_returns = aligned_returns - aligned_factors["RF"]
    else:
        excess_returns = aligned_returns

    return excess_returns, aligned_factors
