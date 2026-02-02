"""Tests for edge cases and error conditions across the package."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation
from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.exceptions import ValidationError
from portfolio_analysis.metrics.benchmark import BenchmarkComparison
from portfolio_analysis.metrics.performance import PerformanceMetrics


@pytest.fixture
def minimal_data():
    """Create minimal price data (just a few days)."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="B")
    data = pd.DataFrame(
        {"A": [100, 101, 102, 101, 103, 102, 104], "B": [100, 100.5, 101, 100.5, 101.5, 101, 102]},
        index=dates[:7],
    )
    return data


@pytest.fixture
def single_asset_data():
    """Create single asset price data."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
    n_days = len(dates)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))
    return pd.DataFrame({"STOCK": prices}, index=dates)


@pytest.fixture
def flat_data():
    """Create data with no returns (flat prices)."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
    n_days = len(dates)
    return pd.DataFrame(
        {"A": [100.0] * n_days, "B": [50.0] * n_days},
        index=dates,
    )


@pytest.fixture
def extreme_data():
    """Create data with extreme returns."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
    n_days = len(dates)

    # Include some extreme moves
    returns = np.random.normal(0.001, 0.05, n_days)  # Very high volatility
    returns[50] = 0.20  # 20% up day
    returns[100] = -0.15  # 15% down day

    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"VOLATILE": prices}, index=dates)


class TestSingleAssetPortfolio:
    """Tests for single-asset portfolios."""

    def test_portfolio_analysis_single_asset(self, single_asset_data):
        """Test PortfolioAnalysis with single asset."""
        portfolio = PortfolioAnalysis(single_asset_data, [1.0])

        assert portfolio.calculate_portfolio_return() is not None
        assert portfolio.calculate_portfolio_volatility() > 0
        assert portfolio.calculate_max_drawdown() <= 0

    def test_montecarlo_single_asset(self, single_asset_data):
        """Test MonteCarloSimulation with single asset."""
        mc = MonteCarloSimulation(
            single_asset_data,
            [1.0],
            num_simulations=10,
            time_horizon=10,
        )
        results = mc.simulate()

        assert results.shape == (10, 10)
        assert (results > 0).all()

    def test_performance_metrics_single_asset(self, single_asset_data):
        """Test PerformanceMetrics with single asset (Series)."""
        series = single_asset_data["STOCK"]

        annual_return = PerformanceMetrics.calculate_annual_return(series)
        annual_vol = PerformanceMetrics.calculate_annual_volatility(series)
        max_dd = PerformanceMetrics.calculate_max_drawdown(series)

        assert isinstance(annual_return, float)
        assert isinstance(annual_vol, float)
        assert isinstance(max_dd, float)


class TestMinimalData:
    """Tests for minimal data scenarios."""

    def test_portfolio_with_minimal_data(self, minimal_data):
        """Test portfolio analysis with very few data points."""
        portfolio = PortfolioAnalysis(minimal_data, [0.5, 0.5])

        # Should still work with minimal data
        assert portfolio.calculate_portfolio_return() is not None
        assert portfolio.calculate_portfolio_volatility() >= 0

    def test_var_with_minimal_data(self, minimal_data):
        """Test VaR calculation with minimal data."""
        var = PerformanceMetrics.calculate_var(minimal_data, confidence_level=0.95)
        assert var is not None


class TestExtremeValues:
    """Tests for extreme value scenarios."""

    def test_extreme_volatility(self, extreme_data):
        """Test metrics with extreme volatility data."""
        vol = PerformanceMetrics.calculate_annual_volatility(extreme_data)
        assert vol.iloc[0] > 0

    def test_extreme_drawdown(self, extreme_data):
        """Test max drawdown with extreme data."""
        max_dd = PerformanceMetrics.calculate_max_drawdown(extreme_data)
        assert max_dd.iloc[0] < 0  # Should have significant drawdown

    def test_cvar_with_extreme_data(self, extreme_data):
        """Test CVaR with extreme data."""
        cvar = PerformanceMetrics.calculate_cvar(extreme_data)
        var = PerformanceMetrics.calculate_var(extreme_data)

        # CVaR should be at least as extreme as VaR
        assert cvar.iloc[0] <= var.iloc[0]


class TestFlatData:
    """Tests for flat (no returns) data."""

    def test_volatility_near_zero(self, flat_data):
        """Test volatility is near zero for flat data."""
        vol = PerformanceMetrics.calculate_annual_volatility(flat_data)
        assert (vol < 0.0001).all()

    def test_sharpe_with_flat_data(self, flat_data):
        """Test Sharpe ratio with flat data (should handle zero volatility)."""
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(flat_data)
        # With near-zero volatility, Sharpe may be inf or very large negative
        assert sharpe is not None


class TestWeightValidation:
    """Tests for weight validation edge cases."""

    def test_weights_exactly_one(self):
        """Test weights that sum exactly to 1."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
        data = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
                "B": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
            },
            index=dates,
        )

        # These should work
        portfolio = PortfolioAnalysis(data, [0.5, 0.5])
        assert portfolio is not None

        portfolio = PortfolioAnalysis(data, [1.0, 0.0])
        assert portfolio is not None

    def test_weights_close_to_one(self):
        """Test weights that are very close to 1 (floating point)."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
        data = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
                "B": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
                "C": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
            },
            index=dates,
        )

        # 1/3 + 1/3 + 1/3 may not exactly equal 1.0 due to floating point
        weights = [1 / 3, 1 / 3, 1 / 3]
        portfolio = PortfolioAnalysis(data, weights)
        assert portfolio is not None


class TestNewMetrics:
    """Tests for newly added performance metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
        n_days = len(dates)

        returns = np.random.normal(0.0005, 0.01, n_days)
        prices = 100 * np.cumprod(1 + returns)

        return pd.DataFrame({"STOCK": prices}, index=dates)

    def test_cvar(self, sample_data):
        """Test CVaR calculation."""
        cvar = PerformanceMetrics.calculate_cvar(sample_data)
        var = PerformanceMetrics.calculate_var(sample_data)

        assert cvar.iloc[0] <= var.iloc[0]  # CVaR should be more extreme
        assert cvar.iloc[0] < 0  # Should be negative

    def test_omega_ratio(self, sample_data):
        """Test Omega ratio calculation."""
        omega = PerformanceMetrics.calculate_omega_ratio(sample_data)

        assert omega.iloc[0] > 0  # Should be positive for profitable data

    def test_ulcer_index(self, sample_data):
        """Test Ulcer Index calculation."""
        ulcer = PerformanceMetrics.calculate_ulcer_index(sample_data)

        assert ulcer.iloc[0] >= 0  # Should be non-negative
        assert ulcer.iloc[0] < 1  # Should be reasonable

    def test_recovery_factor(self, sample_data):
        """Test Recovery Factor calculation."""
        recovery = PerformanceMetrics.calculate_recovery_factor(sample_data)

        assert isinstance(recovery.iloc[0], float)

    def test_win_rate(self, sample_data):
        """Test Win Rate calculation."""
        win_rate = PerformanceMetrics.calculate_win_rate(sample_data)

        assert 0 <= win_rate.iloc[0] <= 1

    def test_profit_factor(self, sample_data):
        """Test Profit Factor calculation."""
        profit_factor = PerformanceMetrics.calculate_profit_factor(sample_data)

        assert profit_factor.iloc[0] > 0

    def test_payoff_ratio(self, sample_data):
        """Test Payoff Ratio calculation."""
        payoff = PerformanceMetrics.calculate_payoff_ratio(sample_data)

        assert payoff.iloc[0] > 0

    def test_herfindahl_index(self):
        """Test Herfindahl Index calculation."""
        # Equal weights
        hhi_equal = PerformanceMetrics.calculate_herfindahl_index([0.25, 0.25, 0.25, 0.25])
        assert np.isclose(hhi_equal, 0.25)  # 4 * (0.25)^2 = 0.25

        # Concentrated weights
        hhi_conc = PerformanceMetrics.calculate_herfindahl_index([1.0])
        assert np.isclose(hhi_conc, 1.0)

        # Mixed weights
        hhi_mixed = PerformanceMetrics.calculate_herfindahl_index([0.5, 0.3, 0.2])
        expected = 0.5**2 + 0.3**2 + 0.2**2
        assert np.isclose(hhi_mixed, expected)

    def test_effective_n(self):
        """Test Effective N calculation."""
        # Equal weights - effective N should equal actual N
        eff_n = PerformanceMetrics.calculate_effective_n([0.25, 0.25, 0.25, 0.25])
        assert np.isclose(eff_n, 4.0)

        # Concentrated - effective N should be 1
        eff_n_conc = PerformanceMetrics.calculate_effective_n([1.0])
        assert np.isclose(eff_n_conc, 1.0)


class TestDataFrameVsSeries:
    """Tests for consistent behavior between DataFrame and Series inputs."""

    @pytest.fixture
    def price_data(self):
        """Create price data as both DataFrame and Series."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
        n_days = len(dates)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))

        df = pd.DataFrame({"STOCK": prices}, index=dates)
        series = pd.Series(prices, index=dates, name="STOCK")

        return df, series

    def test_annual_return_consistency(self, price_data):
        """Test annual return is consistent for DataFrame and Series."""
        df, series = price_data

        ret_df = PerformanceMetrics.calculate_annual_return(df)
        ret_series = PerformanceMetrics.calculate_annual_return(series)

        assert np.isclose(ret_df.iloc[0], ret_series)

    def test_annual_volatility_consistency(self, price_data):
        """Test annual volatility is consistent for DataFrame and Series."""
        df, series = price_data

        vol_df = PerformanceMetrics.calculate_annual_volatility(df)
        vol_series = PerformanceMetrics.calculate_annual_volatility(series)

        assert np.isclose(vol_df.iloc[0], vol_series)

    def test_max_drawdown_consistency(self, price_data):
        """Test max drawdown is consistent for DataFrame and Series."""
        df, series = price_data

        dd_df = PerformanceMetrics.calculate_max_drawdown(df)
        dd_series = PerformanceMetrics.calculate_max_drawdown(series)

        assert np.isclose(dd_df.iloc[0], dd_series)

    def test_sortino_consistency(self, price_data):
        """Test Sortino ratio is consistent for DataFrame and Series."""
        df, series = price_data

        sortino_df = PerformanceMetrics.calculate_sortino_ratio(df)
        sortino_series = PerformanceMetrics.calculate_sortino_ratio(series)

        assert np.isclose(sortino_df.iloc[0], sortino_series, rtol=0.01)


class TestErrorConditions:
    """Tests for proper error handling."""

    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
            PerformanceMetrics.calculate_annual_return(empty_df)

    def test_mismatched_weights_length(self):
        """Test error for mismatched weights length."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
        data = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
                "B": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
            },
            index=dates,
        )

        with pytest.raises(ValueError):
            PortfolioAnalysis(data, [0.5, 0.3, 0.2])  # 3 weights for 2 assets

    def test_extreme_confidence_level(self):
        """Test handling of extreme confidence levels."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
        data = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))},
            index=dates,
        )

        # Very high confidence level (99.9%)
        var_high = PerformanceMetrics.calculate_var(data, confidence_level=0.999)
        assert var_high is not None

        # Very low confidence level (50%)
        var_low = PerformanceMetrics.calculate_var(data, confidence_level=0.5)
        assert var_low is not None

        # High confidence should give more extreme VaR
        assert var_high.iloc[0] < var_low.iloc[0]
