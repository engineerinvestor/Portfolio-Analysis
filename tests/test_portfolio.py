"""Tests for portfolio analysis functionality."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.metrics.performance import PerformanceMetrics
from portfolio_analysis.utils.helpers import validate_weights, normalize_weights


# Create sample data for testing
@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
    n_days = len(dates)

    # Simulate two assets with different characteristics
    returns1 = np.random.normal(0.0005, 0.01, n_days)
    returns2 = np.random.normal(0.0003, 0.005, n_days)

    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 100 * np.cumprod(1 + returns2)

    data = pd.DataFrame({
        'STOCK': prices1,
        'BOND': prices2
    }, index=dates)

    return data


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return [0.6, 0.4]


class TestPortfolioAnalysis:
    """Tests for PortfolioAnalysis class."""

    def test_initialization(self, sample_data, sample_weights):
        """Test portfolio initialization."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        assert portfolio is not None
        np.testing.assert_array_equal(portfolio.weights, sample_weights)

    def test_invalid_weights_length(self, sample_data):
        """Test that mismatched weights length raises error."""
        with pytest.raises(ValueError, match="Weights length"):
            PortfolioAnalysis(sample_data, [0.5, 0.3, 0.2])  # 3 weights, 2 assets

    def test_weights_not_summing_to_one(self, sample_data):
        """Test that weights not summing to 1 raises error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            PortfolioAnalysis(sample_data, [0.5, 0.3])  # Sums to 0.8

    def test_portfolio_return(self, sample_data, sample_weights):
        """Test portfolio return calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        port_return = portfolio.calculate_portfolio_return()

        assert isinstance(port_return, float)
        assert -1.0 < port_return < 2.0  # Reasonable annual return bounds

    def test_portfolio_volatility(self, sample_data, sample_weights):
        """Test portfolio volatility calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        port_vol = portfolio.calculate_portfolio_volatility()

        assert isinstance(port_vol, float)
        assert port_vol > 0  # Volatility must be positive
        assert port_vol < 1.0  # Reasonable annual volatility bound

    def test_portfolio_sharpe_ratio(self, sample_data, sample_weights):
        """Test Sharpe ratio calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        sharpe = portfolio.calculate_portfolio_sharpe_ratio(risk_free_rate=0.02)

        assert isinstance(sharpe, float)
        assert -5.0 < sharpe < 5.0  # Reasonable Sharpe ratio bounds

    def test_max_drawdown(self, sample_data, sample_weights):
        """Test max drawdown calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        max_dd = portfolio.calculate_max_drawdown()

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown is always negative or zero
        assert max_dd >= -1.0  # Can't lose more than 100%

    def test_portfolio_returns_series(self, sample_data, sample_weights):
        """Test daily returns calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        returns = portfolio.calculate_portfolio_returns()

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_data) - 1  # One less due to pct_change

    def test_cumulative_returns(self, sample_data, sample_weights):
        """Test cumulative returns calculation."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        cum_returns = portfolio.calculate_cumulative_returns()

        assert isinstance(cum_returns, pd.Series)
        assert cum_returns.iloc[0] > 0  # Starts positive

    def test_get_summary(self, sample_data, sample_weights):
        """Test summary dictionary."""
        portfolio = PortfolioAnalysis(sample_data, sample_weights)
        summary = portfolio.get_summary()

        assert isinstance(summary, dict)
        assert 'annual_return' in summary
        assert 'annual_volatility' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown' in summary


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_annual_return(self, sample_data):
        """Test annual return calculation."""
        annual_ret = PerformanceMetrics.calculate_annual_return(sample_data)

        assert isinstance(annual_ret, pd.Series)
        assert len(annual_ret) == 2

    def test_annual_volatility(self, sample_data):
        """Test annual volatility calculation."""
        annual_vol = PerformanceMetrics.calculate_annual_volatility(sample_data)

        assert isinstance(annual_vol, pd.Series)
        assert (annual_vol > 0).all()

    def test_sharpe_ratio(self, sample_data):
        """Test Sharpe ratio calculation."""
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(sample_data, risk_free_rate=0.02)

        assert isinstance(sharpe, pd.Series)

    def test_max_drawdown(self, sample_data):
        """Test max drawdown calculation."""
        max_dd = PerformanceMetrics.calculate_max_drawdown(sample_data)

        assert isinstance(max_dd, pd.Series)
        assert (max_dd <= 0).all()

    def test_var(self, sample_data):
        """Test Value at Risk calculation."""
        var = PerformanceMetrics.calculate_var(sample_data, confidence_level=0.95)

        assert isinstance(var, pd.Series)
        assert (var < 0).all()  # VaR is typically negative


class TestHelpers:
    """Tests for utility functions."""

    def test_validate_weights_valid(self):
        """Test valid weights."""
        assert validate_weights([0.5, 0.3, 0.2])
        assert validate_weights([1.0])
        assert validate_weights([0.25, 0.25, 0.25, 0.25])

    def test_validate_weights_invalid(self):
        """Test invalid weights."""
        assert not validate_weights([0.5, 0.3])  # Sums to 0.8
        assert not validate_weights([0.5, 0.6])  # Sums to 1.1

    def test_normalize_weights(self):
        """Test weight normalization."""
        normalized = normalize_weights([2, 3, 5])
        np.testing.assert_array_almost_equal(normalized, [0.2, 0.3, 0.5])
        assert np.isclose(normalized.sum(), 1.0)


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""

    def test_simulation(self, sample_data, sample_weights):
        """Test Monte Carlo simulation runs."""
        from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
            initial_investment=10000
        )

        results = mc.simulate()

        assert results.shape == (100, 21)
        assert (results > 0).all()  # Portfolio values should be positive

    def test_statistics(self, sample_data, sample_weights):
        """Test Monte Carlo statistics."""
        from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
            initial_investment=10000
        )

        stats = mc.get_statistics()

        assert 'final_values' in stats
        assert 'mean' in stats['final_values']
        assert 'median' in stats['final_values']
        assert 'prob_loss' in stats['final_values']
        assert 0 <= stats['final_values']['prob_loss'] <= 100
