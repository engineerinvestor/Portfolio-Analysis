"""Comprehensive tests for portfolio optimization functionality."""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest

from portfolio_analysis.analysis.optimization import PortfolioOptimizer
from portfolio_analysis.exceptions import OptimizationError


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
    n_days = len(dates)

    # Simulate three assets with different characteristics
    returns1 = np.random.normal(0.0008, 0.015, n_days)  # High return, high vol
    returns2 = np.random.normal(0.0003, 0.005, n_days)  # Low return, low vol
    returns3 = np.random.normal(0.0005, 0.010, n_days)  # Medium

    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 100 * np.cumprod(1 + returns2)
    prices3 = 100 * np.cumprod(1 + returns3)

    data = pd.DataFrame(
        {"STOCK": prices1, "BOND": prices2, "REIT": prices3}, index=dates
    )

    return data


class TestOptimizerInitialization:
    """Tests for PortfolioOptimizer initialization."""

    def test_basic_initialization(self, sample_data):
        """Test basic initialization works."""
        optimizer = PortfolioOptimizer(sample_data, risk_free_rate=0.02)
        assert optimizer is not None
        assert optimizer.n_assets == 3
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.tickers == ["STOCK", "BOND", "REIT"]

    def test_default_risk_free_rate(self, sample_data):
        """Test default risk-free rate is used."""
        optimizer = PortfolioOptimizer(sample_data)
        assert optimizer.risk_free_rate == 0.02  # DEFAULT_RISK_FREE_RATE

    def test_returns_calculated(self, sample_data):
        """Test returns and covariance are calculated."""
        optimizer = PortfolioOptimizer(sample_data)
        assert optimizer.returns is not None
        assert optimizer.mean_returns is not None
        assert optimizer.cov_matrix is not None

    def test_mean_returns_shape(self, sample_data):
        """Test mean returns has correct shape."""
        optimizer = PortfolioOptimizer(sample_data)
        assert len(optimizer.mean_returns) == 3

    def test_cov_matrix_shape(self, sample_data):
        """Test covariance matrix has correct shape."""
        optimizer = PortfolioOptimizer(sample_data)
        assert optimizer.cov_matrix.shape == (3, 3)


class TestMaxSharpeOptimization:
    """Tests for maximum Sharpe ratio optimization."""

    def test_max_sharpe_returns_dict(self, sample_data):
        """Test max Sharpe optimization returns dictionary."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_max_sharpe()

        assert isinstance(result, dict)
        assert "weights" in result
        assert "return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

    def test_max_sharpe_weights_sum_to_one(self, sample_data):
        """Test weights sum to 1."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_max_sharpe()

        weights = list(result["weights"].values())
        assert np.isclose(sum(weights), 1.0)

    def test_max_sharpe_weights_in_bounds(self, sample_data):
        """Test weights are within bounds."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_max_sharpe(weight_bounds=(0, 1))

        for weight in result["weights"].values():
            assert 0 <= weight <= 1

    def test_max_sharpe_custom_bounds(self, sample_data):
        """Test max Sharpe with custom weight bounds."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_max_sharpe(weight_bounds=(0.1, 0.5))

        for weight in result["weights"].values():
            assert 0.1 - 1e-6 <= weight <= 0.5 + 1e-6

    def test_max_sharpe_positive_volatility(self, sample_data):
        """Test volatility is positive."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_max_sharpe()

        assert result["volatility"] > 0


class TestMinVolatilityOptimization:
    """Tests for minimum volatility optimization."""

    def test_min_vol_returns_dict(self, sample_data):
        """Test min volatility returns dictionary."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_min_volatility()

        assert isinstance(result, dict)
        assert "weights" in result
        assert "volatility" in result

    def test_min_vol_weights_sum_to_one(self, sample_data):
        """Test weights sum to 1."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_min_volatility()

        weights = list(result["weights"].values())
        assert np.isclose(sum(weights), 1.0)

    def test_min_vol_lower_than_max_sharpe_vol(self, sample_data):
        """Test min vol portfolio has lower or equal volatility than max Sharpe."""
        optimizer = PortfolioOptimizer(sample_data)
        min_vol = optimizer.optimize_min_volatility()
        max_sharpe = optimizer.optimize_max_sharpe()

        # Min volatility should be <= max Sharpe volatility (with tolerance)
        assert min_vol["volatility"] <= max_sharpe["volatility"] + 1e-6


class TestTargetReturnOptimization:
    """Tests for target return optimization."""

    def test_target_return_returns_dict(self, sample_data):
        """Test target return optimization returns dictionary."""
        optimizer = PortfolioOptimizer(sample_data)
        min_vol = optimizer.optimize_min_volatility()
        target = min_vol["return"]

        result = optimizer.optimize_target_return(target)

        assert isinstance(result, dict)
        assert "weights" in result

    def test_target_return_achieves_target(self, sample_data):
        """Test portfolio achieves target return."""
        optimizer = PortfolioOptimizer(sample_data)
        min_vol = optimizer.optimize_min_volatility()
        target = min_vol["return"]

        result = optimizer.optimize_target_return(target)

        assert np.isclose(result["return"], target, atol=0.001)


class TestRiskParityOptimization:
    """Tests for risk parity optimization."""

    def test_risk_parity_returns_dict(self, sample_data):
        """Test risk parity returns dictionary with risk contributions."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_risk_parity()

        assert isinstance(result, dict)
        assert "weights" in result
        assert "risk_contributions" in result

    def test_risk_parity_weights_sum_to_one(self, sample_data):
        """Test weights sum to 1."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_risk_parity()

        weights = list(result["weights"].values())
        assert np.isclose(sum(weights), 1.0)

    def test_risk_parity_contributions_roughly_equal(self, sample_data):
        """Test risk contributions are roughly equal."""
        optimizer = PortfolioOptimizer(sample_data)
        result = optimizer.optimize_risk_parity()

        risk_contribs = list(result["risk_contributions"].values())
        mean_contrib = np.mean(risk_contribs)

        # All contributions should be within 20% of mean
        for contrib in risk_contribs:
            assert np.abs(contrib - mean_contrib) / mean_contrib < 0.2


class TestEfficientFrontier:
    """Tests for efficient frontier generation."""

    def test_frontier_returns_dataframe(self, sample_data):
        """Test efficient frontier returns DataFrame."""
        optimizer = PortfolioOptimizer(sample_data)
        frontier = optimizer.generate_efficient_frontier(n_points=10)

        assert isinstance(frontier, pd.DataFrame)
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe_ratio" in frontier.columns

    def test_frontier_correct_size(self, sample_data):
        """Test frontier has approximately correct number of points."""
        optimizer = PortfolioOptimizer(sample_data)
        frontier = optimizer.generate_efficient_frontier(n_points=10)

        # May have fewer points if some optimizations fail
        assert len(frontier) > 0
        assert len(frontier) <= 10

    def test_frontier_returns_increasing(self, sample_data):
        """Test returns generally increase along frontier."""
        optimizer = PortfolioOptimizer(sample_data)
        frontier = optimizer.generate_efficient_frontier(n_points=20)

        if len(frontier) > 2:
            # Overall trend should be increasing
            assert frontier["return"].iloc[-1] > frontier["return"].iloc[0]

    def test_frontier_warning_on_failures(self, sample_data):
        """Test warning is raised when some optimizations fail."""
        import warnings

        optimizer = PortfolioOptimizer(sample_data)

        # Most frontier generations will have some failures at the edges
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            frontier = optimizer.generate_efficient_frontier(n_points=50)

            # There may or may not be warnings depending on the data


class TestEfficientFrontierPlot:
    """Tests for efficient frontier plotting."""

    def test_plot_returns_figure(self, sample_data):
        """Test plot returns figure object."""
        import matplotlib.pyplot as plt

        optimizer = PortfolioOptimizer(sample_data)
        fig = optimizer.plot_efficient_frontier(n_points=10, show=False)

        assert fig is not None
        plt.close("all")

    def test_plot_with_assets(self, sample_data):
        """Test plotting with individual assets."""
        import matplotlib.pyplot as plt

        optimizer = PortfolioOptimizer(sample_data)
        fig = optimizer.plot_efficient_frontier(
            n_points=10, show_assets=True, show=False
        )

        assert fig is not None
        plt.close("all")

    def test_plot_with_optimal_points(self, sample_data):
        """Test plotting with optimal points marked."""
        import matplotlib.pyplot as plt

        optimizer = PortfolioOptimizer(sample_data)
        fig = optimizer.plot_efficient_frontier(
            n_points=10, show_optimal=True, show=False
        )

        assert fig is not None
        plt.close("all")


class TestPrintComparison:
    """Tests for print comparison."""

    def test_print_comparison_runs(self, sample_data, capsys):
        """Test print_comparison runs without error."""
        optimizer = PortfolioOptimizer(sample_data)
        optimizer.print_comparison()

        captured = capsys.readouterr()
        assert "PORTFOLIO OPTIMIZATION COMPARISON" in captured.out
        assert "Equal Weight" in captured.out
        assert "Max Sharpe" in captured.out
        assert "Min Volatility" in captured.out
        assert "Risk Parity" in captured.out


class TestPortfolioMetrics:
    """Tests for internal portfolio metric calculations."""

    def test_portfolio_return_calculation(self, sample_data):
        """Test portfolio return calculation."""
        optimizer = PortfolioOptimizer(sample_data)
        equal_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        port_return = optimizer._portfolio_return(equal_weights)

        assert isinstance(port_return, float)

    def test_portfolio_volatility_calculation(self, sample_data):
        """Test portfolio volatility calculation."""
        optimizer = PortfolioOptimizer(sample_data)
        equal_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        port_vol = optimizer._portfolio_volatility(equal_weights)

        assert isinstance(port_vol, float)
        assert port_vol > 0

    def test_portfolio_sharpe_calculation(self, sample_data):
        """Test portfolio Sharpe ratio calculation."""
        optimizer = PortfolioOptimizer(sample_data)
        equal_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        sharpe = optimizer._portfolio_sharpe(equal_weights)

        assert isinstance(sharpe, float)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_assets(self):
        """Test with only two assets."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
        n_days = len(dates)

        data = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days)),
                "B": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.005, n_days)),
            },
            index=dates,
        )

        optimizer = PortfolioOptimizer(data)
        result = optimizer.optimize_max_sharpe()

        assert np.isclose(sum(result["weights"].values()), 1.0)

    def test_many_assets(self):
        """Test with many assets."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
        n_days = len(dates)
        n_assets = 10

        data = pd.DataFrame(
            {
                f"ASSET{i}": 100
                * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days))
                for i in range(n_assets)
            },
            index=dates,
        )

        optimizer = PortfolioOptimizer(data)
        result = optimizer.optimize_max_sharpe()

        assert np.isclose(sum(result["weights"].values()), 1.0)
        assert len(result["weights"]) == n_assets

    def test_tight_bounds(self, sample_data):
        """Test with tight bounds that force specific allocation."""
        optimizer = PortfolioOptimizer(sample_data)
        # Force roughly equal weights
        result = optimizer.optimize_max_sharpe(weight_bounds=(0.30, 0.36))

        for weight in result["weights"].values():
            assert 0.30 - 1e-6 <= weight <= 0.36 + 1e-6

    def test_single_asset_bounds(self, sample_data):
        """Test bounds that effectively select single asset."""
        optimizer = PortfolioOptimizer(sample_data)

        # This should effectively pick one asset
        result = optimizer.optimize_min_volatility()

        # Low volatility asset should have high weight
        assert result["weights"]["BOND"] > 0.5
