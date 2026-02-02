"""Comprehensive tests for Monte Carlo simulation functionality."""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest

from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation
from portfolio_analysis.exceptions import ValidationError


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
    n_days = len(dates)

    # Simulate two assets with different characteristics
    returns1 = np.random.normal(0.0005, 0.01, n_days)
    returns2 = np.random.normal(0.0003, 0.005, n_days)

    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 100 * np.cumprod(1 + returns2)

    data = pd.DataFrame({"STOCK": prices1, "BOND": prices2}, index=dates)

    return data


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return [0.6, 0.4]


class TestMonteCarloInitialization:
    """Tests for MonteCarloSimulation initialization."""

    def test_basic_initialization(self, sample_data, sample_weights):
        """Test basic initialization works."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
            initial_investment=10000,
        )
        assert mc.num_simulations == 100
        assert mc.time_horizon == 21
        assert mc.initial_investment == 10000
        np.testing.assert_array_equal(mc.weights, sample_weights)

    def test_default_parameters(self, sample_data, sample_weights):
        """Test default parameters are used."""
        mc = MonteCarloSimulation(sample_data, sample_weights)
        assert mc.num_simulations == 1000  # DEFAULT_NUM_SIMULATIONS
        assert mc.time_horizon == 252  # DEFAULT_PROJECTION_DAYS
        assert mc.initial_investment == 10000

    def test_invalid_weights_length(self, sample_data):
        """Test that mismatched weights length raises error."""
        with pytest.raises(ValidationError, match="Number of weights"):
            MonteCarloSimulation(
                sample_data, [0.5, 0.3, 0.2], num_simulations=10, time_horizon=10
            )

    def test_weights_not_summing_to_one(self, sample_data):
        """Test that weights not summing to 1 raises error."""
        with pytest.raises(ValidationError, match="sum to 1.0"):
            MonteCarloSimulation(
                sample_data, [0.5, 0.3], num_simulations=10, time_horizon=10
            )


class TestMonteCarloSimulate:
    """Tests for Monte Carlo simulation execution."""

    def test_simulation_shape(self, sample_data, sample_weights):
        """Test simulation results have correct shape."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=50,
            time_horizon=21,
        )
        results = mc.simulate()
        assert results.shape == (50, 21)

    def test_simulation_values_positive(self, sample_data, sample_weights):
        """Test all portfolio values are positive."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=50,
            time_horizon=21,
            initial_investment=10000,
        )
        results = mc.simulate()
        assert (results > 0).all()

    def test_simulation_reproducibility_with_seed(self, sample_data, sample_weights):
        """Test simulation is reproducible with numpy seed."""
        np.random.seed(123)
        mc1 = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        results1 = mc1.simulate()

        np.random.seed(123)
        mc2 = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        results2 = mc2.simulate()

        np.testing.assert_array_almost_equal(results1, results2)

    def test_simulation_starts_near_initial(self, sample_data, sample_weights):
        """Test first period values are near initial investment."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
            initial_investment=10000,
        )
        results = mc.simulate()

        # First period should be close to initial (accounting for one day return)
        first_values = results[:, 0]
        # Most should be within 5% of initial on day 1
        assert (np.abs(first_values - 10000) / 10000 < 0.05).mean() > 0.9


class TestMonteCarloStatistics:
    """Tests for Monte Carlo statistics."""

    def test_statistics_structure(self, sample_data, sample_weights):
        """Test statistics dictionary has correct structure."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
        )
        stats = mc.get_statistics()

        assert "percentiles" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "final_values" in stats

    def test_final_values_stats(self, sample_data, sample_weights):
        """Test final values statistics are present."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
            initial_investment=10000,
        )
        stats = mc.get_statistics()
        final = stats["final_values"]

        assert "mean" in final
        assert "median" in final
        assert "std" in final
        assert "min" in final
        assert "max" in final
        assert "percentile_5" in final
        assert "percentile_95" in final
        assert "prob_loss" in final

    def test_prob_loss_range(self, sample_data, sample_weights):
        """Test probability of loss is between 0 and 100."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
        )
        stats = mc.get_statistics()
        prob_loss = stats["final_values"]["prob_loss"]

        assert 0 <= prob_loss <= 100

    def test_percentile_ordering(self, sample_data, sample_weights):
        """Test percentiles are in correct order."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
        )
        stats = mc.get_statistics()
        final = stats["final_values"]

        assert final["percentile_5"] <= final["median"]
        assert final["median"] <= final["percentile_95"]
        assert final["min"] <= final["percentile_5"]
        assert final["percentile_95"] <= final["max"]

    def test_custom_percentiles(self, sample_data, sample_weights):
        """Test custom percentiles work."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=21,
        )
        stats = mc.get_statistics(percentiles=[10, 50, 90])

        assert 10 in stats["percentiles"]
        assert 50 in stats["percentiles"]
        assert 90 in stats["percentiles"]

    def test_auto_simulate_on_statistics(self, sample_data, sample_weights):
        """Test statistics auto-runs simulation if not done."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=50,
            time_horizon=21,
        )
        # Don't call simulate() first
        stats = mc.get_statistics()

        assert stats is not None
        assert mc._results is not None


class TestMonteCarloPlotting:
    """Tests for Monte Carlo plotting."""

    def test_plot_returns_axes(self, sample_data, sample_weights):
        """Test plot_simulation returns axes."""
        import matplotlib.pyplot as plt

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        ax = mc.plot_simulation(show=False)

        assert ax is not None
        plt.close("all")

    def test_plot_with_custom_axes(self, sample_data, sample_weights):
        """Test plotting on custom axes."""
        import matplotlib.pyplot as plt

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        fig, ax = plt.subplots()
        returned_ax = mc.plot_simulation(ax=ax, show=False)

        assert returned_ax is ax
        plt.close("all")

    def test_plot_without_percentiles(self, sample_data, sample_weights):
        """Test plotting without percentile bands."""
        import matplotlib.pyplot as plt

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        ax = mc.plot_simulation(show_percentiles=False, show=False)

        assert ax is not None
        plt.close("all")

    def test_plot_with_fewer_paths(self, sample_data, sample_weights):
        """Test plotting fewer paths than simulations."""
        import matplotlib.pyplot as plt

        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=100,
            time_horizon=10,
        )
        ax = mc.plot_simulation(num_paths=10, show=False)

        assert ax is not None
        plt.close("all")


class TestMonteCarloSummary:
    """Tests for Monte Carlo summary printing."""

    def test_print_summary_runs(self, sample_data, sample_weights, capsys):
        """Test print_summary runs without error."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
        )
        mc.print_summary()

        captured = capsys.readouterr()
        assert "MONTE CARLO SIMULATION SUMMARY" in captured.out
        assert "Initial Investment" in captured.out
        assert "Time Horizon" in captured.out
        assert "Probability of Loss" in captured.out


class TestMonteCarloEdgeCases:
    """Tests for edge cases."""

    def test_single_asset(self, sample_data):
        """Test with single asset portfolio."""
        single_asset = sample_data[["STOCK"]]
        mc = MonteCarloSimulation(
            single_asset,
            [1.0],
            num_simulations=10,
            time_horizon=10,
        )
        results = mc.simulate()

        assert results.shape == (10, 10)
        assert (results > 0).all()

    def test_short_time_horizon(self, sample_data, sample_weights):
        """Test with very short time horizon."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=1,
        )
        results = mc.simulate()

        assert results.shape == (10, 1)

    def test_small_initial_investment(self, sample_data, sample_weights):
        """Test with small initial investment."""
        mc = MonteCarloSimulation(
            sample_data,
            sample_weights,
            num_simulations=10,
            time_horizon=10,
            initial_investment=1,
        )
        results = mc.simulate()
        stats = mc.get_statistics()

        assert stats["final_values"]["mean"] > 0

    def test_many_assets(self):
        """Test with many assets."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
        n_days = len(dates)
        n_assets = 10

        data = pd.DataFrame(
            {
                f"ASSET{i}": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days))
                for i in range(n_assets)
            },
            index=dates,
        )
        weights = [1.0 / n_assets] * n_assets

        mc = MonteCarloSimulation(
            data,
            weights,
            num_simulations=10,
            time_horizon=10,
        )
        results = mc.simulate()

        assert results.shape == (10, 10)
        assert (results > 0).all()
