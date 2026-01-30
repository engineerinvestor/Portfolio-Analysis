"""Tests for factor analysis functionality."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from portfolio_analysis.factors.data import FactorDataLoader, align_returns_with_factors
from portfolio_analysis.factors.models import (
    FactorModel,
    RegressionResults,
    FactorRegression,
)
from portfolio_analysis.factors.exposures import FactorExposures
from portfolio_analysis.factors.attribution import FactorAttribution
from portfolio_analysis.factors.optimization import FactorOptimizer
from portfolio_analysis.factors.visualization import FactorVisualization


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dates():
    """Create sample date range."""
    return pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')


@pytest.fixture
def sample_factor_data(sample_dates):
    """Create sample Fama-French factor data."""
    np.random.seed(42)
    n = len(sample_dates)

    data = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0004, 0.01, n),
        'SMB': np.random.normal(0.0001, 0.005, n),
        'HML': np.random.normal(0.0001, 0.004, n),
        'RMW': np.random.normal(0.0001, 0.003, n),
        'CMA': np.random.normal(0.0001, 0.003, n),
        'MOM': np.random.normal(0.0002, 0.008, n),
        'RF': np.full(n, 0.0001),  # ~2.5% annual risk-free rate
    }, index=sample_dates)

    return data


@pytest.fixture
def sample_returns(sample_dates, sample_factor_data):
    """Create sample portfolio returns correlated with factors."""
    np.random.seed(42)
    n = len(sample_dates)

    # Returns = alpha + beta_mkt * Mkt + beta_smb * SMB + beta_hml * HML + epsilon
    alpha = 0.0002  # ~5% annual alpha
    beta_mkt = 1.1
    beta_smb = 0.3
    beta_hml = 0.2
    epsilon = np.random.normal(0, 0.005, n)

    returns = (
        alpha +
        beta_mkt * sample_factor_data['Mkt-RF'] +
        beta_smb * sample_factor_data['SMB'] +
        beta_hml * sample_factor_data['HML'] +
        sample_factor_data['RF'] +  # Add back risk-free for total returns
        epsilon
    )

    return pd.Series(returns, index=sample_dates, name='Portfolio')


@pytest.fixture
def sample_price_data(sample_dates):
    """Create sample multi-asset price data."""
    np.random.seed(42)
    n = len(sample_dates)

    # Create returns for 5 assets with different factor exposures
    assets = {}

    # Large-cap value (low SMB, high HML)
    assets['LCV'] = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))

    # Small-cap growth (high SMB, low HML)
    assets['SCG'] = 100 * np.cumprod(1 + np.random.normal(0.0006, 0.015, n))

    # Market-like
    assets['MKT'] = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, n))

    # Bonds (low beta)
    assets['BND'] = 100 * np.cumprod(1 + np.random.normal(0.0002, 0.003, n))

    # High momentum
    assets['MOM'] = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.012, n))

    return pd.DataFrame(assets, index=sample_dates)


# ============================================================================
# Tests for FactorModel Enum
# ============================================================================

class TestFactorModel:
    """Tests for FactorModel enum."""

    def test_capm_factors(self):
        """Test CAPM factor list."""
        assert FactorModel.CAPM.value == ['Mkt-RF']

    def test_ff3_factors(self):
        """Test FF3 factor list."""
        assert FactorModel.FF3.value == ['Mkt-RF', 'SMB', 'HML']

    def test_ff5_factors(self):
        """Test FF5 factor list."""
        assert FactorModel.FF5.value == ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    def test_carhart_factors(self):
        """Test Carhart factor list."""
        assert FactorModel.CARHART.value == ['Mkt-RF', 'SMB', 'HML', 'MOM']


# ============================================================================
# Tests for RegressionResults
# ============================================================================

class TestRegressionResults:
    """Tests for RegressionResults dataclass."""

    def test_creation(self):
        """Test creating RegressionResults."""
        results = RegressionResults(
            alpha=0.05,
            alpha_pvalue=0.02,
            betas={'Mkt-RF': 1.1, 'SMB': 0.3},
            beta_pvalues={'Mkt-RF': 0.001, 'SMB': 0.05},
            beta_tstats={'Mkt-RF': 5.5, 'SMB': 2.1},
            r_squared=0.85,
            adj_r_squared=0.84,
            residual_std=0.10,
            n_observations=250,
            model='FF3',
            factors=['Mkt-RF', 'SMB']
        )

        assert results.alpha == 0.05
        assert results.betas['Mkt-RF'] == 1.1
        assert results.r_squared == 0.85
        assert results.model == 'FF3'

    def test_summary(self):
        """Test summary generation."""
        results = RegressionResults(
            alpha=0.05,
            alpha_pvalue=0.02,
            betas={'Mkt-RF': 1.1},
            beta_pvalues={'Mkt-RF': 0.001},
            beta_tstats={'Mkt-RF': 5.5},
            r_squared=0.85,
            adj_r_squared=0.84,
            residual_std=0.10,
            n_observations=250,
            model='CAPM',
            factors=['Mkt-RF']
        )

        summary = results.summary()
        assert 'CAPM' in summary
        assert 'Alpha' in summary
        assert 'Mkt-RF' in summary

    def test_repr(self):
        """Test string representation."""
        results = RegressionResults(
            alpha=0.05,
            alpha_pvalue=0.02,
            betas={'Mkt-RF': 1.1},
            beta_pvalues={'Mkt-RF': 0.001},
            beta_tstats={'Mkt-RF': 5.5},
            r_squared=0.85,
            adj_r_squared=0.84,
            residual_std=0.10,
            n_observations=250,
            model='CAPM',
            factors=['Mkt-RF']
        )

        repr_str = repr(results)
        assert 'CAPM' in repr_str
        assert '0.85' in repr_str or '0.8' in repr_str


# ============================================================================
# Tests for align_returns_with_factors
# ============================================================================

class TestAlignReturnsWithFactors:
    """Tests for the align_returns_with_factors utility."""

    def test_basic_alignment(self, sample_returns, sample_factor_data):
        """Test basic alignment works."""
        excess_returns, factors = align_returns_with_factors(
            sample_returns, sample_factor_data
        )

        assert len(excess_returns) == len(factors)
        assert len(excess_returns) > 0

    def test_computes_excess_returns(self, sample_returns, sample_factor_data):
        """Test that excess returns are computed correctly."""
        excess_returns, factors = align_returns_with_factors(
            sample_returns, sample_factor_data, compute_excess=True
        )

        # Excess returns should be different from raw returns
        raw_aligned = sample_returns.loc[excess_returns.index]
        rf_aligned = factors['RF']

        expected_excess = raw_aligned - rf_aligned
        np.testing.assert_array_almost_equal(
            excess_returns.values, expected_excess.values
        )

    def test_no_overlap_raises_error(self, sample_factor_data):
        """Test that no date overlap raises error."""
        # Create returns with different dates
        different_dates = pd.date_range('1990-01-01', '1990-12-31', freq='B')
        returns = pd.Series(np.random.normal(0, 0.01, len(different_dates)),
                           index=different_dates)

        with pytest.raises(ValueError, match="No overlapping dates"):
            align_returns_with_factors(returns, sample_factor_data)


# ============================================================================
# Tests for FactorRegression
# ============================================================================

class TestFactorRegression:
    """Tests for FactorRegression class."""

    def test_initialization(self, sample_returns, sample_factor_data):
        """Test FactorRegression initialization."""
        regression = FactorRegression(sample_returns, sample_factor_data)

        assert regression is not None
        assert len(regression.excess_returns) > 0

    def test_capm_regression(self, sample_returns, sample_factor_data):
        """Test CAPM regression."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('capm')

        assert isinstance(results, RegressionResults)
        assert 'Mkt-RF' in results.betas
        assert 0 < results.r_squared < 1
        assert results.model == 'CAPM'

    def test_ff3_regression(self, sample_returns, sample_factor_data):
        """Test FF3 regression."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('ff3')

        assert 'Mkt-RF' in results.betas
        assert 'SMB' in results.betas
        assert 'HML' in results.betas
        assert results.model == 'FF3'

    def test_ff5_regression(self, sample_returns, sample_factor_data):
        """Test FF5 regression."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('ff5')

        assert 'RMW' in results.betas
        assert 'CMA' in results.betas
        assert results.model == 'FF5'

    def test_carhart_regression(self, sample_returns, sample_factor_data):
        """Test Carhart regression."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('carhart')

        assert 'MOM' in results.betas
        assert results.model == 'CARHART'

    def test_invalid_model_raises_error(self, sample_returns, sample_factor_data):
        """Test that invalid model name raises error."""
        regression = FactorRegression(sample_returns, sample_factor_data)

        with pytest.raises(ValueError, match="Unknown model"):
            regression.run_regression('invalid_model')

    def test_rolling_regression(self, sample_returns, sample_factor_data):
        """Test rolling regression."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        rolling = regression.run_rolling_regression('ff3', window=60)

        assert isinstance(rolling, pd.DataFrame)
        assert 'alpha' in rolling.columns
        assert 'Mkt-RF' in rolling.columns
        assert len(rolling) > 0

    def test_compare_models(self, sample_returns, sample_factor_data):
        """Test model comparison."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        comparison = regression.compare_models()

        assert isinstance(comparison, pd.DataFrame)
        assert 'Model' in comparison.columns
        assert 'R-squared' in comparison.columns
        assert len(comparison) >= 2  # At least CAPM and FF3

    def test_beta_significance(self, sample_returns, sample_factor_data):
        """Test that p-values are computed for betas."""
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('ff3')

        for factor in results.factors:
            assert factor in results.beta_pvalues
            assert 0 <= results.beta_pvalues[factor] <= 1


# ============================================================================
# Tests for FactorExposures
# ============================================================================

class TestFactorExposures:
    """Tests for FactorExposures class."""

    def test_initialization(self):
        """Test FactorExposures initialization."""
        exposures = FactorExposures(['AAPL', 'MSFT'], [0.6, 0.4])

        assert exposures is not None
        assert len(exposures.tickers) == 2
        np.testing.assert_array_almost_equal(exposures.weights, [0.6, 0.4])

    def test_invalid_weights_length(self):
        """Test that mismatched tickers/weights raises error."""
        with pytest.raises(ValueError, match="must match"):
            FactorExposures(['AAPL', 'MSFT'], [0.5, 0.3, 0.2])

    def test_weights_not_summing_to_one(self):
        """Test that weights not summing to 1 raises error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            FactorExposures(['AAPL', 'MSFT'], [0.5, 0.3])

    def test_get_all_tilts_returns_dict(self):
        """Test that get_all_tilts returns expected structure."""
        with patch.object(FactorExposures, '_fetch_characteristics') as mock_fetch:
            # Mock characteristics data
            mock_fetch.return_value = pd.DataFrame({
                'market_cap': [2e12, 1e11],
                'pe_ratio': [25, 15],
                'pb_ratio': [3.5, 1.5],
                'dividend_yield': [0.01, 0.02],
                'beta': [1.2, 0.8],
                'profit_margin': [0.20, 0.15],
                'roe': [0.18, 0.12],
                'debt_to_equity': [50, 80],
                'revenue_growth': [0.15, 0.05],
                'earnings_growth': [0.12, 0.08],
            }, index=['AAPL', 'MSFT'])

            with patch.object(FactorExposures, '_calculate_momentum') as mock_mom:
                mock_mom.return_value = pd.Series([0.15, 0.05], index=['AAPL', 'MSFT'])

                exposures = FactorExposures(['AAPL', 'MSFT'], [0.6, 0.4])
                tilts = exposures.get_all_tilts()

                assert isinstance(tilts, dict)
                assert 'size' in tilts
                assert 'value' in tilts
                assert 'momentum' in tilts
                assert 'quality' in tilts
                assert 'investment' in tilts

    def test_tilt_values_in_range(self):
        """Test that tilts are in expected range [-1, 1]."""
        with patch.object(FactorExposures, '_fetch_characteristics') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'market_cap': [2e12, 5e9],
                'pe_ratio': [30, 12],
                'pb_ratio': [4.0, 1.0],
                'dividend_yield': [0.01, 0.03],
                'beta': [1.3, 0.7],
                'profit_margin': [0.25, 0.10],
                'roe': [0.20, 0.08],
                'debt_to_equity': [30, 100],
                'revenue_growth': [0.25, 0.03],
                'earnings_growth': [0.20, 0.05],
            }, index=['GROWTH', 'VALUE'])

            with patch.object(FactorExposures, '_calculate_momentum') as mock_mom:
                mock_mom.return_value = pd.Series([0.25, -0.05], index=['GROWTH', 'VALUE'])

                exposures = FactorExposures(['GROWTH', 'VALUE'], [0.5, 0.5])
                tilts = exposures.get_all_tilts()

                for factor, tilt in tilts.items():
                    assert -1.5 <= tilt <= 1.5, f"{factor} tilt {tilt} out of range"

    def test_summary(self):
        """Test summary generation."""
        with patch.object(FactorExposures, 'get_all_tilts') as mock_tilts:
            mock_tilts.return_value = {
                'size': -0.5,
                'value': 0.3,
                'momentum': 0.1,
                'quality': 0.4,
                'investment': -0.2,
            }

            exposures = FactorExposures(['A', 'B'], [0.5, 0.5])
            summary = exposures.summary()

            assert 'Size' in summary
            assert 'Value' in summary
            assert 'Large Cap' in summary or 'Small Cap' in summary


# ============================================================================
# Tests for FactorAttribution
# ============================================================================

class TestFactorAttribution:
    """Tests for FactorAttribution class."""

    def test_initialization(self, sample_returns, sample_factor_data):
        """Test FactorAttribution initialization."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        assert attribution is not None

    def test_decompose_returns(self, sample_returns, sample_factor_data):
        """Test return decomposition."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        decomp = attribution.decompose_returns('ff3')

        assert isinstance(decomp, dict)
        assert 'total' in decomp
        assert 'Mkt-RF' in decomp
        assert 'SMB' in decomp
        assert 'HML' in decomp
        assert 'alpha' in decomp

    def test_decompose_returns_sums_correctly(self, sample_returns, sample_factor_data):
        """Test that return components sum to approximately total."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        decomp = attribution.decompose_returns('ff3')

        # Sum of components should approximate total
        # (excluding 'total' itself and 'risk_free' which is already in factor returns)
        component_sum = (
            decomp['risk_free'] +
            decomp['Mkt-RF'] +
            decomp['SMB'] +
            decomp['HML'] +
            decomp['alpha']
        )

        # Allow for some numerical imprecision
        assert abs(component_sum - decomp['total']) < 0.05

    def test_decompose_risk(self, sample_returns, sample_factor_data):
        """Test risk decomposition."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        risk_decomp = attribution.decompose_risk('ff3')

        assert isinstance(risk_decomp, dict)
        assert 'total' in risk_decomp
        assert 'Mkt-RF' in risk_decomp
        assert 'idiosyncratic' in risk_decomp
        assert 'r_squared' in risk_decomp

    def test_risk_decomposition_positive_values(self, sample_returns, sample_factor_data):
        """Test that variance contributions are non-negative."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        risk_decomp = attribution.decompose_risk('ff3')

        assert risk_decomp['total'] > 0
        for factor in ['Mkt-RF', 'SMB', 'HML']:
            assert risk_decomp[factor] >= 0

    def test_get_attribution_summary(self, sample_returns, sample_factor_data):
        """Test attribution summary table."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        summary = attribution.get_attribution_summary('ff3')

        assert isinstance(summary, pd.DataFrame)
        assert 'Component' in summary.columns
        assert 'Return (%)' in summary.columns

    def test_rolling_attribution(self, sample_returns, sample_factor_data):
        """Test rolling attribution."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        rolling = attribution.get_rolling_attribution('ff3', window=60)

        assert isinstance(rolling, pd.DataFrame)
        assert 'alpha' in rolling.columns
        assert len(rolling) > 0

    def test_summary_text(self, sample_returns, sample_factor_data):
        """Test text summary generation."""
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        summary = attribution.summary('ff3')

        assert isinstance(summary, str)
        assert 'RETURN ATTRIBUTION' in summary
        assert 'RISK ATTRIBUTION' in summary


# ============================================================================
# Tests for FactorOptimizer
# ============================================================================

class TestFactorOptimizer:
    """Tests for FactorOptimizer class."""

    def test_initialization(self, sample_price_data, sample_factor_data):
        """Test FactorOptimizer initialization."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)

        assert optimizer is not None
        assert optimizer.n_assets == 5

    def test_asset_betas_computed(self, sample_price_data, sample_factor_data):
        """Test that individual asset betas are computed."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        betas = optimizer.get_asset_betas()

        assert isinstance(betas, pd.DataFrame)
        assert 'Mkt-RF' in betas.columns
        assert len(betas) == 5

    def test_optimize_target_exposures(self, sample_price_data, sample_factor_data):
        """Test target exposure optimization."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        result = optimizer.optimize_target_exposures(
            target_betas={'Mkt-RF': 1.0},
            tolerance=0.2
        )

        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'achieved_betas' in result
        assert 'sharpe_ratio' in result

        # Weights should sum to 1
        weight_sum = sum(result['weights'].values())
        assert abs(weight_sum - 1.0) < 0.01

    def test_optimize_factor_neutral(self, sample_price_data, sample_factor_data):
        """Test factor-neutral optimization."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        result = optimizer.optimize_factor_neutral(
            factors=['SMB'],
            tolerance=0.1
        )

        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'neutralized_factors' in result
        assert 'SMB' in result['neutralized_factors']

    def test_optimize_max_alpha(self, sample_price_data, sample_factor_data):
        """Test max alpha optimization."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        result = optimizer.optimize_max_alpha('ff3')

        assert isinstance(result, dict)
        assert 'expected_alpha' in result
        assert 'asset_alphas' in result
        assert 'weights' in result

    def test_generate_factor_frontier(self, sample_price_data, sample_factor_data):
        """Test factor frontier generation."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        frontier = optimizer.generate_factor_frontier('Mkt-RF', n_points=5)

        assert isinstance(frontier, pd.DataFrame)
        # May have fewer points if optimization fails for some targets

    def test_invalid_factor_raises_error(self, sample_price_data, sample_factor_data):
        """Test that invalid factor name raises error."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)

        with pytest.raises(ValueError, match="not available"):
            optimizer.optimize_target_exposures(
                target_betas={'INVALID_FACTOR': 1.0}
            )

    def test_summary(self, sample_price_data, sample_factor_data):
        """Test summary generation."""
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        summary = optimizer.summary()

        assert isinstance(summary, str)
        assert 'Assets:' in summary
        assert 'Available factors:' in summary


# ============================================================================
# Tests for FactorVisualization (just test they don't error)
# ============================================================================

class TestFactorVisualization:
    """Tests for FactorVisualization class."""

    def test_plot_factor_exposures(self, sample_returns, sample_factor_data):
        """Test factor exposures plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('ff3')

        # Just test it doesn't raise an error
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode

        try:
            FactorVisualization.plot_factor_exposures(results)
        finally:
            plt.close('all')

    def test_plot_rolling_betas(self, sample_returns, sample_factor_data):
        """Test rolling betas plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

        regression = FactorRegression(sample_returns, sample_factor_data)
        rolling = regression.run_rolling_regression('ff3', window=60)

        try:
            FactorVisualization.plot_rolling_betas(rolling)
        finally:
            plt.close('all')

    def test_plot_return_attribution(self, sample_returns, sample_factor_data):
        """Test return attribution plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

        attribution = FactorAttribution(sample_returns, sample_factor_data)
        decomp = attribution.decompose_returns('ff3')

        try:
            FactorVisualization.plot_return_attribution(decomp)
        finally:
            plt.close('all')

    def test_plot_factor_tilts(self):
        """Test factor tilts plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

        tilts = {
            'size': 0.3,
            'value': -0.2,
            'momentum': 0.1,
            'quality': 0.5,
            'investment': -0.1
        }

        try:
            FactorVisualization.plot_factor_tilts(tilts)
        finally:
            plt.close('all')

    def test_plot_model_comparison(self, sample_returns, sample_factor_data):
        """Test model comparison plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

        regression = FactorRegression(sample_returns, sample_factor_data)
        comparison = regression.compare_models()

        try:
            FactorVisualization.plot_model_comparison(comparison)
        finally:
            plt.close('all')

    def test_plot_risk_attribution(self, sample_returns, sample_factor_data):
        """Test risk attribution plot doesn't error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

        attribution = FactorAttribution(sample_returns, sample_factor_data)
        risk_decomp = attribution.decompose_risk('ff3')

        try:
            FactorVisualization.plot_risk_attribution(risk_decomp)
        finally:
            plt.close('all')


# ============================================================================
# Integration Tests
# ============================================================================

class TestFactorIntegration:
    """Integration tests for the factor analysis module."""

    def test_full_factor_analysis_workflow(self, sample_returns, sample_factor_data):
        """Test complete workflow from data to analysis."""
        # 1. Run regression
        regression = FactorRegression(sample_returns, sample_factor_data)
        results = regression.run_regression('ff3')

        assert results.r_squared > 0.5  # Should explain significant variance

        # 2. Attribution
        attribution = FactorAttribution(sample_returns, sample_factor_data)
        return_decomp = attribution.decompose_returns('ff3')

        assert 'total' in return_decomp
        assert 'alpha' in return_decomp

        # 3. Compare models
        comparison = regression.compare_models()
        assert len(comparison) >= 2

    def test_optimization_with_factor_analysis(
        self, sample_price_data, sample_factor_data
    ):
        """Test factor-aware optimization workflow."""
        # Optimize for target exposures
        optimizer = FactorOptimizer(sample_price_data, sample_factor_data)
        result = optimizer.optimize_target_exposures(
            target_betas={'Mkt-RF': 0.8, 'SMB': 0.2},
            tolerance=0.3
        )

        # Verify weights are valid
        weights = list(result['weights'].values())
        assert abs(sum(weights) - 1.0) < 0.01
        assert all(0 <= w <= 1 for w in weights)

        # Verify betas are close to target
        achieved = result['achieved_betas']
        assert abs(achieved['Mkt-RF'] - 0.8) < 0.4
