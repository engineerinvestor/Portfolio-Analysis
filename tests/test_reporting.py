"""Tests for the HTML reporting module."""

import os
import tempfile

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

import numpy as np
import pandas as pd
import pytest

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.reporting import ReportBuilder
from portfolio_analysis.reporting.chart_utils import fig_to_base64, create_figure
from portfolio_analysis.reporting.sections import (
    HeaderSection,
    PerformanceSection,
    DrawdownSection,
    ReturnsSection,
    RiskSection,
    BenchmarkSection,
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    n_days = len(dates)

    returns1 = np.random.normal(0.0004, 0.012, n_days)
    returns2 = np.random.normal(0.0001, 0.004, n_days)

    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 100 * np.cumprod(1 + returns2)

    data = pd.DataFrame({
        'VTI': prices1,
        'BND': prices2
    }, index=dates)

    return data


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return [0.6, 0.4]


@pytest.fixture
def portfolio(sample_data, sample_weights):
    """Create a PortfolioAnalysis instance."""
    return PortfolioAnalysis(sample_data, sample_weights)


@pytest.fixture
def mock_benchmark(sample_data, sample_weights):
    """Create a mock benchmark comparison object."""
    np.random.seed(123)
    n_days = len(sample_data)

    # Create mock benchmark returns
    returns_spy = np.random.normal(0.0003, 0.011, n_days)
    prices_spy = 100 * np.cumprod(1 + returns_spy)
    spy_data = pd.Series(prices_spy, index=sample_data.index, name='SPY')

    # Calculate returns
    portfolio_returns = sample_data.pct_change().dropna().dot(sample_weights)
    benchmark_returns = spy_data.pct_change().dropna()

    # Align dates
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common]
    benchmark_returns = benchmark_returns.loc[common]

    class MockBenchmark:
        BENCHMARKS = {'SPY': 'S&P 500 (US Large Cap)'}

        def __init__(self):
            self.benchmark_ticker = 'SPY'
            self.portfolio_returns = portfolio_returns
            self.benchmark_returns = benchmark_returns
            self.risk_free_rate = 0.02

        def get_metrics(self):
            port_ret = self.portfolio_returns.mean() * 252
            bench_ret = self.benchmark_returns.mean() * 252
            port_vol = self.portfolio_returns.std() * np.sqrt(252)
            bench_vol = self.benchmark_returns.std() * np.sqrt(252)

            cov = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
            var = np.var(self.benchmark_returns)
            beta = cov / var if var > 0 else 1

            rf_daily = self.risk_free_rate / 252
            alpha = (self.portfolio_returns.mean() - (rf_daily + beta * (self.benchmark_returns.mean() - rf_daily))) * 252

            corr = np.corrcoef(self.portfolio_returns, self.benchmark_returns)[0, 1]

            active = self.portfolio_returns - self.benchmark_returns
            te = active.std() * np.sqrt(252)
            ir = (port_ret - bench_ret) / te if te > 0 else 0

            up = self.benchmark_returns > 0
            down = self.benchmark_returns < 0
            up_cap = (self.portfolio_returns[up].mean() / self.benchmark_returns[up].mean()) * 100 if up.any() else 0
            down_cap = (self.portfolio_returns[down].mean() / self.benchmark_returns[down].mean()) * 100 if down.any() else 0

            return {
                'portfolio_return': port_ret,
                'benchmark_return': bench_ret,
                'portfolio_volatility': port_vol,
                'benchmark_volatility': bench_vol,
                'beta': beta,
                'alpha': alpha,
                'r_squared': corr ** 2,
                'correlation': corr,
                'tracking_error': te,
                'information_ratio': ir,
                'up_capture': up_cap,
                'down_capture': down_cap,
            }

    return MockBenchmark()


class TestChartUtils:
    """Tests for chart utility functions."""

    def test_create_figure(self):
        """Test figure creation."""
        fig, ax = create_figure(figsize=(10, 6))
        assert fig is not None
        assert ax is not None
        matplotlib.pyplot.close(fig)

    def test_create_figure_custom_size(self):
        """Test figure creation with custom size."""
        fig, ax = create_figure(figsize=(8, 4))
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 4
        matplotlib.pyplot.close(fig)

    def test_fig_to_base64(self):
        """Test figure to base64 conversion."""
        fig, ax = create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        result = fig_to_base64(fig, close_fig=True)

        assert result.startswith('data:image/png;base64,')
        assert len(result) > 100  # Should have substantial content

    def test_fig_to_base64_svg(self):
        """Test figure to base64 with SVG format."""
        fig, ax = create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        result = fig_to_base64(fig, format='svg', close_fig=True)

        assert result.startswith('data:image/svg;base64,')


class TestReportBuilder:
    """Tests for ReportBuilder class."""

    def test_initialization(self, portfolio):
        """Test ReportBuilder initialization."""
        report = ReportBuilder(portfolio)
        assert report is not None
        assert report.title == "Portfolio Analysis"
        assert report.portfolio == portfolio

    def test_initialization_with_title(self, portfolio):
        """Test ReportBuilder with custom title."""
        report = ReportBuilder(portfolio, title="My Portfolio")
        assert report.title == "My Portfolio"

    def test_initialization_with_benchmark(self, portfolio, mock_benchmark):
        """Test ReportBuilder with benchmark."""
        report = ReportBuilder(portfolio, benchmark=mock_benchmark)
        assert report.benchmark == mock_benchmark

    def test_generate_creates_file(self, portfolio):
        """Test that generate() creates an HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")

            report = ReportBuilder(portfolio, title="Test Portfolio")
            result = report.generate(output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            assert result == output_path

    def test_generate_html_content(self, portfolio):
        """Test that generated HTML contains expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")

            report = ReportBuilder(portfolio, title="Test Portfolio")
            report.generate(output_path)

            with open(output_path, 'r') as f:
                html = f.read()

            # Check for key elements
            assert '<!DOCTYPE html>' in html
            assert 'Test Portfolio' in html
            assert 'Cumulative Returns' in html
            assert 'Drawdown' in html
            assert 'Monthly Returns' in html
            assert 'Risk Analysis' in html
            assert 'QuantStats' in html  # Attribution

    def test_generate_with_benchmark(self, portfolio, mock_benchmark):
        """Test report generation with benchmark section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")

            report = ReportBuilder(portfolio, benchmark=mock_benchmark)
            report.generate(output_path)

            with open(output_path, 'r') as f:
                html = f.read()

            assert 'Benchmark Comparison' in html
            assert 'SPY' in html
            assert 'Alpha' in html
            assert 'Beta' in html

    def test_to_html(self, portfolio):
        """Test to_html() returns HTML string."""
        report = ReportBuilder(portfolio)
        html = report.to_html()

        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert len(html) > 1000

    def test_embedded_charts(self, portfolio):
        """Test that charts are embedded as base64."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")

            report = ReportBuilder(portfolio)
            report.generate(output_path)

            with open(output_path, 'r') as f:
                html = f.read()

            # Check for embedded images
            assert 'data:image/png;base64,' in html


class TestHeaderSection:
    """Tests for HeaderSection."""

    def test_compute_data(self, portfolio):
        """Test header section data computation."""
        from jinja2 import Template
        template = Template("{{ title }}")

        section = HeaderSection(
            portfolio,
            template,
            title="Test Portfolio",
            tickers=['VTI', 'BND']
        )
        data = section.compute_data()

        assert data['title'] == "Test Portfolio"
        assert 'start_date' in data
        assert 'end_date' in data
        assert 'allocation' in data
        assert 'metrics' in data
        assert len(data['allocation']) == 2

    def test_metrics_values(self, portfolio):
        """Test that metrics have reasonable values."""
        from jinja2 import Template
        template = Template("")

        section = HeaderSection(portfolio, template)
        data = section.compute_data()

        metrics = data['metrics']
        assert 'total_return' in metrics
        assert 'cagr' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

        # Volatility should be positive
        assert metrics['annual_volatility'] > 0
        # Max drawdown should be negative
        assert metrics['max_drawdown'] <= 0


class TestPerformanceSection:
    """Tests for PerformanceSection."""

    def test_compute_data(self, portfolio):
        """Test performance section data computation."""
        from jinja2 import Template
        template = Template("")

        section = PerformanceSection(portfolio, template)
        data = section.compute_data()

        assert 'chart' in data
        assert 'total_return' in data
        assert 'best_day' in data
        assert 'worst_day' in data
        assert 'positive_days' in data
        assert 'negative_days' in data

    def test_chart_is_base64(self, portfolio):
        """Test that chart is base64 encoded."""
        from jinja2 import Template
        template = Template("")

        section = PerformanceSection(portfolio, template)
        data = section.compute_data()

        assert data['chart'].startswith('data:image/png;base64,')


class TestDrawdownSection:
    """Tests for DrawdownSection."""

    def test_compute_data(self, portfolio):
        """Test drawdown section data computation."""
        from jinja2 import Template
        template = Template("")

        section = DrawdownSection(portfolio, template)
        data = section.compute_data()

        assert 'chart' in data
        assert 'max_drawdown' in data
        assert 'max_drawdown_date' in data
        assert 'current_drawdown' in data
        assert 'worst_periods' in data

    def test_max_drawdown_negative(self, portfolio):
        """Test that max drawdown is negative or zero."""
        from jinja2 import Template
        template = Template("")

        section = DrawdownSection(portfolio, template)
        data = section.compute_data()

        assert data['max_drawdown'] <= 0

    def test_worst_periods_structure(self, portfolio):
        """Test worst periods table structure."""
        from jinja2 import Template
        template = Template("")

        section = DrawdownSection(portfolio, template, top_n=3)
        data = section.compute_data()

        assert len(data['worst_periods']) <= 3
        if data['worst_periods']:
            period = data['worst_periods'][0]
            assert 'start' in period
            assert 'trough' in period
            assert 'drawdown' in period


class TestReturnsSection:
    """Tests for ReturnsSection."""

    def test_compute_data(self, portfolio):
        """Test returns section data computation."""
        from jinja2 import Template
        template = Template("")

        section = ReturnsSection(portfolio, template)
        data = section.compute_data()

        assert 'chart' in data
        assert 'best_month' in data
        assert 'worst_month' in data
        assert 'avg_monthly_return' in data
        assert 'win_rate' in data
        assert 'annual_returns' in data

    def test_win_rate_range(self, portfolio):
        """Test that win rate is between 0 and 100."""
        from jinja2 import Template
        template = Template("")

        section = ReturnsSection(portfolio, template)
        data = section.compute_data()

        assert 0 <= data['win_rate'] <= 100

    def test_annual_returns_structure(self, portfolio):
        """Test annual returns structure."""
        from jinja2 import Template
        template = Template("")

        section = ReturnsSection(portfolio, template)
        data = section.compute_data()

        assert len(data['annual_returns']) > 0
        for yr in data['annual_returns']:
            assert 'year' in yr
            assert 'return' in yr


class TestRiskSection:
    """Tests for RiskSection."""

    def test_compute_data(self, portfolio):
        """Test risk section data computation."""
        from jinja2 import Template
        template = Template("")

        section = RiskSection(portfolio, template)
        data = section.compute_data()

        assert 'risk_metrics' in data
        assert 'return_metrics' in data
        assert 'ratio_metrics' in data

    def test_metrics_structure(self, portfolio):
        """Test metrics table structure."""
        from jinja2 import Template
        template = Template("")

        section = RiskSection(portfolio, template)
        data = section.compute_data()

        # Check each metric has name and value
        for metric in data['risk_metrics']:
            assert 'name' in metric
            assert 'value' in metric

    def test_var_values(self, portfolio):
        """Test that VaR metrics exist."""
        from jinja2 import Template
        template = Template("")

        section = RiskSection(portfolio, template)
        data = section.compute_data()

        metric_names = [m['name'] for m in data['risk_metrics']]
        assert 'VaR (95%)' in metric_names
        assert 'CVaR (95%)' in metric_names


class TestBenchmarkSection:
    """Tests for BenchmarkSection."""

    def test_compute_data(self, mock_benchmark):
        """Test benchmark section data computation."""
        from jinja2 import Template
        template = Template("")

        section = BenchmarkSection(mock_benchmark, template)
        data = section.compute_data()

        assert 'benchmark_ticker' in data
        assert 'benchmark_name' in data
        assert 'chart' in data
        assert 'return_metrics' in data
        assert 'risk_metrics' in data
        assert 'capm_metrics' in data
        assert 'performance_metrics' in data

    def test_benchmark_ticker(self, mock_benchmark):
        """Test benchmark ticker is correct."""
        from jinja2 import Template
        template = Template("")

        section = BenchmarkSection(mock_benchmark, template)
        data = section.compute_data()

        assert data['benchmark_ticker'] == 'SPY'
        assert 'S&P 500' in data['benchmark_name']
