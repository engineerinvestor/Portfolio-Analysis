"""
Main ReportBuilder class for generating HTML tear sheet reports.

Design inspired by QuantStats (https://github.com/ranaroussi/quantstats)
by Ran Aroussi.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, Template

from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.metrics.benchmark import BenchmarkComparison
from portfolio_analysis.reporting.sections.header import HeaderSection
from portfolio_analysis.reporting.sections.performance import PerformanceSection
from portfolio_analysis.reporting.sections.drawdown import DrawdownSection
from portfolio_analysis.reporting.sections.returns import ReturnsSection
from portfolio_analysis.reporting.sections.risk import RiskSection
from portfolio_analysis.reporting.sections.benchmark import BenchmarkSection


class ReportBuilder:
    """
    Build professional HTML tear sheet reports for portfolio analysis.

    Generates self-contained HTML files with embedded charts and
    comprehensive performance metrics.

    Parameters
    ----------
    portfolio : PortfolioAnalysis
        The portfolio analysis object containing price data and weights
    benchmark : BenchmarkComparison, optional
        Benchmark comparison object for relative performance analysis
    title : str, optional
        Report title (default: "Portfolio Analysis")
    tickers : list of str, optional
        Ticker symbols for display (default: uses column names from data)
    risk_free_rate : float, default 0.02
        Annual risk-free rate for Sharpe ratio and other calculations

    Examples
    --------
    Basic usage:

    >>> from portfolio_analysis import DataLoader, PortfolioAnalysis
    >>> from portfolio_analysis.reporting import ReportBuilder
    >>>
    >>> loader = DataLoader(['VTI', 'BND'], '2020-01-01', '2024-01-01')
    >>> portfolio = PortfolioAnalysis(loader.fetch_data(), [0.6, 0.4])
    >>> report = ReportBuilder(portfolio, title="My Portfolio")
    >>> report.generate("tearsheet.html")

    With benchmark comparison:

    >>> from portfolio_analysis import BenchmarkComparison
    >>> benchmark = BenchmarkComparison(data, [0.6, 0.4], benchmark_ticker='SPY')
    >>> report = ReportBuilder(portfolio, benchmark=benchmark)
    >>> report.generate("tearsheet.html")

    Notes
    -----
    The design and structure of these reports are inspired by QuantStats
    (https://github.com/ranaroussi/quantstats) by Ran Aroussi.
    """

    def __init__(
        self,
        portfolio: PortfolioAnalysis,
        benchmark: Optional[BenchmarkComparison] = None,
        title: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        risk_free_rate: float = 0.02
    ):
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.title = title or "Portfolio Analysis"
        self.tickers = tickers or list(portfolio.data.columns)
        self.risk_free_rate = risk_free_rate

        # Set up Jinja2 environment
        templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader([
                str(templates_dir),
                str(templates_dir / "sections")
            ]),
            autoescape=True
        )

    def _load_template(self, name: str) -> Template:
        """Load a template by name."""
        return self.env.get_template(name)

    def _build_sections(self) -> List[str]:
        """Build all report sections and return their HTML."""
        sections = []

        # Header section
        header = HeaderSection(
            self.portfolio,
            self._load_template("header.html"),
            title=self.title,
            tickers=self.tickers
        )
        sections.append(header.render())

        # Performance section
        performance = PerformanceSection(
            self.portfolio,
            self._load_template("performance.html")
        )
        sections.append(performance.render())

        # Drawdown section
        drawdown = DrawdownSection(
            self.portfolio,
            self._load_template("drawdown.html")
        )
        sections.append(drawdown.render())

        # Returns heatmap section
        returns = ReturnsSection(
            self.portfolio,
            self._load_template("returns_heatmap.html")
        )
        sections.append(returns.render())

        # Risk metrics section
        risk = RiskSection(
            self.portfolio,
            self._load_template("risk_metrics.html"),
            risk_free_rate=self.risk_free_rate
        )
        sections.append(risk.render())

        # Benchmark section (optional)
        if self.benchmark is not None:
            benchmark_section = BenchmarkSection(
                self.benchmark,
                self._load_template("benchmark.html")
            )
            sections.append(benchmark_section.render())

        return sections

    def generate(self, output_path: str) -> str:
        """
        Generate the HTML tear sheet and save to file.

        Parameters
        ----------
        output_path : str
            Path where the HTML file will be saved

        Returns
        -------
        str
            The absolute path to the generated file
        """
        # Build all sections
        section_html = "\n".join(self._build_sections())

        # Render the base template
        base_template = self._load_template("base.html")
        html_content = base_template.render(
            title=self.title,
            content=section_html,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Write to file
        output_file = Path(output_path)
        output_file.write_text(html_content, encoding="utf-8")

        return str(output_file.absolute())

    def to_html(self) -> str:
        """
        Generate the HTML tear sheet and return as a string.

        Returns
        -------
        str
            The complete HTML document as a string
        """
        # Build all sections
        section_html = "\n".join(self._build_sections())

        # Render the base template
        base_template = self._load_template("base.html")
        return base_template.render(
            title=self.title,
            content=section_html,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
