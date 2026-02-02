"""
Interactive widget-based portfolio analyzer for Jupyter/Colab.
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from portfolio_analysis.analysis.montecarlo import MonteCarloSimulation
from portfolio_analysis.analysis.portfolio import PortfolioAnalysis
from portfolio_analysis.data.loader import DataLoader

# Preset portfolios for quick selection
PRESET_PORTFOLIOS = {
    "Custom": {},
    "60/40 Traditional": {"VTI": 0.60, "BND": 0.40},
    "Three-Fund Portfolio": {"VTI": 0.40, "VXUS": 0.20, "BND": 0.40},
    "All-Weather (Ray Dalio)": {
        "VTI": 0.30,
        "TLT": 0.40,
        "IEF": 0.15,
        "GLD": 0.075,
        "DBC": 0.075,
    },
    "Golden Butterfly": {
        "VTI": 0.20,
        "VBR": 0.20,
        "TLT": 0.20,
        "SHY": 0.20,
        "GLD": 0.20,
    },
    "Aggressive Growth": {"VTI": 0.50, "VGT": 0.25, "VXUS": 0.25},
    "Conservative Income": {"VTI": 0.20, "BND": 0.50, "VTIP": 0.15, "VNQ": 0.15},
    "S&P 500 Only": {"SPY": 1.0},
}


class InteractivePortfolioAnalyzer:
    """
    Interactive widget-based portfolio analyzer for Jupyter/Colab.

    Provides a user-friendly interface for:
    - Selecting preset or custom portfolios
    - Setting date ranges
    - Running performance analysis
    - Monte Carlo simulations

    Examples
    --------
    >>> analyzer = InteractivePortfolioAnalyzer()
    >>> analyzer.display()
    """

    def __init__(self):
        if not HAS_WIDGETS:
            raise ImportError(
                "ipywidgets is required for interactive analysis. "
                "Install with: pip install ipywidgets"
            )

        self.data = None
        self.tickers = []
        self.weights = []
        self._setup_widgets()

    def _setup_widgets(self):
        """Set up all interactive widgets."""
        # Portfolio Selection
        self.preset_dropdown = widgets.Dropdown(
            options=list(PRESET_PORTFOLIOS.keys()),
            value="Three-Fund Portfolio",
            description="Preset:",
            style={"description_width": "80px"},
        )
        self.preset_dropdown.observe(self._on_preset_change, names="value")

        # Custom portfolio inputs
        self.custom_tickers = widgets.Text(
            value="VTI, VXUS, BND",
            description="Tickers:",
            placeholder="e.g., VTI, VXUS, BND",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="400px"),
        )

        self.custom_weights = widgets.Text(
            value="0.4, 0.2, 0.4",
            description="Weights:",
            placeholder="e.g., 0.4, 0.2, 0.4",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="400px"),
        )

        # Date range
        self.start_date = widgets.DatePicker(
            description="Start:",
            value=datetime.now() - timedelta(days=5 * 365),
            style={"description_width": "80px"},
        )

        self.end_date = widgets.DatePicker(
            description="End:",
            value=datetime.now(),
            style={"description_width": "80px"},
        )

        # Analysis options
        self.show_performance = widgets.Checkbox(
            value=True,
            description="Performance Metrics",
            layout=widgets.Layout(width="200px"),
        )

        self.show_cumulative = widgets.Checkbox(
            value=True,
            description="Cumulative Returns",
            layout=widgets.Layout(width="200px"),
        )

        self.show_monte_carlo = widgets.Checkbox(
            value=True,
            description="Monte Carlo Simulation",
            layout=widgets.Layout(width="200px"),
        )

        # Monte Carlo parameters
        self.mc_simulations = widgets.IntSlider(
            value=1000,
            min=100,
            max=5000,
            step=100,
            description="Simulations:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="400px"),
        )

        self.mc_horizon = widgets.IntSlider(
            value=252,
            min=21,
            max=1260,
            step=21,
            description="Days Forward:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="400px"),
        )

        self.mc_initial = widgets.IntText(
            value=10000,
            description="Initial ($):",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="200px"),
        )

        # Risk-free rate
        self.risk_free_rate = widgets.FloatSlider(
            value=0.04,
            min=0.0,
            max=0.10,
            step=0.005,
            description="Risk-Free Rate:",
            readout_format=".1%",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="400px"),
        )

        # Analyze button
        self.analyze_button = widgets.Button(
            description="Analyze Portfolio",
            button_style="primary",
            icon="chart-line",
            layout=widgets.Layout(width="200px", height="40px"),
        )
        self.analyze_button.on_click(self._on_analyze)

        # Output area
        self.output = widgets.Output()

        # Initialize with preset
        self._on_preset_change({"new": "Three-Fund Portfolio"})

    def _on_preset_change(self, change):
        """Handle preset portfolio selection."""
        preset_name = change["new"]
        if preset_name != "Custom" and preset_name in PRESET_PORTFOLIOS:
            portfolio = PRESET_PORTFOLIOS[preset_name]
            self.custom_tickers.value = ", ".join(portfolio.keys())
            self.custom_weights.value = ", ".join([str(w) for w in portfolio.values()])

    def _parse_portfolio(self):
        """Parse ticker and weight inputs."""
        tickers = [t.strip().upper() for t in self.custom_tickers.value.split(",")]
        weights = [float(w.strip()) for w in self.custom_weights.value.split(",")]

        if len(tickers) != len(weights):
            raise ValueError(
                f"Number of tickers ({len(tickers)}) must match weights ({len(weights)})"
            )

        weight_sum = sum(weights)
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0 (currently {weight_sum:.2f})")

        return tickers, weights

    def _on_analyze(self, button):
        """Handle analyze button click."""
        with self.output:
            clear_output(wait=True)

            try:
                self.tickers, self.weights = self._parse_portfolio()

                print(f"Analyzing portfolio: {dict(zip(self.tickers, self.weights))}")
                print(f"Date range: {self.start_date.value} to {self.end_date.value}")
                print("\nFetching data...")

                loader = DataLoader(
                    self.tickers,
                    self.start_date.value.strftime("%Y-%m-%d"),
                    self.end_date.value.strftime("%Y-%m-%d"),
                )
                self.data = loader.fetch_data(progress=False)

                if self.data.empty:
                    print(
                        "Error: No data returned. Check ticker symbols and date range."
                    )
                    return

                if isinstance(self.data, pd.Series):
                    self.data = self.data.to_frame(name=self.tickers[0])

                print(f"Data loaded: {len(self.data)} trading days\n")

                if self.show_performance.value:
                    self._display_metrics()

                num_plots = sum(
                    [self.show_cumulative.value, self.show_monte_carlo.value]
                )
                if num_plots > 0:
                    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
                    if num_plots == 1:
                        axes = [axes]

                    plot_idx = 0

                    if self.show_cumulative.value:
                        self._plot_cumulative(axes[plot_idx])
                        plot_idx += 1

                    if self.show_monte_carlo.value:
                        self._plot_monte_carlo(axes[plot_idx])
                        plot_idx += 1

                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback

                traceback.print_exc()

    def _display_metrics(self):
        """Display portfolio metrics."""
        portfolio = PortfolioAnalysis(self.data, self.weights)

        port_return = portfolio.calculate_portfolio_return()
        port_vol = portfolio.calculate_portfolio_volatility()
        port_sharpe = portfolio.calculate_portfolio_sharpe_ratio(
            self.risk_free_rate.value
        )
        max_dd = portfolio.calculate_max_drawdown()

        print("=" * 50)
        print("PORTFOLIO PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Annual Return:      {port_return*100:>10.2f}%")
        print(f"Annual Volatility:  {port_vol*100:>10.2f}%")
        print(f"Sharpe Ratio:       {port_sharpe:>10.2f}")
        print(f"Max Drawdown:       {max_dd*100:>10.2f}%")
        print(f"Risk-Free Rate:     {self.risk_free_rate.value*100:>10.2f}%")
        print("=" * 50 + "\n")

    def _plot_cumulative(self, ax):
        """Plot cumulative returns."""
        returns = self.data.pct_change().dropna()
        weighted_returns = returns.dot(self.weights)
        cumulative = (1 + weighted_returns).cumprod()

        ax.plot(cumulative.index, cumulative.values, linewidth=2, color="blue")
        ax.set_title("Portfolio Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)

        final_val = cumulative.iloc[-1]
        ax.annotate(
            f"${final_val:.2f}",
            xy=(cumulative.index[-1], final_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color="blue",
        )

    def _plot_monte_carlo(self, ax):
        """Plot Monte Carlo simulation."""
        mc = MonteCarloSimulation(
            self.data,
            self.weights,
            num_simulations=self.mc_simulations.value,
            time_horizon=self.mc_horizon.value,
            initial_investment=self.mc_initial.value,
        )
        mc.plot_simulation(ax=ax)

        stats = mc.get_statistics()
        print("MONTE CARLO PROJECTION")
        print("-" * 40)
        print(f"Initial Investment:  ${self.mc_initial.value:,.0f}")
        print(f"Time Horizon:        {self.mc_horizon.value} days")
        print(f"Median Final Value:  ${stats['final_values']['median']:,.0f}")
        print(f"5th Percentile:      ${stats['final_values']['percentile_5']:,.0f}")
        print(f"95th Percentile:     ${stats['final_values']['percentile_95']:,.0f}")
        print(f"Probability of Loss: {stats['final_values']['prob_loss']:.1f}%")
        print("-" * 40 + "\n")

    def display(self):
        """Display the interactive widget interface."""
        portfolio_box = widgets.VBox(
            [
                widgets.HTML("<h3>Portfolio Selection</h3>"),
                self.preset_dropdown,
                self.custom_tickers,
                self.custom_weights,
            ]
        )

        date_box = widgets.VBox(
            [
                widgets.HTML("<h3>Date Range</h3>"),
                widgets.HBox([self.start_date, self.end_date]),
            ]
        )

        options_box = widgets.VBox(
            [
                widgets.HTML("<h3>Analysis Options</h3>"),
                widgets.HBox(
                    [self.show_performance, self.show_cumulative, self.show_monte_carlo]
                ),
                self.risk_free_rate,
            ]
        )

        mc_box = widgets.VBox(
            [
                widgets.HTML("<h3>Monte Carlo Parameters</h3>"),
                self.mc_simulations,
                self.mc_horizon,
                self.mc_initial,
            ]
        )

        left_panel = widgets.VBox([portfolio_box, date_box])
        right_panel = widgets.VBox([options_box, mc_box])

        top_panel = widgets.HBox(
            [left_panel, right_panel],
            layout=widgets.Layout(justify_content="space-around"),
        )

        display(
            widgets.VBox(
                [
                    widgets.HTML("<h2>Interactive Portfolio Analyzer</h2>"),
                    top_panel,
                    widgets.HBox(
                        [self.analyze_button],
                        layout=widgets.Layout(
                            justify_content="center", margin="20px 0"
                        ),
                    ),
                    widgets.HTML("<hr>"),
                    self.output,
                ]
            )
        )
