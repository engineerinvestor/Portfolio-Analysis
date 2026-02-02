"""
Benchmark comparison functionality.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Optional, Union


class BenchmarkComparison:
    """
    Compare portfolio performance against market benchmarks.

    Calculates alpha, beta, tracking error, information ratio, and
    generates comparison reports and visualizations.

    Parameters
    ----------
    portfolio_data : pd.DataFrame
        Historical price data for portfolio assets
    weights : array-like
        Portfolio weights (must sum to 1.0)
    benchmark_ticker : str, default 'SPY'
        Ticker symbol for benchmark
    risk_free_rate : float, default 0.02
        Annual risk-free rate for calculations

    Examples
    --------
    >>> comparison = BenchmarkComparison(data, weights, benchmark_ticker='SPY')
    >>> comparison.generate_report()
    >>> comparison.plot_cumulative_returns()
    """

    BENCHMARKS = {
        "SPY": "S&P 500 (US Large Cap)",
        "VTI": "Total US Stock Market",
        "BND": "Total US Bond Market",
        "VT": "Total World Stock Market",
        "QQQ": "NASDAQ 100",
        "IWM": "Russell 2000 (US Small Cap)",
        "EFA": "Developed Markets ex-US",
        "AGG": "US Aggregate Bond",
    }

    def __init__(
        self,
        portfolio_data: pd.DataFrame,
        weights: List[float],
        benchmark_ticker: str = "SPY",
        risk_free_rate: float = 0.02,
    ):
        self.portfolio_data = portfolio_data
        self.weights = np.array(weights)
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate

        # Calculate portfolio returns
        returns = portfolio_data.pct_change().dropna()
        self.portfolio_returns = returns.dot(self.weights)

        # Fetch benchmark data
        self.benchmark_data = self._fetch_benchmark()
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()

        # Align dates
        common_dates = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )
        self.portfolio_returns = self.portfolio_returns.loc[common_dates]
        self.benchmark_returns = self.benchmark_returns.loc[common_dates]

    def _fetch_benchmark(self) -> pd.Series:
        """Fetch benchmark price data."""
        start_date = self.portfolio_data.index.min()
        end_date = self.portfolio_data.index.max()

        raw_data = yf.download(
            self.benchmark_ticker, start=start_date, end=end_date, progress=False
        )

        # Handle yfinance column format changes across versions
        # Check for MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(raw_data.columns, pd.MultiIndex):
            price_types = raw_data.columns.get_level_values(0).unique()
            if "Adj Close" in price_types:
                benchmark = raw_data["Adj Close"]
            elif "Close" in price_types:
                benchmark = raw_data["Close"]
            else:
                raise ValueError(
                    f"No Close or Adj Close column found. Available: {price_types.tolist()}"
                )
        else:
            if "Adj Close" in raw_data.columns:
                benchmark = raw_data["Adj Close"]
            elif "Close" in raw_data.columns:
                benchmark = raw_data["Close"]
            else:
                raise ValueError(
                    f"No Close or Adj Close column found. Available: {raw_data.columns.tolist()}"
                )

        # Ensure we return a Series, not a DataFrame
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.squeeze()

        return benchmark

    def calculate_beta(self) -> float:
        """Calculate portfolio beta relative to benchmark."""
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        return covariance / benchmark_variance

    def calculate_alpha(self, annualized: bool = True) -> float:
        """Calculate Jensen's alpha (CAPM alpha)."""
        beta = self.calculate_beta()

        portfolio_mean = self.portfolio_returns.mean()
        benchmark_mean = self.benchmark_returns.mean()
        rf_daily = self.risk_free_rate / 252

        alpha = portfolio_mean - (rf_daily + beta * (benchmark_mean - rf_daily))

        if annualized:
            alpha = alpha * 252

        return alpha

    def calculate_tracking_error(self, annualized: bool = True) -> float:
        """Calculate tracking error (active risk)."""
        active_returns = self.portfolio_returns - self.benchmark_returns
        tracking_error = active_returns.std()

        if annualized:
            tracking_error = tracking_error * np.sqrt(252)

        return tracking_error

    def calculate_information_ratio(self) -> float:
        """Calculate information ratio."""
        active_return = (
            self.portfolio_returns.mean() - self.benchmark_returns.mean()
        ) * 252
        tracking_error = self.calculate_tracking_error(annualized=True)

        if tracking_error == 0:
            return np.inf if active_return > 0 else -np.inf

        return active_return / tracking_error

    def calculate_correlation(self) -> float:
        """Calculate correlation with benchmark."""
        return np.corrcoef(self.portfolio_returns, self.benchmark_returns)[0, 1]

    def calculate_r_squared(self) -> float:
        """Calculate R-squared."""
        return self.calculate_correlation() ** 2

    def calculate_up_capture(self) -> float:
        """Calculate upside capture ratio."""
        up_mask = self.benchmark_returns > 0
        if up_mask.sum() == 0:
            return np.nan

        portfolio_up = self.portfolio_returns[up_mask].mean()
        benchmark_up = self.benchmark_returns[up_mask].mean()

        return (portfolio_up / benchmark_up) * 100

    def calculate_down_capture(self) -> float:
        """Calculate downside capture ratio."""
        down_mask = self.benchmark_returns < 0
        if down_mask.sum() == 0:
            return np.nan

        portfolio_down = self.portfolio_returns[down_mask].mean()
        benchmark_down = self.benchmark_returns[down_mask].mean()

        return (portfolio_down / benchmark_down) * 100

    def get_metrics(self) -> dict:
        """Get all benchmark comparison metrics as a dictionary."""
        return {
            "beta": self.calculate_beta(),
            "alpha": self.calculate_alpha(),
            "tracking_error": self.calculate_tracking_error(),
            "information_ratio": self.calculate_information_ratio(),
            "correlation": self.calculate_correlation(),
            "r_squared": self.calculate_r_squared(),
            "up_capture": self.calculate_up_capture(),
            "down_capture": self.calculate_down_capture(),
            "portfolio_return": self.portfolio_returns.mean() * 252,
            "benchmark_return": self.benchmark_returns.mean() * 252,
            "portfolio_volatility": self.portfolio_returns.std() * np.sqrt(252),
            "benchmark_volatility": self.benchmark_returns.std() * np.sqrt(252),
        }

    def generate_report(self) -> None:
        """Print a comprehensive comparison report."""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON REPORT")
        print("=" * 60)
        print(f"Benchmark: {self.benchmark_ticker}", end="")
        if self.benchmark_ticker in self.BENCHMARKS:
            print(f" ({self.BENCHMARKS[self.benchmark_ticker]})")
        else:
            print()
        print(
            f"Period: {self.portfolio_returns.index.min().strftime('%Y-%m-%d')} to "
            f"{self.portfolio_returns.index.max().strftime('%Y-%m-%d')}"
        )
        print("-" * 60)

        print("\nAnnualized Returns:")
        print(f"  Portfolio:  {metrics['portfolio_return']*100:.2f}%")
        print(f"  Benchmark:  {metrics['benchmark_return']*100:.2f}%")
        print(
            f"  Difference: {(metrics['portfolio_return']-metrics['benchmark_return'])*100:+.2f}%"
        )

        print("\nAnnualized Volatility:")
        print(f"  Portfolio:  {metrics['portfolio_volatility']*100:.2f}%")
        print(f"  Benchmark:  {metrics['benchmark_volatility']*100:.2f}%")

        print("\nRisk Metrics:")
        print(f"  Beta:              {metrics['beta']:.3f}")
        print(f"  Alpha (annual):    {metrics['alpha']*100:.2f}%")
        print(f"  R-squared:         {metrics['r_squared']:.3f}")
        print(f"  Correlation:       {metrics['correlation']:.3f}")

        print("\nPerformance Metrics:")
        print(f"  Tracking Error:    {metrics['tracking_error']*100:.2f}%")
        print(f"  Information Ratio: {metrics['information_ratio']:.3f}")
        print(f"  Up Capture:        {metrics['up_capture']:.1f}%")
        print(f"  Down Capture:      {metrics['down_capture']:.1f}%")
        print("=" * 60)

    def plot_cumulative_returns(self, initial_value: float = 10000) -> None:
        """Plot cumulative returns comparison."""
        portfolio_cum = (1 + self.portfolio_returns).cumprod() * initial_value
        benchmark_cum = (1 + self.benchmark_returns).cumprod() * initial_value

        plt.figure(figsize=(12, 6))

        plt.plot(
            portfolio_cum.index,
            portfolio_cum.values,
            label="Portfolio",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            benchmark_cum.index,
            benchmark_cum.values,
            label=f"Benchmark ({self.benchmark_ticker})",
            linewidth=2,
            color="orange",
        )

        plt.title("Cumulative Returns: Portfolio vs Benchmark")
        plt.xlabel("Date")
        plt.ylabel(f"Value (starting from ${initial_value:,.0f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rolling_metrics(self, window: int = 252) -> None:
        """Plot rolling alpha and beta."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        rolling_beta = []
        rolling_alpha = []
        dates = []

        for i in range(window, len(self.portfolio_returns)):
            port_window = self.portfolio_returns.iloc[i - window : i]
            bench_window = self.benchmark_returns.iloc[i - window : i]

            cov = np.cov(port_window, bench_window)[0, 1]
            var = np.var(bench_window)
            beta = cov / var

            rf_daily = self.risk_free_rate / 252
            alpha = (
                port_window.mean()
                - (rf_daily + beta * (bench_window.mean() - rf_daily))
            ) * 252

            rolling_beta.append(beta)
            rolling_alpha.append(alpha)
            dates.append(self.portfolio_returns.index[i])

        axes[0].plot(dates, rolling_beta, linewidth=1.5, color="blue")
        axes[0].axhline(
            y=1.0, color="red", linestyle="--", linewidth=1, label="Beta = 1.0"
        )
        axes[0].set_ylabel("Beta")
        axes[0].set_title(f"Rolling {window}-Day Beta and Alpha")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            dates, [a * 100 for a in rolling_alpha], linewidth=1.5, color="green"
        )
        axes[1].axhline(
            y=0, color="red", linestyle="--", linewidth=1, label="Alpha = 0"
        )
        axes[1].set_ylabel("Alpha (%)")
        axes[1].set_xlabel("Date")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
