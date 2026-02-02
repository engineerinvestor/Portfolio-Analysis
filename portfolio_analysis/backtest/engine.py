"""
Backtesting engine for portfolio strategies.

This module provides the core BacktestEngine class that runs strategy
backtests with realistic transaction costs and generates comprehensive results.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portfolio_analysis.backtest.strategy import Strategy
from portfolio_analysis.constants import TRADING_DAYS_PER_YEAR


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes
    ----------
    portfolio_value : pd.Series
        Daily portfolio value over the backtest period
    daily_returns : pd.Series
        Daily returns of the portfolio
    weights_history : pd.DataFrame
        Portfolio weights at each date
    trades : list
        List of trade records
    metrics : dict
        Performance metrics summary
    strategy_name : str
        Name of the strategy used
    """

    portfolio_value: pd.Series
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    strategy_name: str = ""

    def __repr__(self) -> str:
        start = self.portfolio_value.index[0].strftime("%Y-%m-%d")
        end = self.portfolio_value.index[-1].strftime("%Y-%m-%d")
        return (
            f"BacktestResult(strategy='{self.strategy_name}', "
            f"period='{start}' to '{end}', "
            f"total_return={self.metrics.get('total_return', 0):.2%})"
        )


class BacktestEngine:
    """
    Engine for running portfolio strategy backtests.

    Simulates portfolio performance over historical data with realistic
    transaction costs, slippage, and rebalancing rules.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data with datetime index
    strategy : Strategy
        Strategy instance to backtest
    initial_capital : float, default 100000
        Starting portfolio value
    transaction_cost : float, default 0.001
        Transaction cost as a fraction (0.001 = 0.1% = 10 bps)
    slippage : float, default 0.0005
        Slippage as a fraction (0.0005 = 0.05% = 5 bps)

    Examples
    --------
    >>> from portfolio_analysis.backtest import BacktestEngine, RebalanceStrategy
    >>> strategy = RebalanceStrategy({'SPY': 0.6, 'AGG': 0.4}, rebalance_frequency='M')
    >>> engine = BacktestEngine(data, strategy, initial_capital=100000)
    >>> result = engine.run()
    >>> print(result.metrics)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Validate that all strategy tickers are in data
        missing_tickers = set(strategy.initial_weights.keys()) - set(data.columns)
        if missing_tickers:
            raise ValueError(f"Strategy tickers not found in data: {missing_tickers}")

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns
        -------
        BacktestResult
            Complete backtest results including portfolio value,
            returns, weights history, and performance metrics.
        """
        dates = self.data.index
        tickers = list(self.strategy.initial_weights.keys())

        # Initialize tracking
        portfolio_values = []
        weights_history = []
        trades = []
        current_weights = {ticker: 0.0 for ticker in tickers}
        current_value = self.initial_capital

        # Calculate daily returns for each asset
        returns = self.data[tickers].pct_change().fillna(0)

        for i, date in enumerate(dates):
            # Get target weights from strategy
            available_data = self.data.loc[:date]
            target_weights = self.strategy.generate_weights(
                available_data, current_weights, date
            )

            # Check if we should rebalance
            should_rebalance = self.strategy.should_rebalance(
                available_data, current_weights, target_weights, date
            )

            if should_rebalance:
                # Calculate transaction costs
                trade_cost = self._calculate_trade_cost(
                    current_weights, target_weights, current_value
                )
                current_value -= trade_cost

                # Record trades
                for ticker in tickers:
                    old_w = current_weights.get(ticker, 0)
                    new_w = target_weights.get(ticker, 0)
                    if abs(new_w - old_w) > 1e-6:
                        trades.append(
                            {
                                "date": date,
                                "ticker": ticker,
                                "old_weight": old_w,
                                "new_weight": new_w,
                                "trade_value": (new_w - old_w) * current_value,
                                "cost": trade_cost,
                            }
                        )

                current_weights = target_weights.copy()

            # Apply daily returns
            if i > 0:
                daily_return = sum(
                    current_weights.get(ticker, 0) * returns.loc[date, ticker]
                    for ticker in tickers
                )
                current_value *= 1 + daily_return

                # Update weights based on price changes (weights drift)
                total_value = sum(
                    current_weights.get(ticker, 0) * (1 + returns.loc[date, ticker])
                    for ticker in tickers
                )
                if total_value > 0:
                    current_weights = {
                        ticker: current_weights.get(ticker, 0)
                        * (1 + returns.loc[date, ticker])
                        / total_value
                        for ticker in tickers
                    }

            portfolio_values.append(current_value)
            weights_history.append(current_weights.copy())

        # Create result series and DataFrame
        portfolio_series = pd.Series(
            portfolio_values, index=dates, name="portfolio_value"
        )
        daily_returns = portfolio_series.pct_change().fillna(0)
        weights_df = pd.DataFrame(weights_history, index=dates)

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_series, daily_returns, trades)

        return BacktestResult(
            portfolio_value=portfolio_series,
            daily_returns=daily_returns,
            weights_history=weights_df,
            trades=trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
        )

    def _calculate_trade_cost(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> float:
        """Calculate total transaction cost for rebalancing."""
        turnover = (
            sum(
                abs(target_weights.get(ticker, 0) - current_weights.get(ticker, 0))
                for ticker in set(current_weights) | set(target_weights)
            )
            / 2
        )  # Divide by 2 because we count each trade once

        trade_value = turnover * portfolio_value
        cost = trade_value * (self.transaction_cost + self.slippage)

        return cost

    def _calculate_metrics(
        self,
        portfolio_value: pd.Series,
        daily_returns: pd.Series,
        trades: list,
    ) -> dict:
        """Calculate comprehensive performance metrics."""
        # Basic returns
        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe = (
            (cagr - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        )

        # Sortino ratio
        downside_returns = daily_returns.where(daily_returns < 0, 0)
        downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = (
            (cagr - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        total_trades = len(trades)
        total_costs = sum(t["cost"] for t in trades) if trades else 0
        turnover = sum(abs(t["trade_value"]) for t in trades) if trades else 0

        # Win rate
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns[daily_returns != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "total_trades": total_trades,
            "total_costs": total_costs,
            "total_turnover": turnover,
            "win_rate": win_rate,
            "best_day": daily_returns.max(),
            "worst_day": daily_returns.min(),
            "years": years,
        }

    def compare_strategies(
        self,
        strategies: list[Strategy],
        include_baseline: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Parameters
        ----------
        strategies : list of Strategy
            Strategies to compare
        include_baseline : bool, default True
            Include buy-and-hold as baseline

        Returns
        -------
        pd.DataFrame
            Comparison of strategy metrics
        """
        from portfolio_analysis.backtest.strategy import BuyAndHoldStrategy

        results = []

        if include_baseline:
            baseline_strategy = BuyAndHoldStrategy(
                self.strategy.initial_weights, name="Buy & Hold Baseline"
            )
            strategies = [baseline_strategy] + list(strategies)

        for strategy in strategies:
            engine = BacktestEngine(
                self.data,
                strategy,
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
            )
            result = engine.run()
            metrics = result.metrics.copy()
            metrics["strategy"] = strategy.name
            results.append(metrics)

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index("strategy")

        return comparison_df
