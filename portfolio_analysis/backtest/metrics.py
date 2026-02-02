"""
Backtest-specific performance metrics.

This module provides additional metrics calculations for backtest results
beyond those available in the core PerformanceMetrics class.
"""

from typing import Optional

import numpy as np
import pandas as pd

from portfolio_analysis.constants import TRADING_DAYS_PER_YEAR


class BacktestMetrics:
    """
    Calculate backtest-specific performance metrics.

    These metrics are designed for evaluating strategy backtests
    and include trade-related and timing-related measures.

    Parameters
    ----------
    portfolio_value : pd.Series
        Daily portfolio value
    trades : list
        List of trade records from backtest

    Examples
    --------
    >>> from portfolio_analysis.backtest import BacktestEngine, RebalanceStrategy
    >>> engine = BacktestEngine(data, strategy)
    >>> result = engine.run()
    >>> metrics = BacktestMetrics(result.portfolio_value, result.trades)
    >>> print(metrics.get_all_metrics())
    """

    def __init__(
        self,
        portfolio_value: pd.Series,
        trades: Optional[list] = None,
    ):
        self.portfolio_value = portfolio_value
        self.trades = trades or []
        self.daily_returns = portfolio_value.pct_change().fillna(0)

    def calculate_rolling_sharpe(
        self,
        window: int = TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = 0.02,
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Parameters
        ----------
        window : int, default 252
            Rolling window in trading days
        risk_free_rate : float, default 0.02
            Annual risk-free rate

        Returns
        -------
        pd.Series
            Rolling Sharpe ratio
        """
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        excess_returns = self.daily_returns - daily_rf

        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = self.daily_returns.rolling(window=window).std()

        rolling_sharpe = (
            rolling_mean
            * np.sqrt(TRADING_DAYS_PER_YEAR)
            / (rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        )

        return rolling_sharpe

    def calculate_rolling_volatility(
        self,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Parameters
        ----------
        window : int, default 21
            Rolling window in trading days
        annualize : bool, default True
            Whether to annualize the volatility

        Returns
        -------
        pd.Series
            Rolling volatility
        """
        rolling_vol = self.daily_returns.rolling(window=window).std()

        if annualize:
            rolling_vol = rolling_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        return rolling_vol

    def calculate_drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series over time.

        Returns
        -------
        pd.Series
            Drawdown at each point in time
        """
        cumulative = (1 + self.daily_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown

    def calculate_underwater_periods(self) -> list[dict]:
        """
        Calculate periods where portfolio was underwater (in drawdown).

        Returns
        -------
        list of dict
            List of underwater periods with start, end, depth, and duration
        """
        drawdown = self.calculate_drawdown_series()
        periods = []
        in_drawdown = False
        start_date = None
        max_depth = 0

        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
                max_depth = dd
            elif dd < 0 and in_drawdown:
                max_depth = min(max_depth, dd)
            elif dd == 0 and in_drawdown:
                periods.append(
                    {
                        "start": start_date,
                        "end": date,
                        "max_drawdown": max_depth,
                        "duration_days": (date - start_date).days,
                    }
                )
                in_drawdown = False
                max_depth = 0

        # Handle case where we end in drawdown
        if in_drawdown:
            periods.append(
                {
                    "start": start_date,
                    "end": drawdown.index[-1],
                    "max_drawdown": max_depth,
                    "duration_days": (drawdown.index[-1] - start_date).days,
                    "recovery": False,
                }
            )

        return periods

    def calculate_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns.

        Returns
        -------
        pd.Series
            Monthly returns
        """
        monthly = self.portfolio_value.resample("ME").last()
        return monthly.pct_change().dropna()

    def calculate_annual_returns(self) -> pd.Series:
        """
        Calculate annual returns.

        Returns
        -------
        pd.Series
            Annual returns
        """
        annual = self.portfolio_value.resample("YE").last()
        return annual.pct_change().dropna()

    def calculate_trade_statistics(self) -> dict:
        """
        Calculate trade-related statistics.

        Returns
        -------
        dict
            Trade statistics including count, frequency, and costs
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "unique_dates": 0,
                "avg_trades_per_rebalance": 0,
                "total_costs": 0,
                "avg_trade_size": 0,
            }

        trade_dates = set(t["date"] for t in self.trades)
        total_costs = sum(t["cost"] for t in self.trades)
        total_value = sum(abs(t["trade_value"]) for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "unique_dates": len(trade_dates),
            "avg_trades_per_rebalance": (
                len(self.trades) / len(trade_dates) if trade_dates else 0
            ),
            "total_costs": total_costs,
            "avg_trade_size": total_value / len(self.trades) if self.trades else 0,
        }

    def calculate_timing_statistics(self) -> dict:
        """
        Calculate market timing statistics.

        Returns
        -------
        dict
            Statistics about strategy timing effectiveness
        """
        # Best and worst periods
        monthly = self.calculate_monthly_returns()

        best_month = monthly.max() if len(monthly) > 0 else 0
        worst_month = monthly.min() if len(monthly) > 0 else 0
        positive_months = (monthly > 0).sum() if len(monthly) > 0 else 0
        total_months = len(monthly)

        # Daily statistics
        best_day = self.daily_returns.max()
        worst_day = self.daily_returns.min()
        positive_days = (self.daily_returns > 0).sum()
        total_days = len(self.daily_returns[self.daily_returns != 0])

        return {
            "best_month": best_month,
            "worst_month": worst_month,
            "positive_months": positive_months,
            "total_months": total_months,
            "monthly_win_rate": (
                positive_months / total_months if total_months > 0 else 0
            ),
            "best_day": best_day,
            "worst_day": worst_day,
            "positive_days": positive_days,
            "total_days": total_days,
            "daily_win_rate": positive_days / total_days if total_days > 0 else 0,
        }

    def calculate_tail_risk(self, confidence: float = 0.95) -> dict:
        """
        Calculate tail risk metrics.

        Parameters
        ----------
        confidence : float, default 0.95
            Confidence level for VaR and CVaR

        Returns
        -------
        dict
            Tail risk metrics including VaR and CVaR
        """
        returns = self.daily_returns.dropna()

        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()

        # Annualize
        var_annual = var * np.sqrt(TRADING_DAYS_PER_YEAR)
        cvar_annual = cvar * np.sqrt(TRADING_DAYS_PER_YEAR)

        return {
            "daily_var": var,
            "daily_cvar": cvar,
            "annual_var": var_annual,
            "annual_cvar": cvar_annual,
            "confidence_level": confidence,
        }

    def get_all_metrics(self) -> dict:
        """
        Get all available metrics.

        Returns
        -------
        dict
            Comprehensive metrics dictionary
        """
        metrics = {}

        # Basic performance
        total_return = self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1
        years = (
            self.portfolio_value.index[-1] - self.portfolio_value.index[0]
        ).days / 365.25

        metrics["total_return"] = total_return
        metrics["cagr"] = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        metrics["annual_volatility"] = self.daily_returns.std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        # Drawdown
        drawdown = self.calculate_drawdown_series()
        metrics["max_drawdown"] = drawdown.min()
        metrics["avg_drawdown"] = (
            drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        )

        # Add trade statistics
        metrics.update(self.calculate_trade_statistics())

        # Add timing statistics
        metrics.update(self.calculate_timing_statistics())

        # Add tail risk
        tail_risk = self.calculate_tail_risk()
        metrics["var_95"] = tail_risk["annual_var"]
        metrics["cvar_95"] = tail_risk["annual_cvar"]

        return metrics
