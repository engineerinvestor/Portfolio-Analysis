"""
Engineer Investor Portfolio Analyzer - Streamlit Web Application

A professional portfolio analysis tool built with transparency and user value in mind.

Author: Engineer Investor (@egr_investor)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Page configuration
st.set_page_config(
    page_title="Engineer Investor Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TRADING_DAYS = 252

PRESET_PORTFOLIOS = {
    'Custom': {},
    '60/40 Traditional': {'VTI': 0.60, 'BND': 0.40},
    'Three-Fund Portfolio': {'VTI': 0.40, 'VXUS': 0.20, 'BND': 0.40},
    'All-Weather (Ray Dalio)': {'VTI': 0.30, 'TLT': 0.40, 'IEF': 0.15, 'GLD': 0.075, 'DBC': 0.075},
    'Golden Butterfly': {'VTI': 0.20, 'VBR': 0.20, 'TLT': 0.20, 'SHY': 0.20, 'GLD': 0.20},
    'Aggressive Growth': {'VTI': 0.50, 'VGT': 0.25, 'VXUS': 0.25},
    'Conservative Income': {'VTI': 0.20, 'BND': 0.50, 'VTIP': 0.15, 'VNQ': 0.15},
    'S&P 500 Only': {'SPY': 1.0},
    'Total World Stock': {'VT': 1.0},
}

BENCHMARKS = {
    'SPY': 'S&P 500',
    'VTI': 'Total US Market',
    'VT': 'Total World',
    'BND': 'US Bonds',
    'QQQ': 'NASDAQ 100',
}


# ============================================
# Data Functions
# ============================================

@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    """Fetch adjusted price data from Yahoo Finance with caching."""
    try:
        # Use Ticker.history() API - returns adjusted prices by default
        # (auto_adjust=True means Close is already adjusted for dividends & splits)
        data_frames = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                # auto_adjust=True (default) returns dividend/split-adjusted prices
                hist = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns:
                    # Remove timezone info from index for consistency
                    hist.index = hist.index.tz_localize(None)
                    data_frames[ticker] = hist['Close']  # This is Adjusted Close
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)

        if failed_tickers:
            st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")

        if not data_frames:
            return None

        data = pd.DataFrame(data_frames)
        data = data.dropna()

        return data if not data.empty else None

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# ============================================
# Analysis Functions
# ============================================

def calculate_portfolio_metrics(data, weights, risk_free_rate=0.02):
    """Calculate portfolio performance metrics."""
    returns = data.pct_change().dropna()
    weighted_returns = returns.dot(weights)

    # Calculate metrics
    annual_return = weighted_returns.mean() * TRADING_DAYS
    annual_vol = weighted_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_vol

    # Sortino ratio
    downside_returns = weighted_returns[weighted_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    # Max drawdown
    cumulative = (1 + weighted_returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    max_dd = drawdown.min()

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'cumulative_returns': cumulative,
        'weighted_returns': weighted_returns,
    }


def run_monte_carlo(data, weights, num_simulations=1000, time_horizon=252, initial_investment=10000):
    """Run Monte Carlo simulation."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    results = np.zeros((num_simulations, time_horizon))

    for i in range(num_simulations):
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = sim_returns @ weights
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        results[i, :] = initial_investment * cumulative_returns

    final_values = results[:, -1]

    return {
        'results': results,
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'p5': np.percentile(final_values, 5),
        'p95': np.percentile(final_values, 95),
        'prob_loss': np.mean(final_values < initial_investment) * 100,
        'percentiles': {
            5: np.percentile(results, 5, axis=0),
            50: np.percentile(results, 50, axis=0),
            95: np.percentile(results, 95, axis=0),
        }
    }


def optimize_portfolio(data, strategy='max_sharpe', risk_free_rate=0.02):
    """Optimize portfolio weights."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    n_assets = len(data.columns)

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -(ret - risk_free_rate) / vol

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)

    if strategy == 'max_sharpe':
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == 'min_volatility':
        result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        return None

    optimal_weights = result.x
    return {
        'weights': dict(zip(data.columns, optimal_weights)),
        'return': portfolio_return(optimal_weights),
        'volatility': portfolio_volatility(optimal_weights),
        'sharpe': (portfolio_return(optimal_weights) - risk_free_rate) / portfolio_volatility(optimal_weights)
    }


def calculate_benchmark_comparison(data, weights, benchmark_ticker, risk_free_rate=0.02):
    """Compare portfolio against a benchmark."""
    # Fetch benchmark data using Ticker API
    yf_ticker = yf.Ticker(benchmark_ticker)
    benchmark_hist = yf_ticker.history(start=data.index.min(), end=data.index.max())
    benchmark_hist.index = benchmark_hist.index.tz_localize(None)
    benchmark_data = benchmark_hist['Close']

    # Calculate returns
    portfolio_returns = data.pct_change().dropna().dot(weights)
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Align dates
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]

    # Beta
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance

    # Alpha
    portfolio_mean = portfolio_returns.mean()
    benchmark_mean = benchmark_returns.mean()
    rf_daily = risk_free_rate / TRADING_DAYS
    alpha = (portfolio_mean - (rf_daily + beta * (benchmark_mean - rf_daily))) * TRADING_DAYS

    # Other metrics
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(TRADING_DAYS)
    info_ratio = (portfolio_returns.mean() - benchmark_returns.mean()) * TRADING_DAYS / tracking_error if tracking_error > 0 else 0

    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]

    # Capture ratios
    up_mask = benchmark_returns > 0
    down_mask = benchmark_returns < 0

    up_capture = (portfolio_returns[up_mask].mean() / benchmark_returns[up_mask].mean() * 100) if up_mask.sum() > 0 else 0
    down_capture = (portfolio_returns[down_mask].mean() / benchmark_returns[down_mask].mean() * 100) if down_mask.sum() > 0 else 0

    return {
        'beta': beta,
        'alpha': alpha,
        'tracking_error': tracking_error,
        'information_ratio': info_ratio,
        'correlation': correlation,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'portfolio_annual_return': portfolio_returns.mean() * TRADING_DAYS,
        'benchmark_annual_return': benchmark_returns.mean() * TRADING_DAYS,
    }


# ============================================
# Sidebar
# ============================================

st.sidebar.title("📊 Engineer Investor")
st.sidebar.markdown("*Data-driven. No hype. Just math.*")
st.sidebar.markdown("---")

# Portfolio Selection
st.sidebar.header("Portfolio Settings")

preset = st.sidebar.selectbox(
    "Preset Portfolio",
    options=list(PRESET_PORTFOLIOS.keys()),
    index=2  # Default to Three-Fund
)

if preset != 'Custom':
    default_tickers = ', '.join(PRESET_PORTFOLIOS[preset].keys())
    default_weights = ', '.join([str(w) for w in PRESET_PORTFOLIOS[preset].values()])
else:
    default_tickers = 'VTI, VXUS, BND'
    default_weights = '0.4, 0.2, 0.4'

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
weights_input = st.sidebar.text_input("Weights (comma-separated)", default_weights)

# Date range
st.sidebar.header("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", datetime.now() - timedelta(days=5*365))
with col2:
    end_date = st.date_input("End", datetime.now())

# Risk-free rate
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005, format="%.1f%%")

st.sidebar.markdown("---")
st.sidebar.markdown("[@egr_investor](https://twitter.com/egr_investor)")
st.sidebar.markdown("[GitHub](https://github.com/engineerinvestor/Portfolio-Analysis)")
st.sidebar.caption("Not investment advice. Educational tools only.")


# ============================================
# Parse Portfolio
# ============================================

try:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    weights = np.array([float(w.strip()) for w in weights_input.split(',')])

    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights")
        st.stop()

    if not np.isclose(weights.sum(), 1.0):
        st.error(f"Weights must sum to 1.0 (currently {weights.sum():.2f})")
        st.stop()

except Exception as e:
    st.error(f"Error parsing portfolio: {e}")
    st.stop()


# ============================================
# Main Content
# ============================================

st.title("Portfolio Analyzer")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "🎲 Monte Carlo",
    "⚖️ Optimization",
    "📊 Benchmark",
    "ℹ️ About"
])


# ============================================
# Tab 1: Performance Analysis
# ============================================

with tab1:
    st.header("Performance Analysis")

    # Fetch data
    with st.spinner("Fetching market data..."):
        data = fetch_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if data is None or data.empty:
        st.error("No data available. Check ticker symbols and date range.")
        st.stop()

    # Calculate metrics
    metrics = calculate_portfolio_metrics(data, weights, risk_free_rate)

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annual Return", f"{metrics['Annual Return']*100:.2f}%")
    col2.metric("Annual Volatility", f"{metrics['Annual Volatility']*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col4.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    col5.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")

    # Allocation pie chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Allocation")
        fig_pie = px.pie(
            values=weights,
            names=tickers,
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Cumulative Returns")
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=metrics['cumulative_returns'].index,
            y=metrics['cumulative_returns'].values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        fig_cumulative.update_layout(
            yaxis_title='Growth of $1',
            xaxis_title='Date',
            hovermode='x unified'
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

    # Individual asset performance
    st.subheader("Individual Asset Returns")
    asset_returns = data.pct_change().dropna()
    asset_cumulative = (1 + asset_returns).cumprod()

    fig_assets = go.Figure()
    for col in asset_cumulative.columns:
        fig_assets.add_trace(go.Scatter(
            x=asset_cumulative.index,
            y=asset_cumulative[col].values,
            mode='lines',
            name=col
        ))
    fig_assets.update_layout(
        yaxis_title='Growth of $1',
        xaxis_title='Date',
        hovermode='x unified'
    )
    st.plotly_chart(fig_assets, use_container_width=True)


# ============================================
# Tab 2: Monte Carlo Simulation
# ============================================

with tab2:
    st.header("Monte Carlo Simulation")

    col1, col2, col3 = st.columns(3)
    with col1:
        mc_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
    with col2:
        mc_horizon = st.slider("Time Horizon (days)", 21, 1260, 252, 21)
    with col3:
        mc_initial = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = run_monte_carlo(data, weights, mc_simulations, mc_horizon, mc_initial)

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Final Value", f"${mc_results['median']:,.0f}")
        col2.metric("5th Percentile", f"${mc_results['p5']:,.0f}")
        col3.metric("95th Percentile", f"${mc_results['p95']:,.0f}")
        col4.metric("Probability of Loss", f"{mc_results['prob_loss']:.1f}%")

        # Plot simulation
        fig_mc = go.Figure()

        # Add percentile bands
        days = np.arange(mc_horizon)
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][95],
            mode='lines', line=dict(width=0),
            showlegend=False
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][5],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 100, 255, 0.2)',
            name='5th-95th Percentile'
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=mc_results['percentiles'][50],
            mode='lines', line=dict(color='blue', width=2),
            name='Median'
        ))
        fig_mc.add_hline(y=mc_initial, line_dash="dash", line_color="red",
                        annotation_text=f"Initial: ${mc_initial:,}")

        fig_mc.update_layout(
            title=f'Monte Carlo Simulation ({mc_simulations:,} paths)',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_mc, use_container_width=True)


# ============================================
# Tab 3: Portfolio Optimization
# ============================================

with tab3:
    st.header("Portfolio Optimization")

    strategy = st.selectbox(
        "Optimization Strategy",
        ["max_sharpe", "min_volatility"],
        format_func=lambda x: "Maximum Sharpe Ratio" if x == "max_sharpe" else "Minimum Volatility"
    )

    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner("Optimizing portfolio..."):
            optimal = optimize_portfolio(data, strategy, risk_free_rate)

        if optimal:
            st.subheader("Optimal Weights")

            col1, col2 = st.columns([1, 2])

            with col1:
                # Display weights
                for ticker, weight in optimal['weights'].items():
                    if weight > 0.01:
                        st.write(f"**{ticker}:** {weight*100:.1f}%")

            with col2:
                # Pie chart
                fig_opt = px.pie(
                    values=list(optimal['weights'].values()),
                    names=list(optimal['weights'].keys()),
                    hole=0.4,
                    title="Optimized Allocation"
                )
                st.plotly_chart(fig_opt, use_container_width=True)

            # Metrics comparison
            st.subheader("Comparison: Current vs Optimized")

            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{optimal['return']*100:.2f}%",
                       f"{(optimal['return'] - metrics['Annual Return'])*100:+.2f}%")
            col2.metric("Volatility", f"{optimal['volatility']*100:.2f}%",
                       f"{(optimal['volatility'] - metrics['Annual Volatility'])*100:+.2f}%")
            col3.metric("Sharpe Ratio", f"{optimal['sharpe']:.2f}",
                       f"{optimal['sharpe'] - metrics['Sharpe Ratio']:+.2f}")


# ============================================
# Tab 4: Benchmark Comparison
# ============================================

with tab4:
    st.header("Benchmark Comparison")

    benchmark_ticker = st.selectbox(
        "Select Benchmark",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{x} - {BENCHMARKS[x]}"
    )

    if st.button("Compare to Benchmark", type="primary"):
        with st.spinner("Calculating benchmark comparison..."):
            comparison = calculate_benchmark_comparison(data, weights, benchmark_ticker, risk_free_rate)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Beta", f"{comparison['beta']:.3f}")
        col2.metric("Alpha (annual)", f"{comparison['alpha']*100:.2f}%")
        col3.metric("Tracking Error", f"{comparison['tracking_error']*100:.2f}%")
        col4.metric("Information Ratio", f"{comparison['information_ratio']:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", f"{comparison['correlation']:.3f}")
        col2.metric("Up Capture", f"{comparison['up_capture']:.1f}%")
        col3.metric("Down Capture", f"{comparison['down_capture']:.1f}%")
        col4.metric("Return Difference",
                   f"{(comparison['portfolio_annual_return'] - comparison['benchmark_annual_return'])*100:+.2f}%")

        # Cumulative returns comparison
        st.subheader("Cumulative Returns Comparison")

        port_cum = (1 + comparison['portfolio_returns']).cumprod()
        bench_cum = (1 + comparison['benchmark_returns']).cumprod()

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=port_cum.index, y=port_cum.values,
            mode='lines', name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        fig_bench.add_trace(go.Scatter(
            x=bench_cum.index, y=bench_cum.values,
            mode='lines', name=f'Benchmark ({benchmark_ticker})',
            line=dict(color='orange', width=2)
        ))
        fig_bench.update_layout(
            yaxis_title='Growth of $1',
            xaxis_title='Date',
            hovermode='x unified'
        )
        st.plotly_chart(fig_bench, use_container_width=True)


# ============================================
# Tab 5: About
# ============================================

with tab5:
    st.header("About Engineer Investor Portfolio Analyzer")

    st.markdown("""
    ### An electrical engineer's approach to portfolio analysis

    This tool was built with the philosophy that investors deserve **transparent, data-driven tools**
    without hidden agendas or gamification.

    **What This Tool Does:**
    - Calculates standard portfolio metrics (return, volatility, Sharpe ratio)
    - Runs Monte Carlo simulations to visualize uncertainty
    - Optimizes portfolios using mean-variance optimization
    - Compares your portfolio against common benchmarks

    **What This Tool Does NOT Do:**
    - Sell you anything
    - Gamify your investing experience
    - Encourage excessive trading
    - Provide "hot tips" or "signals"

    ---

    ### Disclaimer

    **This is not investment advice.**

    These tools are for educational purposes only. Past performance does not guarantee
    future results. Always do your own research and consider consulting a qualified
    financial advisor before making investment decisions.

    ---

    ### Open Source

    This project is open source and available on GitHub:

    [github.com/engineerinvestor/Portfolio-Analysis](https://github.com/engineerinvestor/Portfolio-Analysis)

    Contributions welcome!

    ---

    ### Contact

    - Twitter: [@egr_investor](https://twitter.com/egr_investor)
    - GitHub: [engineerinvestor](https://github.com/engineerinvestor)
    """)
