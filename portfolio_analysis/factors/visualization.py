"""
Factor analysis visualization functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

from portfolio_analysis.factors.models import RegressionResults


class FactorVisualization:
    """
    Static methods for visualizing factor analysis results.

    Examples
    --------
    >>> from portfolio_analysis.factors import FactorVisualization
    >>> FactorVisualization.plot_factor_exposures(regression_results)
    >>> FactorVisualization.plot_rolling_betas(rolling_data)
    >>> FactorVisualization.plot_return_attribution(attribution_dict)
    """

    @staticmethod
    def plot_factor_exposures(
        results: RegressionResults,
        figsize: tuple = (10, 6),
        show_significance: bool = True
    ) -> None:
        """
        Plot factor exposures (betas) as a bar chart.

        Parameters
        ----------
        results : RegressionResults
            Results from FactorRegression
        figsize : tuple, default (10, 6)
            Figure size
        show_significance : bool, default True
            Color bars by statistical significance
        """
        factors = results.factors
        betas = [results.betas[f] for f in factors]
        pvalues = [results.beta_pvalues[f] for f in factors]

        # Color by significance
        if show_significance:
            colors = ['green' if p < 0.05 else 'gray' for p in pvalues]
        else:
            colors = ['steelblue'] * len(factors)

        plt.figure(figsize=figsize)
        bars = plt.bar(factors, betas, color=colors, edgecolor='black', alpha=0.8)

        # Add horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels on bars
        for bar, beta in zip(bars, betas):
            height = bar.get_height()
            plt.annotate(
                f'{beta:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -10),
                textcoords='offset points',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10
            )

        plt.xlabel('Factor')
        plt.ylabel('Beta (Factor Loading)')
        plt.title(f'Factor Exposures - {results.model} Model\n'
                  f'(R² = {results.r_squared:.3f}, Alpha = {results.alpha*100:.2f}%)')

        if show_significance:
            plt.legend(['p < 0.05 (significant)', 'p >= 0.05'],
                       handles=[plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.8),
                                plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.8)],
                       loc='best')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rolling_betas(
        rolling_data: pd.DataFrame,
        figsize: tuple = (12, 8),
        factors: Optional[List[str]] = None
    ) -> None:
        """
        Plot rolling factor betas over time.

        Parameters
        ----------
        rolling_data : pd.DataFrame
            Output from FactorRegression.run_rolling_regression()
        figsize : tuple, default (12, 8)
            Figure size
        factors : list of str, optional
            Specific factors to plot. If None, plots all.
        """
        if factors is None:
            # Get all columns except 'alpha' and 'r_squared'
            factors = [c for c in rolling_data.columns
                       if c not in ['alpha', 'r_squared']]

        n_factors = len(factors)
        fig, axes = plt.subplots(n_factors + 1, 1, figsize=figsize, sharex=True)

        # Plot each factor beta
        for i, factor in enumerate(factors):
            ax = axes[i]
            ax.plot(rolling_data.index, rolling_data[factor], linewidth=1.5)
            ax.axhline(y=rolling_data[factor].mean(), color='red',
                       linestyle='--', alpha=0.7, label='Mean')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel(factor)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot rolling alpha
        ax = axes[-1]
        ax.plot(rolling_data.index, rolling_data['alpha'] * 100, linewidth=1.5, color='green')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Alpha (%)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        fig.suptitle('Rolling Factor Exposures', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_return_attribution(
        attribution: Dict[str, float],
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot return attribution as a waterfall chart.

        Parameters
        ----------
        attribution : dict
            Output from FactorAttribution.decompose_returns()
        figsize : tuple, default (10, 6)
            Figure size
        """
        # Order: risk-free, factors, alpha -> total
        components = []
        values = []

        # Start with risk-free
        if 'risk_free' in attribution:
            components.append('Risk-Free')
            values.append(attribution['risk_free'] * 100)

        # Add factors (exclude 'total', 'risk_free', 'alpha')
        factor_keys = [k for k in attribution.keys()
                       if k not in ['total', 'risk_free', 'alpha']]
        for factor in factor_keys:
            components.append(factor)
            values.append(attribution[factor] * 100)

        # Add alpha
        if 'alpha' in attribution:
            components.append('Alpha')
            values.append(attribution['alpha'] * 100)

        # Calculate cumulative for waterfall
        cumulative = np.cumsum([0] + values[:-1])
        total = sum(values)

        plt.figure(figsize=figsize)

        # Color bars based on positive/negative
        colors = ['green' if v >= 0 else 'red' for v in values]

        # Create waterfall bars
        bars = plt.bar(components, values, bottom=cumulative,
                       color=colors, edgecolor='black', alpha=0.8)

        # Add total bar
        plt.bar(['Total'], [total], color='steelblue', edgecolor='black', alpha=0.8)

        # Add value labels
        for i, (comp, val) in enumerate(zip(components, values)):
            height = cumulative[i] + val
            plt.annotate(
                f'{val:.2f}%',
                xy=(i, height),
                xytext=(0, 3 if val >= 0 else -10),
                textcoords='offset points',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9
            )

        # Total label
        plt.annotate(
            f'{total:.2f}%',
            xy=(len(components), total),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.ylabel('Return Contribution (%)')
        plt.title('Return Attribution by Factor')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_factor_tilts(
        tilts: Dict[str, float],
        figsize: tuple = (8, 8)
    ) -> None:
        """
        Plot characteristic-based factor tilts as a radar chart.

        Parameters
        ----------
        tilts : dict
            Output from FactorExposures.get_all_tilts()
        figsize : tuple, default (8, 8)
            Figure size
        """
        factors = list(tilts.keys())
        values = list(tilts.values())

        # Close the radar chart
        angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
        values = values + [values[0]]
        angles = angles + [angles[0]]
        factors = factors + [factors[0]]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Plot the tilt values
        ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
        ax.fill(angles, values, alpha=0.25)

        # Add reference circles
        for val in [-1, -0.5, 0, 0.5, 1]:
            ax.plot(angles, [val] * len(angles), '--', color='gray', alpha=0.3, linewidth=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.capitalize() for f in factors[:-1]], fontsize=11)
        ax.set_ylim(-1.2, 1.2)

        # Add gridlines at factor positions
        ax.set_thetagrids(np.degrees(angles[:-1]), [f.capitalize() for f in factors[:-1]])

        plt.title('Portfolio Factor Tilts\n(Characteristic-Based)', fontsize=14, y=1.1)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        figsize: tuple = (12, 5)
    ) -> None:
        """
        Plot comparison of different factor models.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output from FactorRegression.compare_models()
        figsize : tuple, default (12, 5)
            Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        models = comparison_df['Model'].tolist()
        x = np.arange(len(models))
        width = 0.6

        # Alpha comparison
        ax = axes[0]
        alphas = comparison_df['Alpha (%)'].values
        colors = ['green' if a > 0 else 'red' for a in alphas]
        bars = ax.bar(x, alphas, width, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel('Alpha (%)')
        ax.set_title('Alpha by Model')
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance indicators
        for i, (bar, pval) in enumerate(zip(bars, comparison_df['Alpha p-value'])):
            if pval < 0.05:
                ax.annotate('*', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha='center', fontsize=14)

        # R-squared comparison
        ax = axes[1]
        r2 = comparison_df['R-squared'].values
        ax.bar(x, r2, width, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel('R-squared')
        ax.set_title('Model Fit (R²)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Market beta comparison
        ax = axes[2]
        mkt_betas = comparison_df['Mkt Beta'].values
        ax.bar(x, mkt_betas, width, color='purple', edgecolor='black', alpha=0.8)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Market (β=1)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel('Market Beta')
        ax.set_title('Market Exposure')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Factor Model Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_factor_frontier(
        frontier_df: pd.DataFrame,
        factor: str,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot factor-efficient frontier.

        Parameters
        ----------
        frontier_df : pd.DataFrame
            Output from FactorOptimizer.generate_factor_frontier()
        factor : str
            Name of the factor
        figsize : tuple, default (10, 6)
            Figure size
        """
        beta_col = f'{factor}_beta'

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot: Return vs Factor Beta
        ax = axes[0]
        scatter = ax.scatter(
            frontier_df[beta_col],
            frontier_df['return'] * 100,
            c=frontier_df['sharpe_ratio'],
            cmap='RdYlGn',
            s=60,
            edgecolors='black',
            alpha=0.8
        )
        ax.set_xlabel(f'{factor} Beta')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title(f'Return vs {factor} Exposure')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

        # Plot: Sharpe vs Factor Beta
        ax = axes[1]
        ax.plot(frontier_df[beta_col], frontier_df['sharpe_ratio'],
                'o-', linewidth=2, markersize=6)
        ax.set_xlabel(f'{factor} Beta')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Risk-Adjusted Return vs {factor} Exposure')
        ax.grid(True, alpha=0.3)

        # Mark optimal point
        max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
        opt_beta = frontier_df.loc[max_sharpe_idx, beta_col]
        opt_sharpe = frontier_df.loc[max_sharpe_idx, 'sharpe_ratio']
        ax.scatter([opt_beta], [opt_sharpe], s=150, c='red', marker='*',
                   zorder=5, label=f'Optimal (β={opt_beta:.2f})')
        ax.legend(loc='best')

        plt.suptitle(f'{factor} Factor Efficient Frontier', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_risk_attribution(
        risk_decomp: Dict[str, float],
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot risk (variance) attribution as a stacked bar or pie chart.

        Parameters
        ----------
        risk_decomp : dict
            Output from FactorAttribution.decompose_risk()
        figsize : tuple, default (10, 6)
            Figure size
        """
        # Extract components (exclude 'total' and 'r_squared')
        components = [k for k in risk_decomp.keys() if k not in ['total', 'r_squared']]
        values = [risk_decomp[k] for k in components]
        total = risk_decomp['total']

        # Convert to percentages
        percentages = [v / total * 100 if total > 0 else 0 for v in values]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Pie chart
        ax = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        wedges, texts, autotexts = ax.pie(
            percentages,
            labels=components,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0.02] * len(components)
        )
        ax.set_title('Variance Attribution (%)')

        # Bar chart with variance values
        ax = axes[1]
        bars = ax.barh(components, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Variance Contribution')
        ax.set_title('Variance Attribution (Absolute)')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.6f}',
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords='offset points',
                ha='left', va='center',
                fontsize=9
            )

        plt.suptitle(f'Risk Attribution (R² = {risk_decomp.get("r_squared", 0):.3f})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
