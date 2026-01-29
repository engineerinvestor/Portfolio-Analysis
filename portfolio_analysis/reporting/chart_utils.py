"""
Chart utilities for HTML report generation.

Provides functions for converting matplotlib figures to base64-encoded
images suitable for embedding in HTML documents.
"""

import base64
import io
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def fig_to_base64(
    fig: Figure,
    format: str = "png",
    dpi: int = 100,
    close_fig: bool = True
) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert
    format : str, default "png"
        Image format (png, svg, etc.)
    dpi : int, default 100
        Resolution in dots per inch
    close_fig : bool, default True
        Whether to close the figure after conversion

    Returns
    -------
    str
        Base64-encoded image string suitable for HTML img src
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    if close_fig:
        plt.close(fig)

    mime_type = f"image/{format}"
    return f"data:{mime_type};base64,{img_base64}"


def create_figure(
    figsize: tuple = (10, 6),
    style: Optional[str] = None
) -> tuple:
    """
    Create a matplotlib figure with consistent styling.

    Parameters
    ----------
    figsize : tuple, default (10, 6)
        Figure size in inches
    style : str, optional
        Matplotlib style to use

    Returns
    -------
    tuple
        (fig, ax) tuple
    """
    if style:
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    return fig, ax
