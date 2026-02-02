"""
Base class for report sections.
"""

from abc import ABC, abstractmethod
from typing import Any

from jinja2 import Template


class ReportSection(ABC):
    """
    Abstract base class for report sections.

    Each section is responsible for:
    1. Computing its data/metrics
    2. Generating any charts
    3. Rendering its HTML template

    Parameters
    ----------
    template : jinja2.Template
        The Jinja2 template for this section
    """

    def __init__(self, template: Template):
        self.template = template

    @abstractmethod
    def compute_data(self) -> dict[str, Any]:
        """
        Compute the data needed for this section.

        Returns
        -------
        dict
            Dictionary of data to pass to the template
        """
        pass

    def render(self) -> str:
        """
        Render the section to HTML.

        Returns
        -------
        str
            Rendered HTML string
        """
        data = self.compute_data()
        return self.template.render(**data)
