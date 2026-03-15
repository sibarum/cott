"""
Base class for traction projection plugins.

A projection maps traction expressions to another domain (e.g. complex numbers,
polar coordinates, custom algebras) and provides numeric evaluation for
visualization.

To create a new projection:
    1. Create a new .py file in solver/projections/
    2. Subclass Projection
    3. Implement project_expr() and eval_grid()
    4. Call self.register() at module level

The projection will be auto-discovered and registered when the projections
package is imported.
"""

import numpy as np
from registry import register


class Projection:
    """Base class for traction projections."""

    name = ''
    description = ''

    # Keys this projection provides in eval_grid output
    output_keys = ['Re', 'Im', 'mag', 'phase']

    def native_x(self, a, b):
        """
        Return the projection's native unit coordinate as a traction expression.
        a = horizontal grid symbol (p), b = vertical grid symbol (q).

        Each projection defines what 'x' means in its coordinate system.
        """
        raise NotImplementedError

    def project_expr(self, traction_expr, a, b):
        """
        Project a traction expression to the target domain.

        Args:
            traction_expr: A SymPy expression (may contain Zero, Omega, etc.)
            a, b: Real SymPy symbols for the grid axes

        Returns:
            A SymPy expression suitable for lambdify, or None if projection fails.
        """
        raise NotImplementedError

    def eval_grid(self, projected_expr, a, b, AA, BB):
        """
        Evaluate a projected expression on a numpy grid.

        Args:
            projected_expr: Result of project_expr()
            a, b: The SymPy symbols used in project_expr
            AA, BB: 2D numpy arrays (meshgrid)

        Returns:
            dict of 2D numpy arrays with keys from output_keys.
            At minimum: {Re, Im, mag, phase}.
            May include additional keys specific to the projection.
        """
        raise NotImplementedError

    def register_self(self):
        """Register this projection in the global registry."""
        register('projection', self.name, self, description=self.description)
