"""
Complex Lie Exponential Projection
===================================
Maps traction expressions to C via the Lie group exponential:
    0^z -> e^(-W*z)  where W = sqrt(-i*pi)

In exponents, omega acts as W: omega = W in exponent space.

This is the standard/default projection for the COTT calculator.
"""

import numpy as np
from sympy import symbols, lambdify
from traction import traction_simplify, project_complex, Zero, Omega, Pow, Mul, Rational
from projections.base import Projection, z_to_metrics


class ComplexLieProjection(Projection):
    name = 'complex_lie'
    description = 'Standard complex projection via Lie exponential (0^z = e^{-Wz})'
    output_keys = ['Re', 'Im', 'mag', 'phase']

    def format_projection(self, traction_expr):
        """Format as Lie exponential / Euler complex form."""
        from calculator import format_complex, format_approx
        result = format_complex(traction_expr)
        if result:
            return result
        result = format_approx(traction_expr)
        if result:
            return result
        return ''

    def native_x(self, a, b):
        """Complex Lie native unit: x = p + q*0^(w/2) (complex plane)."""
        return a + b * Pow(Zero(), Mul(Omega(), Rational(1, 2)))

    def project_expr(self, traction_expr, a, b):
        """Project traction expression to a complex SymPy expression in (a, b)."""
        # Expression arrives already substituted and simplified by compute_phase_grid
        projected = project_complex(traction_expr)

        try:
            projected = projected.expand()
        except Exception:
            pass

        return projected

    def eval_grid(self, projected_expr, a, b, AA, BB, **kwargs):
        """Evaluate on grid, return dict of numpy arrays."""
        try:
            f = lambdify((a, b), projected_expr, modules=['numpy'])
        except Exception:
            return None
        try:
            # Use complex arrays so negative bases handle fractional powers correctly
            # (e.g. (-1.0)**0.5 is NaN in real numpy, but (-1.0+0j)**0.5 = 1j)
            AA_c = AA.astype(complex)
            BB_c = BB.astype(complex)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                Z = f(AA_c, BB_c)
                if np.isscalar(Z) or (isinstance(Z, np.ndarray) and Z.ndim == 0):
                    Z = np.full_like(AA, complex(Z), dtype=complex)
                Z = np.asarray(Z, dtype=complex)
        except Exception:
            return None

        return z_to_metrics(Z)


# Auto-register on import
_instance = ComplexLieProjection()
_instance.register_self()
