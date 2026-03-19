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
from projections.base import Projection


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

        return _z_to_metrics(Z)


def _z_to_metrics(Z):
    """Convert a complex Z grid to the standard metrics dict."""
    phase = np.angle(Z)
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    mag = np.abs(Z)
    # Clamp magnitude floor to avoid log(0), but don't discard small values
    log_mag = np.log(np.maximum(mag, 1e-300))

    # Only truly invalid: non-finite Z or exactly zero
    invalid = ~np.isfinite(Z) | (Z == 0)
    phase[invalid] = np.nan
    log_mag[invalid] = np.nan

    brightness = 0.5 + np.arctan(log_mag) / np.pi
    brightness = np.clip(brightness, 0.12, 0.95)
    # Invalid pixels get zero brightness (renders as black, not gray)
    brightness[invalid] = 0.0

    return {
        'Z': Z,
        'Re': Z.real,
        'Im': Z.imag,
        'mag': mag,
        'log_mag': log_mag,
        'phase': phase,
        'brightness': brightness,
    }


# Auto-register on import
_instance = ComplexLieProjection()
_instance.register_self()
