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

    def eval_grid(self, projected_expr, a, b, AA, BB, evaluator='numpy',
                  traction_expr=None):
        """Evaluate on grid, return dict of numpy arrays.

        evaluator: 'numpy' (fast, current), 'hybrid' (TractionValue per-pixel),
                   'sympy' (exact, slow)
        traction_expr: the pre-projection expression (needed for hybrid/sympy modes)

        For hybrid/sympy, returns a GridComputation for async execution.
        For numpy, returns the result dict directly.
        """
        from evaluator import eval_grid_hybrid, eval_grid_sympy, GridComputation

        if evaluator in ('hybrid', 'sympy') and traction_expr is not None:
            func = eval_grid_hybrid if evaluator == 'hybrid' else eval_grid_sympy
            comp = GridComputation()
            comp.start(func, traction_expr, a, b, AA, BB)
            return comp  # caller must poll comp.is_done()
        else:
            # Fast numpy path (default) — synchronous
            try:
                f = lambdify((a, b), projected_expr, modules=['numpy'])
            except Exception:
                return None
            try:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    Z = f(AA, BB)
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
    log_mag = np.log(mag + 1e-15)

    invalid = ~np.isfinite(Z) | (mag < 1e-15)
    phase[invalid] = np.nan
    log_mag[invalid] = np.nan

    brightness = 0.5 + np.arctan(log_mag) / np.pi
    brightness = np.clip(brightness, 0.12, 0.95)
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
