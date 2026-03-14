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

    def project_expr(self, traction_expr, a, b):
        """Project traction expression to a complex SymPy expression in (a, b)."""
        from sympy import Symbol

        x_sym = Symbol('x')
        y_sym = Symbol('y')
        has_x = traction_expr.has(x_sym)
        has_y = traction_expr.has(y_sym)

        if has_x and has_y:
            # Two-variable mode: x=a (horizontal), y=b (vertical)
            subst = traction_expr.subs([(x_sym, a), (y_sym, b)])
        elif has_x:
            # Single-variable complex plane: x -> a + b*0^(w/2)
            i_traction = Pow(Zero(), Mul(Omega(), Rational(1, 2)))
            subst = traction_expr.subs(x_sym, a + b * i_traction)
        elif has_y:
            # Only y: treat as single-variable complex plane
            i_traction = Pow(Zero(), Mul(Omega(), Rational(1, 2)))
            subst = traction_expr.subs(y_sym, a + b * i_traction)
        else:
            return None

        simplified = traction_simplify(subst)
        projected = project_complex(simplified)

        try:
            projected = projected.expand()
        except Exception:
            pass

        return projected

    def eval_grid(self, projected_expr, a, b, AA, BB):
        """Evaluate on grid, return dict of numpy arrays."""
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
