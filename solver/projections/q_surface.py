"""
Q-Surface Projection
=====================
Maps traction zero-powers to coordinates on a q-norm unit surface.

For 0^t:
    phase     = t           (the exponent itself)
    magnitude = (1 - t^2)^(1/q)   (orthogonal component on the q-surface)

The parameter q selects the surface geometry:
    q=1   -> taxicab (diamond)
    q=2   -> Euclidean (circle)
    q->inf -> Chebyshev (square)

For |t| > 1, the magnitude becomes a root of a negative number,
which in traction naturally rotates into the zero-power domain
via 0^(w/2). This gives the projection its "projective" character:
infinity wraps rather than diverges.

Non-zero-power parts of an expression (regular scalars) pass through
as standard magnitudes.
"""

import numpy as np
from sympy import Symbol, Rational, Pow, Mul, symbols, lambdify, S
from traction import Zero, Omega, traction_simplify
from projections.base import Projection


class QSurfaceProjection(Projection):
    name = 'q_surface'
    description = 'Projective coordinates on a q-norm unit surface'
    output_keys = ['Re', 'Im', 'mag', 'phase', 'q_phase', 'q_mag']

    def __init__(self, q=2):
        self.q = q

    def native_x(self, a, b):
        """Q-surface native unit: x = 0^(p + q*0^(w/2)) (complex plane lifted into zero-power)."""
        i_traction = Pow(Zero(), Mul(Omega(), Rational(1, 2)))
        return Pow(Zero(), a + b * i_traction)

    def project_expr(self, traction_expr, a, b):
        """
        Project traction expression to q-surface coordinates.

        Returns a tuple expression (q_phase_expr, q_mag_expr, scalar_expr)
        packed into a dict-like structure for eval_grid.
        We store them as a plain tuple attached to self for eval_grid to unpack.
        """
        # Substitution is now handled by compute_phase_grid
        # This method receives an expression already in terms of a, b

        simplified = traction_simplify(traction_expr)

        # Project each term through the q-surface formula.
        # For Add expressions, decompose and project each term separately.
        from sympy import Add as SymAdd
        from traction import project_complex, W_CONST, _project

        terms = SymAdd.make_args(simplified) if isinstance(simplified, SymAdd) else [simplified]

        q = self.q
        total_phase = S.Zero
        total_mag = S.Zero

        any_decomposed = False
        for term in terms:
            exponent, scalar = self._decompose(term)
            if exponent is None:
                continue

            any_decomposed = True
            # Project exponent (ω → W)
            exp_with_w = exponent.subs(Omega(), W_CONST) if exponent.has(Omega) else exponent
            proj_exp = _project(exp_with_w)
            proj_scalar = project_complex(scalar) if scalar != S.One else S.One

            # Accumulate q-coordinates weighted by scalar
            total_phase = total_phase + proj_scalar * proj_exp
            total_mag = total_mag + proj_scalar * (1 - proj_exp**2) ** Rational(1, q)

        if not any_decomposed:
            return None

        self._exprs = (total_phase, total_mag, S.One)
        return total_phase

    def eval_grid(self, projected_expr, a, b, AA, BB, evaluator='numpy',
                  traction_expr=None):
        """Evaluate on grid, return dict of numpy arrays."""
        q_phase_expr, q_mag_expr, scalar_expr = self._exprs

        try:
            if q_phase_expr is not None:
                f_phase = lambdify((a, b), q_phase_expr, modules=['numpy'])
                f_mag = lambdify((a, b), q_mag_expr, modules=['numpy'])
            f_scalar = lambdify((a, b), scalar_expr, modules=['numpy'])
        except Exception:
            return None

        try:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                if q_phase_expr is not None:
                    Q_phase = np.asarray(f_phase(AA, BB), dtype=complex)
                    Q_mag = np.asarray(f_mag(AA, BB), dtype=complex)
                else:
                    Q_phase = np.zeros_like(AA, dtype=complex)
                    Q_mag = np.zeros_like(AA, dtype=complex)

                Z_scalar = f_scalar(AA, BB)
                if np.isscalar(Z_scalar) or (isinstance(Z_scalar, np.ndarray) and Z_scalar.ndim == 0):
                    Z_scalar = np.full_like(AA, complex(Z_scalar), dtype=complex)
                Z_scalar = np.asarray(Z_scalar, dtype=complex)
        except Exception:
            return None

        # Combine: q-coordinates scaled by scalar factor
        Z = Z_scalar * (Q_phase + 1j * Q_mag.real)

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
            'q_phase': Q_phase,
            'q_mag': Q_mag,
        }

    def _decompose(self, expr):
        """
        Decompose a traction expression into (exponent, scalar) where
        the expression is scalar * 0^exponent.

        Returns (None, None) if the expression is not in zero-power form.
        """
        # Direct zero power: 0^exp
        if isinstance(expr, Pow) and isinstance(expr.base, Zero):
            return expr.exp, S.One

        # Product: scalar * 0^exp
        if isinstance(expr, Mul):
            zero_exp = None
            scalar_parts = []
            for arg in Mul.make_args(expr):
                if isinstance(arg, Zero):
                    zero_exp = S.One if zero_exp is None else zero_exp + S.One
                elif isinstance(arg, Pow) and isinstance(arg.base, Zero):
                    zero_exp = arg.exp if zero_exp is None else zero_exp + arg.exp
                else:
                    scalar_parts.append(arg)
            if zero_exp is not None:
                scalar = Mul(*scalar_parts) if scalar_parts else S.One
                return zero_exp, scalar

        # Bare Zero
        if isinstance(expr, Zero):
            return S.One, S.One

        return None, None


# Auto-register on import
_instance = QSurfaceProjection(q=2)
_instance.register_self()
