"""
Q-Surface Projection (Recursive Identity Decomposition)
=========================================================
Maps traction expressions to coordinates via recursive evaluation
of the zero-power identity layers.

For 0^E:
    phase     = E (the simplified exponent IS the phase coordinate)
    magnitude = one more layer of 0^E:
                - if identities fire (0^(0^n)=hin, 0^(w^n)=-n) -> that scalar
                - if result stays zero-class (0^6) -> magnitude 0
                - if result stays omega-class -> magnitude infinity

The phase space origin is unity (0^0 = 1, phase 0), not zero (0^1 = 0, phase 1).

For bare scalars (not in 0^(...) form):
    phase = 1 (one layer deep in zero-power encoding)
    magnitude = the scalar value
"""

import numpy as np
from sympy import Symbol, Rational, Pow, Mul, symbols, lambdify, S, Integer, Number, Add
from traction import Zero, Omega, Null, traction_simplify, project_complex, OMEGA_EXP_VAL, _project
from projections.base import Projection


class QSurfaceProjection(Projection):
    name = 'q_surface'
    description = 'Recursive identity decomposition (phase = exponent, magnitude = 0^exponent)'
    output_keys = ['Re', 'Im', 'mag', 'phase', 'q_phase', 'q_mag']

    def __init__(self, q=2):
        self.q = q  # reserved for future use

    def format_projection(self, traction_expr):
        """Format as q-surface coordinates: phase and magnitude."""
        from calculator import format_result

        phase, magnitude = self._recursive_decompose(traction_expr)
        phase_str = format_result(phase) if phase is not None else '?'
        mag_str = format_result(magnitude) if magnitude is not None else '?'
        return f'phase={phase_str}, magnitude={mag_str}'

    def native_x(self, a, b):
        """Q-surface native unit: x = 0^(p + q*0^(w/2)) (complex plane lifted into zero-power)."""
        i_traction = Pow(Zero(), Mul(Omega(), Rational(1, 2)))
        return Pow(Zero(), a + b * i_traction)

    def project_expr(self, traction_expr, a, b):
        """
        Project traction expression to q-surface coordinates.

        Phase = the exponent (projected to C for numeric evaluation)
        Magnitude = 0^exponent evaluated one more layer (projected to C)

        For visualization, we encode as Z = magnitude + i*phase so both
        are available in the complex grid.
        """
        simplified = traction_simplify(traction_expr)
        phase_expr, mag_expr = self._recursive_decompose(simplified)

        if phase_expr is None:
            return None

        # Project both to C for numeric evaluation
        phase_proj = _project_for_eval(phase_expr)
        mag_proj = _project_for_eval(mag_expr)

        self._exprs = (phase_proj, mag_proj)
        return phase_proj  # non-None signals success

    def eval_grid(self, projected_expr, a, b, AA, BB, **kwargs):
        """Evaluate on grid, return dict of numpy arrays."""
        phase_expr, mag_expr = self._exprs

        try:
            f_phase = lambdify((a, b), phase_expr, modules=['numpy'])
            f_mag = lambdify((a, b), mag_expr, modules=['numpy'])
        except Exception:
            return None

        try:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                Q_phase = np.asarray(f_phase(AA, BB), dtype=complex)
                Q_mag = np.asarray(f_mag(AA, BB), dtype=complex)

                if np.isscalar(Q_phase) or (isinstance(Q_phase, np.ndarray) and Q_phase.ndim == 0):
                    Q_phase = np.full_like(AA, complex(Q_phase), dtype=complex)
                if np.isscalar(Q_mag) or (isinstance(Q_mag, np.ndarray) and Q_mag.ndim == 0):
                    Q_mag = np.full_like(AA, complex(Q_mag), dtype=complex)
        except Exception:
            return None

        # For the standard coloring pipeline, encode as Z = mag + i*phase
        # This gives: |Z| from magnitude, arg(Z) from phase
        Z = Q_mag + 1j * Q_phase.real

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

    def _recursive_decompose(self, expr):
        """
        Recursively decompose a traction expression into (phase, magnitude).

        For 0^E: factor E into rational_part * class_part, then:
            phase = rational_part
            magnitude = 0^(class_part) evaluated via identities
        For Add exponents: 0^(a+b) = 0^a * 0^b, phases add, magnitudes multiply.
        For scalars: phase = 1, magnitude = the scalar value
        """
        expr = traction_simplify(expr)

        # Zero-power: 0^exponent
        if isinstance(expr, Pow) and isinstance(expr.base, Zero):
            exponent = traction_simplify(expr.exp)

            # Add exponent: 0^(a+b) = 0^a * 0^b → decompose each term, combine
            if isinstance(exponent, Add):
                return self._decompose_additive_exponent(exponent)

            rational_part, class_part = self._factor_exponent(exponent)
            magnitude = self._eval_one_layer(class_part)
            return rational_part, magnitude

        # Bare Zero = 0^1
        if isinstance(expr, Zero):
            return S.One, S.Zero  # phase 1, magnitude 0 (zero-class)

        # Bare Omega = 0^(-1)
        if isinstance(expr, Omega):
            return S.NegativeOne, Omega()  # phase -1, magnitude omega

        # Scalar: phase = 1, magnitude = the scalar itself
        if isinstance(expr, (Integer, Rational, Number)):
            return S.One, expr

        # Null: phase = 0, magnitude = 0
        if isinstance(expr, Null):
            return S.Zero, S.Zero

        # Symbol or complex expression: phase = 1, magnitude = expr
        return S.One, expr

    def _decompose_additive_exponent(self, exponent):
        """
        Decompose an Add exponent: 0^(a+b+...) = 0^a * 0^b * ...

        Phases add, magnitudes multiply with class absorption:
        - scalar × zero-class = zero-class (magnitude 0)
        - scalar × scalar = scalar
        - -0 = 0
        """
        total_phase = S.Zero
        total_mag = S.One
        mag_is_zero = False

        for term in Add.make_args(exponent):
            term = traction_simplify(term)
            # Decompose this term as 0^term
            rational_part, class_part = self._factor_exponent(term)
            term_mag = self._eval_one_layer(class_part)

            # Phase: additive
            total_phase = total_phase + rational_part

            # Magnitude: multiplicative with class absorption
            if isinstance(term_mag, Zero) or term_mag == S.Zero:
                mag_is_zero = True
            elif isinstance(term_mag, Omega):
                # omega × anything = depends, but for now track separately
                if not mag_is_zero:
                    total_mag = total_mag * term_mag
            else:
                total_mag = total_mag * term_mag

        # -0 = 0
        if mag_is_zero:
            total_mag = S.Zero

        return traction_simplify(total_phase), traction_simplify(total_mag)

    def _factor_exponent(self, exponent):
        """
        Factor an exponent into (rational_part, class_part) where:
            exponent = rational_part * class_part

        The rational_part becomes the phase coordinate.
        The class_part determines magnitude via 0^(class_part).

        Examples:
            1/2         -> (1/2, 1)      phase=1/2, mag=0^1=zero-class
            w/2         -> (1/2, w)      phase=1/2, mag=0^w=-1
            w^3         -> (1, w^3)      phase=1, mag=0^(w^3)=-3
            0^4         -> (1, 0^4)      phase=1, mag=0^(0^4)=4
            2           -> (2, 1)        phase=2, mag=0^1=zero-class
            p + q*0^(w/2) -> (1, p+q*0^(w/2))  symbolic, no factoring
        """
        exponent = traction_simplify(exponent)

        # Plain rational/integer: rational_part = the number, class_part = 1
        if isinstance(exponent, (Integer, Rational)):
            return exponent, S.One

        # Bare Omega: rational_part = 1, class_part = omega
        if isinstance(exponent, Omega):
            return S.One, Omega()

        # Bare Zero: rational_part = 1, class_part = zero
        if isinstance(exponent, Zero):
            return S.One, Zero()

        # Mul: separate rational coefficients from traction factors
        if isinstance(exponent, Mul):
            rational_parts = []
            class_parts = []
            for arg in Mul.make_args(exponent):
                if isinstance(arg, (Integer, Rational)):
                    rational_parts.append(arg)
                else:
                    class_parts.append(arg)

            rational = Mul(*rational_parts) if rational_parts else S.One
            cls = Mul(*class_parts) if class_parts else S.One
            cls = traction_simplify(cls)
            return rational, cls

        # Pow of traction type (e.g., w^3, 0^4): atomic class, rational = 1
        if isinstance(exponent, Pow) and isinstance(exponent.base, (Zero, Omega)):
            return S.One, exponent

        # Anything else (symbols, complex expressions): no factoring
        return S.One, exponent

    def _eval_one_layer(self, exponent):
        """
        Evaluate 0^exponent one layer: apply traction identities.
        If they fire, return the scalar result.
        If not (zero-class stays), return S.Zero.
        """
        exponent = traction_simplify(exponent)

        # Try constructing Pow(Zero(), exponent) — _eval_power will fire if possible
        try:
            result = Pow(Zero(), exponent)
        except Exception:
            return S.Zero

        result = traction_simplify(result)

        # If result is no longer a zero-power, the identity fired — return the result
        if not (isinstance(result, Pow) and isinstance(result.base, Zero)):
            if isinstance(result, Zero):
                return S.Zero  # bare Zero = zero-class
            return result  # scalar (the identity result)

        # Still a zero-power: zero-class, magnitude 0
        return S.Zero


def _project_for_eval(expr):
    """Project a traction expression for numeric evaluation (ω → W, etc.)."""
    if expr is None:
        return S.Zero
    expr_w = expr.subs(Omega(), OMEGA_EXP_VAL) if hasattr(expr, 'has') and expr.has(Omega) else expr
    return _project(expr_w)


# Auto-register on import
_instance = QSurfaceProjection(q=2)
_instance.register_self()
