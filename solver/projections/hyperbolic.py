"""
Hyperbolic Projection — Algebraic Geometric Products
=====================================================
Purely algebraic visualization: no exp, log, sin, cos, atan2.

Polynomials expand naturally into real/imaginary parts:
    z^2 = (x^2 - y^2) + i(2xy)
    z^3 = (x^3 - 3xy^2) + i(3x^2y - y^3)

Color derives from primitive operations on p = (x,y) and f(p) = (u,v):
    dot(p, f(p))   = x*u + y*v      — alignment / radial stretch
    cross(p, f(p)) = x*v - y*u      — rotation / angular twist
    |p|^2          = x^2 + y^2      — input norm
    |f(p)|^2       = u^2 + v^2      — output norm

Phase from the diamond angle of (dot, cross) — L1-norm 4-quadrant proxy.
Brightness from |f|^2 / (|f|^2 + |p|^2) — algebraic sigmoid.

For zero-powers (0^z), the Cayley rational transform replaces the
transcendental exponential:
    0^z  ->  (1 + z) / (1 - z)     — conformal, rational, unit-circle preserving
"""

import numpy as np
from sympy import (
    lambdify, S, Integer, Rational, Number, Symbol,
    Pow, Mul, Add, I,
)
from traction import (
    Zero, Omega, Null, Log0, LogW,
    traction_simplify, project_complex,
)
from projections.base import Projection


# ── Cayley projection engine ────────────────────────────────────
# Replaces exp with the rational Cayley transform:
#   0^z → (1 + z_proj) / (1 - z_proj)
# Preserves conformality, stays on the unit circle for real inputs,
# and is purely rational (polynomial / polynomial).

# In the Cayley frame, ω → 2i so that 0^(ω/2) → C(i) = i exactly.
# (Solving (1+t)/(1-t) = i gives t = i, so ω/2 must map to i.)
_OMEGA_CAYLEY = 2 * I


def _project_cayley(expr):
    """Project a traction expression via the Cayley rational transform."""
    if isinstance(expr, (Integer, Rational, Number)):
        return expr
    if isinstance(expr, Zero):
        # 0 = 0^1: Cayley has a pole at z=1 → maps to infinity.
        # In multiplicative context (e.g. x*0), this is the traction zero.
        # Return a symbolic infinity that lambdify handles as inf.
        return S.ComplexInfinity
    if isinstance(expr, Omega):
        # ω = 0^(-1): C(-1) = 0. Omega maps to zero — the reciprocal pole.
        return S.Zero
    if isinstance(expr, Null):
        return S.Zero
    if isinstance(expr, Symbol):
        return expr

    if isinstance(expr, Pow):
        return _project_cayley_pow(expr)

    if isinstance(expr, Mul):
        result = S.One
        for arg in expr.args:
            result = result * _project_cayley(arg)
        return result

    if isinstance(expr, Add):
        result = S.Zero
        for arg in expr.args:
            result = result + _project_cayley(arg)
        return result

    # Logarithms: fall back to Lie (these are inherently transcendental)
    if isinstance(expr, (Log0, LogW)):
        return project_complex(expr)

    return expr


def _project_cayley_pow(expr):
    """Project a Pow expression via the Cayley transform."""
    base = expr.base
    exponent = expr.exp

    # Non-traction base: project base and exponent separately
    if not isinstance(base, (Zero, Omega)):
        if isinstance(base, Mul) and (base.has(Zero) or base.has(Omega)):
            result = S.One
            for factor in Mul.make_args(base):
                result = result * _project_cayley(Pow(factor, exponent))
            return result
        pb = _project_cayley(base)
        pe = _project_cayley(exponent)
        return Pow(pb, pe)

    # Convert omega base: ω^a = 0^(-a)
    if isinstance(base, Omega):
        exponent = -exponent

    exponent = traction_simplify(exponent)

    # Identities that hold exactly
    if exponent == S.Zero or isinstance(exponent, Zero):
        return S.One           # 0^0 = 1
    if isinstance(exponent, Omega):
        return S.NegativeOne   # 0^ω = -1

    # Substitute ω → i in exponents, then project recursively
    exp_sub = exponent.subs(Omega(), _OMEGA_CAYLEY) if exponent.has(Omega) else exponent
    z = _project_cayley(exp_sub)

    # Cayley transform: 0^z → (1 + z) / (1 - z)
    return (S.One + z) / (S.One - z)


# ── Algebraic phase and brightness ──────────────────────────────

def _diamond_phase(dot, cross):
    """Algebraic 4-quadrant phase in [0, 2π) via L1-norm diamond mapping.

    Maps the (dot, cross) plane to [0, 2π) using only abs, +, /.
    Piecewise linear — straight quadrant boundaries instead of curved.
    """
    abs_d = np.abs(dot)
    abs_c = np.abs(cross)
    r = abs_d + abs_c + 1e-15

    # Diamond angle in [0, 4), then scale to [0, 2π)
    angle = np.where(
        dot >= 0,
        np.where(cross >= 0, cross / r, 4 - abs_c / r),
        np.where(cross >= 0, 1 + abs_d / r, 2 + abs_c / r),
    )
    return angle * (np.pi / 2)


# ── Projection class ────────────────────────────────────────────

class HyperbolicProjection(Projection):
    name = 'hyperbolic'
    description = 'Algebraic projection via geometric products (dot/cross/wedge)'
    output_keys = ['Re', 'Im', 'mag', 'phase', 'dot', 'cross']

    def format_projection(self, traction_expr):
        """Show the polynomial expansion as (u, v) components."""
        from calculator import format_approx
        result = format_approx(traction_expr)
        return result if result else ''

    def native_x(self, a, b):
        """x = p + q*0^(w/2) — same complex grid as Lie."""
        return a + b * Pow(Zero(), Mul(Omega(), Rational(1, 2)))

    def project_expr(self, traction_expr, a, b):
        """Project traction expression to an algebraic (rational) form.

        Polynomials pass through as-is (pure arithmetic on a + ib).
        Zero-powers use the Cayley transform: 0^z → (1+z)/(1-z).
        """
        projected = _project_cayley(traction_expr)

        try:
            projected = projected.expand()
        except Exception:
            pass
        return projected

    def eval_grid(self, projected_expr, a, b, AA, BB, **kwargs):
        """Evaluate and color via algebraic geometric products.

        All operations are polynomial/rational — no transcendentals.
        """
        try:
            f = lambdify((a, b), projected_expr, modules=['numpy'])
        except Exception:
            return None

        try:
            AA_c = AA.astype(complex)
            BB_c = BB.astype(complex)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                Z = f(AA_c, BB_c)
                if np.isscalar(Z) or (isinstance(Z, np.ndarray) and Z.ndim == 0):
                    Z = np.full_like(AA, complex(Z), dtype=complex)
                Z = np.asarray(Z, dtype=complex)
        except Exception:
            return None

        U = Z.real
        V = Z.imag

        # ── Geometric products (all algebraic) ──────────────────
        dot = AA * U + BB * V        # p · f(p)  — alignment
        cross = AA * V - BB * U      # p × f(p)  — rotation
        norm_p = AA ** 2 + BB ** 2   # |p|²
        norm_f = U ** 2 + V ** 2     # |f(p)|²

        # ── Phase: diamond angle of (dot, cross) ───────────────
        phase = _diamond_phase(dot, cross)

        # ── Brightness: algebraic sigmoid from norms ───────────
        brightness = norm_f / (norm_f + norm_p + 1e-15)
        brightness = np.clip(brightness, 0.12, 0.95)

        # ── Invalid handling ───────────────────────────────────
        invalid = ~np.isfinite(Z) | ((norm_f < 1e-30) & (norm_p < 1e-30))
        phase[invalid] = np.nan
        brightness[invalid] = 0.0

        mag = np.sqrt(norm_f)
        log_mag = np.log(np.maximum(norm_f, 1e-300)) / 2

        return {
            'Z': Z,
            'Re': U,
            'Im': V,
            'mag': mag,
            'log_mag': log_mag,
            'phase': phase,
            'brightness': brightness,
            'dot': dot,
            'cross': cross,
        }


# Auto-register on import
_instance = HyperbolicProjection()
_instance.register_self()
