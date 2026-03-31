"""
Geometric Algebra Projection
==============================
2D geometric algebra over traction expressions.

The geometric product of two vectors a, b decomposes as:
    ab = a·b + a∧b
       scalar   bivector
       (grade 0) (grade 2)

In 2D the bivector I = e₁e₂ is the oriented unit area element.
Its square determines the algebra's character:

    Circular  (Lie):  e₁² = e₂² = +1  →  I² = -1  →  rotation (phase)
    Hyperbolic (GA):  e₁² = +1, e₂² = -1  →  I² = +1  →  boost (magnitude)

The exponential of a bivector:
    I² = -1:  e^{Iθ} = cos θ + I sin θ     (circular rotation)
    I² = +1:  e^{Iφ} = cosh φ + I sinh φ   (hyperbolic boost)

Same structure, different metric signature → completely different behavior.
Complex number a + bi becomes multivector a + bI: the real part is a scalar,
the imaginary part is an oriented area element.

Coloring uses the geometric product decomposition of p and f(p):
    p · f(p)  = x·u + y·v    — scalar part (dot): radial stretch
    p ∧ f(p)  = x·v − y·u    — bivector part (wedge): oriented rotation
    |p|²      = x² + y²      — input norm
    |f(p)|²   = u² + v²      — output norm

Phase from the diamond angle of (dot, wedge) — algebraic 4-quadrant proxy.
Brightness from |f|² / (|f|² + |p|²) — algebraic sigmoid.

Signature ratio: wedge²/(dot²+wedge²) measures how much of the
transformation is circular rotation (I²=-1) vs hyperbolic boost (I²=+1)
at each point. Traction naturally contains both: 0^(ω/2) squares to -1
(circular) while 0^0 squares to +1 (hyperbolic), so both signatures
coexist in the same algebra. The 'mixed' color mode visualizes this
with cool colors for rotation-dominated regions and warm colors for
boost-dominated regions.

For zero-powers (0^z), the Cayley rational transform replaces the
transcendental exponential:
    0^z  →  (1 + z) / (1 − z)    — conformal, rational, unit-circle preserving
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

    # Substitute ω → 2i in exponents, then project recursively
    exp_sub = exponent.subs(Omega(), _OMEGA_CAYLEY) if exponent.has(Omega) else exponent
    z = _project_cayley(exp_sub)

    # Cayley transform: 0^z → (1 + z) / (1 - z)
    return (S.One + z) / (S.One - z)


# ── Algebraic phase and brightness ──────────────────────────────

def _diamond_phase(dot, wedge):
    """Algebraic 4-quadrant phase in [0, 2π) via L1-norm diamond mapping.

    The geometric product of p and f(p) decomposes into:
        scalar part (dot):   p · f(p)
        bivector part (wedge): p ∧ f(p)

    Maps (dot, wedge) to phase using only abs, +, /.
    Piecewise linear — straight quadrant boundaries instead of curved.
    """
    abs_d = np.abs(dot)
    abs_w = np.abs(wedge)
    r = abs_d + abs_w + 1e-15

    # Diamond angle in [0, 4), then scale to [0, 2π)
    angle = np.where(
        dot >= 0,
        np.where(wedge >= 0, wedge / r, 4 - abs_w / r),
        np.where(wedge >= 0, 1 + abs_d / r, 2 + abs_w / r),
    )
    return angle * (np.pi / 2)


# ── Projection class ────────────────────────────────────────────

class GeometricAlgebraProjection(Projection):
    name = 'geometric_algebra'
    description = 'Geometric algebra: dot + wedge product, Cayley transform'
    output_keys = ['Re', 'Im', 'mag', 'phase', 'dot', 'wedge']

    def format_projection(self, traction_expr):
        """Show the polynomial expansion as (u, v) components."""
        from calculator import format_approx
        result = format_approx(traction_expr)
        return result if result else ''

    def native_x(self, a, b):
        """x = p + q*0^(w/2) — multivector a + bI in the e₁, e₂ basis."""
        return a + b * Pow(Zero(), Mul(Omega(), Rational(1, 2)))

    def project_expr(self, traction_expr, a, b):
        """Project traction expression to an algebraic (rational) form.

        Polynomials pass through as-is (pure arithmetic on a + bI).
        Zero-powers use the Cayley transform: 0^z → (1+z)/(1-z).
        """
        projected = _project_cayley(traction_expr)

        try:
            projected = projected.expand()
        except Exception:
            pass
        return projected

    def eval_grid(self, projected_expr, a, b, AA, BB, **kwargs):
        """Evaluate and extract geometric product components for coloring.

        The geometric product of input p and output f(p) decomposes as:
            pf(p) = p·f(p) + p∧f(p)
                    scalar    bivector

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

        # ── Geometric product decomposition ────────────────────
        # For input p = (x,y) and output f(p) = (u,v):
        #   pf(p) = p·f(p) + p∧f(p)
        dot = AA * U + BB * V        # scalar part: radial stretch
        wedge = AA * V - BB * U      # bivector part: oriented rotation
        norm_p = AA ** 2 + BB ** 2   # |p|²
        norm_f = U ** 2 + V ** 2     # |f(p)|²

        # ── Phase: diamond angle of (dot, wedge) ──────────────
        phase = _diamond_phase(dot, wedge)

        # ── Brightness: algebraic sigmoid from norms ──────────
        brightness = norm_f / (norm_f + norm_p + 1e-15)
        brightness = np.clip(brightness, 0.12, 0.95)

        # ── Invalid handling ──────────────────────────────────
        invalid = ~np.isfinite(Z) | ((norm_f < 1e-30) & (norm_p < 1e-30))
        phase[invalid] = np.nan
        brightness[invalid] = 0.0

        mag = np.sqrt(norm_f)
        log_mag = np.log(np.maximum(norm_f, 1e-300)) / 2

        # ── Signature ratio ──────────────────────────────────────
        # How much of the transformation is circular (rotation) vs
        # hyperbolic (boost) at each point.
        #   wedge² → circular energy (bivector, I² = -1 rotation)
        #   dot²   → hyperbolic energy (scalar, I² = +1 boost)
        circ = wedge ** 2
        hyp = dot ** 2
        total = circ + hyp + 1e-15
        sig_ratio = circ / total  # 1 = pure circular, 0 = pure hyperbolic

        return {
            'Z': Z,
            'Re': U,
            'Im': V,
            'mag': mag,
            'log_mag': log_mag,
            'phase': phase,
            'brightness': brightness,
            'dot': dot,
            'wedge': wedge,
            'sig_ratio': sig_ratio,
        }


# Auto-register on import
_instance = GeometricAlgebraProjection()
_instance.register_self()
