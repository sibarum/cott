"""
Tetrad Projection
=================
Decomposes traction expressions onto the four-axis basis {1, ε, k, i},
where each basis unit is a half-power of zero:

    1  := 0^(0/2)    →  1²  = 0^0    = +1   (scalar)
    ε  := 0^(1/2)    →  ε²  = 0^1    =  0   (parabolic)
    k  := 0^(-1/2)   →  k²  = 0^(-1) =  ω   (omega-imaginary)
    i  := 0^(ω/2)    →  i²  = 0^ω    = −1   (elliptic)

The four traction atoms {1, 0, ω, −1} are the squares of the four basis units.

Decomposition rule: every exponent z splits as z = α + βω with α, β ∈ R, then

    0^(α + βω) = 0^α · (−1)^β

Real-exponent factor (linearized dual / omega form):
    α ≥ 0:  0^α  =  1 + αε
    α ≤ 0:  0^α  =  1 + |α|·k

Omega-exponent factor (elliptic rotation):
    (−1)^β  =  cos(πβ) + i·sin(πβ)

ω^x is rewritten as 0^(−x) before decomposition.

Cross-term snap (option A): εi → i, ki → i. This keeps the basis 4-channel.
Algebraic-closure option C (extending to 6 axes including εi, ki) is a
future upgrade.

The output is four real numpy arrays (c1, cε, ck, ci) plus the standard
phase/brightness/mag for fallback rendering with existing colorizers.
"""

import numpy as np
from sympy import (
    S, Integer, Rational, Number, Symbol, Pow, Mul, Add, lambdify,
)
from traction import (
    Zero, Omega, Null, Log0, LogW, traction_simplify, project_complex,
)
from projections.base import Projection


# ── Tetrad numpy container ─────────────────────────────────────

class Tetrad:
    """4-vector on the {1, ε, k, i} basis with numpy-array components.

    Implements addition (componentwise) and multiplication (via the
    basis multiplication table with snap rule A for cross-terms εi, ki).
    """

    __slots__ = ('c1', 'ce', 'ck', 'ci')

    def __init__(self, c1, ce, ck, ci):
        self.c1 = c1
        self.ce = ce
        self.ck = ck
        self.ci = ci

    @classmethod
    def scalar(cls, value, ref):
        """Pure 1-axis tetrad, broadcast to the shape of `ref`."""
        c1 = np.broadcast_to(np.asarray(value, dtype=float), ref.shape).copy()
        z = np.zeros_like(ref, dtype=float)
        return cls(c1, z.copy(), z.copy(), z.copy())

    @classmethod
    def zeros_like(cls, ref):
        z = np.zeros_like(ref, dtype=float)
        return cls(z.copy(), z.copy(), z.copy(), z.copy())

    def copy(self):
        return Tetrad(self.c1.copy(), self.ce.copy(), self.ck.copy(), self.ci.copy())

    def __add__(self, other):
        if isinstance(other, Tetrad):
            return Tetrad(self.c1 + other.c1, self.ce + other.ce,
                          self.ck + other.ck, self.ci + other.ci)
        # Scalar add lands on 1-axis only
        return Tetrad(self.c1 + other, self.ce.copy(), self.ck.copy(), self.ci.copy())

    __radd__ = __add__

    def __neg__(self):
        return Tetrad(-self.c1, -self.ce, -self.ck, -self.ci)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tetrad) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, Tetrad):
            return Tetrad(self.c1 * other, self.ce * other,
                          self.ck * other, self.ci * other)
        a1, ae, ak, ai = self.c1, self.ce, self.ck, self.ci
        b1, be, bk, bi = other.c1, other.ce, other.ck, other.ci

        # Multiplication table:
        #   1·1 = 1
        #   1·ε = ε,   1·k = k,   1·i = i      (and symmetric)
        #   ε·k = 1                            (parabolic ↔ omega reciprocal)
        #   ε·ε = 0_traction → linearized (1+ε)
        #   k·k = ω_traction → linearized (1+k)
        #   i·i = −1
        #   ε·i = i  (snap A),  k·i = i  (snap A)

        # 1-axis: a1·b1 + ε·k pairs (= 1) + ε² and k² linearizations contribute 1 + (-1)·i²
        c1 = (a1 * b1
              + ae * be          # ε² → 1 + ε: 1-component is 1
              + ak * bk          # k² → 1 + k: 1-component is 1
              + ae * bk + ak * be   # ε·k + k·ε = 2·1
              - ai * bi)         # i² = −1

        # ε-axis: 1·ε mixings + ε² linearization contributes ε
        ce = (a1 * be + ae * b1
              + ae * be)         # ε² → 1 + ε: ε-component is 1

        # k-axis: 1·k mixings + k² linearization contributes k
        ck = (a1 * bk + ak * b1
              + ak * bk)         # k² → 1 + k: k-component is 1

        # i-axis: 1·i mixings + ε·i and k·i (snap to i)
        ci = (a1 * bi + ai * b1
              + ae * bi + ai * be     # εi + iε snap → i
              + ak * bi + ai * bk)    # ki + ik snap → i

        return Tetrad(c1, ce, ck, ci)

    __rmul__ = __mul__

    def __pow__(self, n):
        # Integer non-negative power: repeated multiplication.
        if not isinstance(n, (int, np.integer)):
            raise NotImplementedError(f"Tetrad ** {n!r} (non-integer)")
        if n < 0:
            raise NotImplementedError(f"Tetrad ** {n} (negative)")
        if n == 0:
            return Tetrad.scalar(1.0, self.c1)
        result = self
        for _ in range(n - 1):
            result = result * self
        return result


# ── Exponent decomposition  z = α + βω  ───────────────────────

def _split_omega(expr):
    """Decompose expr = α + βω with α, β having no Omega.

    Falls back to (expr, 0) if the expression isn't linear in ω.
    """
    if isinstance(expr, Omega):
        return S.Zero, S.One
    if isinstance(expr, Add):
        alpha = []
        beta = []
        for arg in expr.args:
            a, b = _split_omega(arg)
            alpha.append(a)
            beta.append(b)
        return Add(*alpha), Add(*beta)
    if isinstance(expr, Mul):
        omega_count = 0
        others = []
        for f in Mul.make_args(expr):
            if isinstance(f, Omega):
                omega_count += 1
            else:
                others.append(f)
        rest = Mul(*others) if others else S.One
        if omega_count == 0:
            return expr, S.Zero
        if omega_count == 1:
            return S.Zero, rest
        # ω² and higher: linearize using ω² = (−1)·... actually fold into α
        # by treating residual omega as -1 (since 0^ω = -1, ω lives "as -1").
        # Higher powers of ω in *exponents* are unusual; punt to α.
        return expr, S.Zero
    return expr, S.Zero


# ── Pow 0^z evaluator  ──────────────────────────────────────────

def _pow_zero_tetrad(alpha, beta):
    """Evaluate 0^(α + βω) on a numpy grid, returning a Tetrad.

    α, β are real numpy arrays (one per grid point).
    """
    # Real-exponent factor R(α) on span{1, ε, k}:
    #   R = 1 + max(α,0)·ε + max(−α,0)·k
    alpha_pos = np.maximum(alpha, 0.0)
    alpha_neg = np.maximum(-alpha, 0.0)

    # Omega-exponent factor Ω(β) on span{1, i}:
    #   Ω = cos(πβ) + sin(πβ)·i
    cos_b = np.cos(np.pi * beta)
    sin_b = np.sin(np.pi * beta)

    # Multiply R · Ω with snap rule (εi→i, ki→i):
    #   (1 + α₊ε + α₋k)·(cos + sin·i)
    #   = cos + α₊·cos·ε + α₋·cos·k + sin·i + α₊·sin·(εi) + α₋·sin·(ki)
    #   snap →  c1 = cos
    #          ce = α₊·cos
    #          ck = α₋·cos
    #          ci = sin·(1 + α₊ + α₋)
    c1 = cos_b
    ce = alpha_pos * cos_b
    ck = alpha_neg * cos_b
    ci = sin_b * (1.0 + alpha_pos + alpha_neg)
    return Tetrad(c1, ce, ck, ci)


# ── Walker: SymPy traction expression  →  Tetrad on grid ──────

def _walk(expr, a, b, AA, BB):
    """Recursively evaluate a traction expression as a Tetrad over the grid."""
    # Atoms
    if isinstance(expr, (Integer, Rational, Number)):
        return Tetrad.scalar(float(expr), AA)
    if isinstance(expr, Symbol):
        if expr == a:
            return Tetrad.scalar(AA.astype(float), AA)
        if expr == b:
            return Tetrad.scalar(BB.astype(float), AA)
        # Unknown symbol — bail to scalar with NaN
        return Tetrad.scalar(np.nan, AA)
    if isinstance(expr, Null):
        return Tetrad.zeros_like(AA)
    if isinstance(expr, Zero):
        # 0 = 0^1 → α=1, β=0 → linearized: (1, 1, 0, 0)
        ones = np.ones_like(AA, dtype=float)
        zeros = np.zeros_like(AA, dtype=float)
        return _pow_zero_tetrad(ones, zeros)
    if isinstance(expr, Omega):
        # ω = 0^(−1) → α=−1, β=0 → linearized: (1, 0, 1, 0)
        ones = np.ones_like(AA, dtype=float)
        zeros = np.zeros_like(AA, dtype=float)
        return _pow_zero_tetrad(-ones, zeros)
    # Logarithms: punt to Lie projection (transcendental); embed result on i-axis.
    if isinstance(expr, (Log0, LogW)):
        return _lie_fallback(expr, a, b, AA, BB)
    if isinstance(expr, Pow):
        return _walk_pow(expr, a, b, AA, BB)
    if isinstance(expr, Add):
        result = Tetrad.zeros_like(AA)
        for arg in expr.args:
            result = result + _walk(arg, a, b, AA, BB)
        return result
    if isinstance(expr, Mul):
        result = Tetrad.scalar(1.0, AA)
        for arg in expr.args:
            result = result * _walk(arg, a, b, AA, BB)
        return result
    # Unknown node — try Lie fallback
    return _lie_fallback(expr, a, b, AA, BB)


def _walk_pow(expr, a, b, AA, BB):
    base = expr.base
    exponent = expr.exp

    # ω^x = 0^(−x): rewrite locally (don't reconstruct Pow — SymPy will canonicalize
    # 0^(−x) right back to ω^x and infinite-loop us).
    if isinstance(base, Omega):
        exponent = -exponent

    # 0^z: split z = α + βω, evaluate
    if isinstance(base, (Zero, Omega)):
        exponent = traction_simplify(exponent)
        # Quick exits
        if exponent == S.Zero or isinstance(exponent, Zero):
            return Tetrad.scalar(1.0, AA)
        if isinstance(exponent, Omega):
            return Tetrad.scalar(-1.0, AA)
        alpha_expr, beta_expr = _split_omega(exponent)
        alpha = _eval_real_on_grid(alpha_expr, a, b, AA, BB)
        beta = _eval_real_on_grid(beta_expr, a, b, AA, BB)
        return _pow_zero_tetrad(alpha, beta)

    # Mul base with traction factors: distribute across factors first
    if isinstance(base, Mul) and (base.has(Zero) or base.has(Omega)):
        result = Tetrad.scalar(1.0, AA)
        for f in Mul.make_args(base):
            result = result * _walk_pow(Pow(f, exponent), a, b, AA, BB)
        return result

    # Non-traction base: try integer-power expansion
    if isinstance(exponent, (Integer, int)) and int(exponent) >= 0:
        base_t = _walk(base, a, b, AA, BB)
        return base_t ** int(exponent)

    # Otherwise: fall back to Lie projection for this subtree.
    return _lie_fallback(expr, a, b, AA, BB)


def _eval_real_on_grid(expr, a, b, AA, BB):
    """Lambdify a SymPy expression (no traction nodes) over the grid as floats."""
    try:
        f = lambdify((a, b), expr, modules='numpy')
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            val = f(AA.astype(float), BB.astype(float))
        if np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
            val = np.full(AA.shape, float(val), dtype=float)
        return np.asarray(val, dtype=float)
    except Exception:
        return np.zeros_like(AA, dtype=float)


def _lie_fallback(expr, a, b, AA, BB):
    """Project an unsupported subtree via the Lie projection, then embed
    the resulting complex value as (Re→1-axis, Im→i-axis) — a placeholder.
    """
    try:
        projected = project_complex(expr)
        f = lambdify((a, b), projected, modules='numpy')
        AA_c = AA.astype(complex)
        BB_c = BB.astype(complex)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Z = f(AA_c, BB_c)
        Z = np.asarray(Z, dtype=complex)
        if Z.shape != AA.shape:
            Z = np.broadcast_to(Z, AA.shape).copy()
        c1 = Z.real
        ci = Z.imag
        zeros = np.zeros_like(AA, dtype=float)
        return Tetrad(c1.astype(float), zeros.copy(), zeros.copy(), ci.astype(float))
    except Exception:
        return Tetrad.zeros_like(AA)


# ── Tetrad → standard render metrics ──────────────────────────

def _tetrad_to_metrics(T):
    """Reduce a Tetrad to phase/brightness/mag/log_mag/Z for fallback rendering.

    Z = c1 + ci·j   (the elliptic projection of the tetrad)
    mag = √(c1² + cε² + ck² + ci²)   (4-norm)
    phase from atan2(ci, c1) — the i/1 plane
    """
    c1, ce, ck, ci = T.c1, T.ce, T.ck, T.ci

    Z = c1 + 1j * ci
    norm4 = np.sqrt(c1 * c1 + ce * ce + ck * ck + ci * ci)
    log_mag = np.log(np.maximum(norm4, 1e-300))

    phase = np.arctan2(ci, c1)
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    invalid = ~np.isfinite(norm4) | (norm4 < 1e-30)
    phase[invalid] = np.nan
    log_mag[invalid] = np.nan

    brightness = 0.5 + np.arctan(log_mag) / np.pi
    brightness = np.clip(brightness, 0.12, 0.95)
    brightness[invalid] = 0.0

    return {
        'Z': Z,
        'Re': c1,
        'Im': ci,
        'mag': norm4,
        'log_mag': log_mag,
        'phase': phase,
        'brightness': brightness,
        # Tetrad-native channels — for the future colorizer:
        'c1': c1,
        'ce': ce,
        'ck': ck,
        'ci': ci,
    }


# ── Projection plugin ──────────────────────────────────────────

class TetradProjection(Projection):
    name = 'tetrad'
    description = 'Tetrad: 4-axis decomposition on basis {1, ε, k, i}'
    output_keys = ['Re', 'Im', 'mag', 'phase', 'c1', 'ce', 'ck', 'ci']

    def format_projection(self, traction_expr):
        from calculator import format_approx
        result = format_approx(traction_expr)
        return result if result else ''

    def native_x(self, a, b):
        """x = p + q·(ω/2) — direct exponent-axis coordinates.

        The other projections set x = p + q·0^(ω/2) (= p + q·i), so x lives in
        the *value* algebra on the i-axis. For tetrad we want x to live in the
        *exponent* algebra: q is a coefficient on ω/2 directly, so that
            0^x = 0^(p + q·ω/2) = 0^p · (−1)^(q/2)
        cycles cleanly through {1, i, −1, −i, ...} as q ∈ {0, 1, 2, 3, ...}.

        If a user wants a value-level i, they can type 0^(ω/2) explicitly.
        """
        return a + b * Mul(Omega(), Rational(1, 2))

    def project_expr(self, traction_expr, a, b):
        """For tetrad we keep the SymPy expression intact; the walker in
        eval_grid does the 4-vector decomposition pixelwise.
        """
        return traction_simplify(traction_expr)

    def eval_grid(self, projected_expr, a, b, AA, BB, **kwargs):
        try:
            T = _walk(projected_expr, a, b, AA, BB)
        except Exception:
            return None
        return _tetrad_to_metrics(T)


# Auto-register on import
_instance = TetradProjection()
_instance.register_self()
