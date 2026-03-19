"""
Numeric Traction Evaluator
===========================
Evaluates traction expressions numerically while preserving the
zero/omega class distinction that numpy loses.

A TractionValue carries a complex number decomposed as:
    scalar + zero_component + omega_component

where the zero and omega parts track their exponents separately,
so that 0^(p*ω) projects correctly as e^{iπp} instead of e^{-W*∞}.
"""

import cmath
import math
import numpy as np
from sympy import (
    Symbol, Integer, Rational, Number, Pow, Mul, Add,
    S, I, pi, symbols, lambdify
)

# Precompute Chebyshev evaluation constants
from traction import W_CONST, CHEB_THETA, OMEGA_EXP_VAL, Zero, Omega, Null, Log0, LogW
import math as _math
_THETA = _math.pi / 4  # Chebyshev angle (matches CHEB_THETA default)
_OMEGA_VAL = _math.pi / _THETA  # omega -> pi/theta in exponents


class TractionExponent:
    """
    An exponent in the traction domain: real_part + omega_part * ω.

    The omega_part determines the phase-rotation behavior:
        0^(a + b*ω) → e^{-W*a} * e^{iπb}
    """
    __slots__ = ['real', 'omega']

    def __init__(self, real=0.0, omega=0.0):
        self.real = complex(real)
        self.omega = complex(omega)

    def project(self):
        """Project to complex via Chebyshev: e^(i*THETA*(real + omega*pi/THETA))."""
        return self.real + self.omega * _OMEGA_VAL

    def __add__(self, other):
        if isinstance(other, TractionExponent):
            return TractionExponent(self.real + other.real, self.omega + other.omega)
        return TractionExponent(self.real + complex(other), self.omega)

    def __radd__(self, other):
        return TractionExponent(complex(other) + self.real, self.omega)

    def __mul__(self, other):
        if isinstance(other, TractionExponent):
            # (a + bω)(c + dω) = ac + (ad+bc)ω + bdω²
            # ω² = W² = -iπ, so bdω² = bd(-iπ) which is a real contribution
            bd_w2 = self.omega * other.omega * (-1j * cmath.pi)
            return TractionExponent(
                self.real * other.real + bd_w2,
                self.real * other.omega + self.omega * other.real
            )
        c = complex(other)
        return TractionExponent(self.real * c, self.omega * c)

    def __rmul__(self, other):
        c = complex(other)
        return TractionExponent(c * self.real, c * self.omega)

    def __neg__(self):
        return TractionExponent(-self.real, -self.omega)

    def __repr__(self):
        return f'TExp({self.real} + {self.omega}ω)'


class TV:
    """
    Traction numeric Value.

    Represents a value in one of three forms:
        SCALAR:     a plain complex number
        ZERO_POWER: 0^exponent, where exponent is a TractionExponent
        OMEGA_VAL:  scalar * ω (result of division by zero)

    The project() method converts to a complex number for rendering.
    The as_exponent() method returns a TractionExponent for use in 0^self.
    """
    SCALAR = 'scalar'
    ZERO_POWER = 'zero_power'
    OMEGA_VAL = 'omega_val'

    __slots__ = ['val', 'kind', 'exponent', 'omega_coeff']

    def __init__(self, val=0.0, kind='scalar', exponent=None, omega_coeff=0.0):
        self.val = complex(val) if val is not None else 0.0
        self.kind = kind
        self.exponent = exponent       # TractionExponent, for ZERO_POWER
        self.omega_coeff = complex(omega_coeff)  # coefficient of ω, for OMEGA_VAL

    @staticmethod
    def scalar(val):
        return TV(val, TV.SCALAR)

    @staticmethod
    def zero():
        """Traction zero = 0^1."""
        return TV(0, TV.ZERO_POWER, TractionExponent(1.0, 0.0))

    @staticmethod
    def omega():
        """Traction omega = 0^(-1)."""
        return TV(0, TV.ZERO_POWER, TractionExponent(-1.0, 0.0))

    @staticmethod
    def zero_pow(exp):
        """0^exp where exp is a TractionExponent."""
        return TV(0, TV.ZERO_POWER, exp)

    @staticmethod
    def omega_scaled(coeff):
        """coeff * ω — result of division by zero."""
        return TV(0, TV.OMEGA_VAL, omega_coeff=coeff)

    def is_zero_class(self):
        return self.kind == TV.ZERO_POWER

    def as_exponent(self):
        """Convert this value to a TractionExponent (for use as an exponent in 0^self)."""
        if self.kind == TV.SCALAR:
            return TractionExponent(self.val, 0.0)
        elif self.kind == TV.ZERO_POWER:
            # 0^exp used as an exponent: the value IS the exponent
            return self.exponent
        elif self.kind == TV.OMEGA_VAL:
            # coeff*ω as an exponent: pure omega component
            return TractionExponent(0.0, self.omega_coeff)
        return TractionExponent(self.val, 0.0)

    def project(self):
        """Project to a complex number for rendering."""
        if self.kind == TV.SCALAR:
            return self.val
        elif self.kind == TV.ZERO_POWER:
            # 0^(a + bω) → e^(i*THETA*(a + b*pi/THETA)) = e^(i*THETA*a + i*pi*b)
            exp = self.exponent
            total_exp = complex(exp.real) + complex(exp.omega) * _OMEGA_VAL
            return cmath.exp(1j * _THETA * total_exp)
        elif self.kind == TV.OMEGA_VAL:
            # coeff*ω → coeff * e^(-i*THETA) (Chebyshev projection of ω)
            if abs(self.omega_coeff) < 1e-300:
                return 0.0
            return self.omega_coeff * cmath.exp(-1j * _THETA)
        return self.val

    def __repr__(self):
        if self.kind == TV.SCALAR:
            return f'TV({self.val})'
        elif self.kind == TV.ZERO_POWER:
            return f'TV(0^{self.exponent})'
        elif self.kind == TV.OMEGA_VAL:
            return f'TV({self.omega_coeff}*ω)'
        return f'TV(?)'


# ============================================================
# Arithmetic Operations
# ============================================================

def tv_add(a, b):
    """a + b"""
    # For now, project both and add as scalars
    # TODO: track structure through addition (e.g. sum of zero-powers)
    return TV.scalar(a.project() + b.project())


def tv_sub(a, b):
    """a - b"""
    return TV.scalar(a.project() - b.project())


def tv_mul(a, b):
    """a * b"""
    # Zero-power * Zero-power: exponents add (0^a * 0^b = 0^(a+b))
    if a.kind == TV.ZERO_POWER and b.kind == TV.ZERO_POWER:
        return TV.zero_pow(a.exponent + b.exponent)

    # Scalar * Zero-power (or vice versa)
    if a.kind == TV.SCALAR and b.kind == TV.ZERO_POWER:
        # Detect omega-class (exp has negative real, no omega): scalar * ω^n
        if abs(b.exponent.omega) < 1e-15 and b.exponent.real.real < -0.5:
            return TV.omega_scaled(a.val)  # n * ω → omega_scaled(n)
        return TV.scalar(a.val * b.project())
    if a.kind == TV.ZERO_POWER and b.kind == TV.SCALAR:
        if abs(a.exponent.omega) < 1e-15 and a.exponent.real.real < -0.5:
            return TV.omega_scaled(b.val)
        return TV.scalar(a.project() * b.val)

    # Scalar * Omega_val (or vice versa)
    if a.kind == TV.SCALAR and b.kind == TV.OMEGA_VAL:
        return TV.omega_scaled(a.val * b.omega_coeff)
    if a.kind == TV.OMEGA_VAL and b.kind == TV.SCALAR:
        return TV.omega_scaled(a.omega_coeff * b.val)

    # Zero-power * Omega_val: 0^exp * coeff*ω = coeff * 0^(exp-1)
    if a.kind == TV.ZERO_POWER and b.kind == TV.OMEGA_VAL:
        return TV.scalar(b.omega_coeff * TV.zero_pow(a.exponent + TractionExponent(-1.0, 0.0)).project())
    if a.kind == TV.OMEGA_VAL and b.kind == TV.ZERO_POWER:
        return TV.scalar(a.omega_coeff * TV.zero_pow(b.exponent + TractionExponent(-1.0, 0.0)).project())

    # Omega_val * Omega_val: (a*ω)(b*ω) = ab*ω² = ab*0^(-2)
    if a.kind == TV.OMEGA_VAL and b.kind == TV.OMEGA_VAL:
        return TV.scalar(a.omega_coeff * b.omega_coeff * TV.zero_pow(TractionExponent(-2.0, 0.0)).project())

    # Scalar * Scalar
    return TV.scalar(a.val * b.val)


def tv_div(a, b):
    """a / b — division by zero produces omega-class values."""
    if b.kind == TV.SCALAR and abs(b.val) < 1e-300:
        # Division by zero: a / 0 = a * ω
        if a.kind == TV.SCALAR:
            return TV.omega_scaled(a.val)
        elif a.kind == TV.ZERO_POWER:
            # 0^exp / 0 = 0^exp * 0^(-1) = 0^(exp-1)
            return TV.zero_pow(a.exponent + TractionExponent(-1.0, 0.0))
        elif a.kind == TV.OMEGA_VAL:
            # (coeff*ω) / 0 = coeff * ω * ω = coeff * 0^(-2)
            return TV.zero_pow(TractionExponent(-2.0, 0.0))

    if b.kind == TV.ZERO_POWER:
        # a / 0^exp = a * 0^(-exp)
        neg_exp = TractionExponent(-b.exponent.real, -b.exponent.omega)
        if a.kind == TV.ZERO_POWER:
            return TV.zero_pow(a.exponent + neg_exp)
        elif a.kind == TV.SCALAR:
            # scalar * 0^(-exp): project and multiply
            return TV.scalar(a.val * TV.zero_pow(neg_exp).project())
        elif a.kind == TV.OMEGA_VAL:
            # coeff*ω * 0^(-exp) = coeff * 0^(-1) * 0^(-exp) = coeff * 0^(-1-exp)
            combined = TractionExponent(-1.0, 0.0) + neg_exp
            return TV.scalar(a.omega_coeff * TV.zero_pow(combined).project())

    if b.kind == TV.OMEGA_VAL:
        # a / (coeff*ω) = a * 0 / coeff (since 1/ω = 0)
        if abs(b.omega_coeff) < 1e-300:
            return TV.scalar(complex('nan'))
        return tv_mul(a, TV.scalar(1.0 / b.omega_coeff)) if a.kind != TV.SCALAR else TV.scalar(a.val / b.omega_coeff * TV.zero().project())

    # Normal division
    bp = b.project()
    if abs(bp) < 1e-300:
        return TV.scalar(complex('nan'))
    return TV.scalar(a.project() / bp)


def tv_pow(base, exp):
    """base ^ exp"""
    # 0^exp: the core case — base is bare Zero (0^1)
    if base.kind == TV.ZERO_POWER and abs(base.exponent.real - 1.0) < 1e-15 and abs(base.exponent.omega) < 1e-15:
        # 0^(0^something) — generalized identity: 0^(0^x) = x
        if exp.kind == TV.ZERO_POWER:
            return TV.scalar(exp.exponent.project())
        # 0^exp for any exp: use as_exponent to get the TractionExponent
        return TV.zero_pow(exp.as_exponent())

    # General zero-power base: (0^a)^b = 0^(a*b)
    if base.kind == TV.ZERO_POWER:
        exp_te = exp.as_exponent()
        new_exp = base.exponent * exp_te
        return TV.zero_pow(new_exp)

    # Scalar base near zero: treat as traction zero for exponentiation
    if base.kind == TV.SCALAR and abs(base.val) < 1e-300:
        return tv_pow(TV.zero(), exp)

    # Scalar base, scalar exp
    if base.kind == TV.SCALAR and exp.kind == TV.SCALAR:
        try:
            # Handle negative real bases: (-a)^n = a^n * e^{iπn}
            if base.val.real < 0 and base.val.imag == 0:
                a = abs(base.val)
                n = exp.val
                return TV.scalar(a ** n * cmath.exp(1j * cmath.pi * n))
            return TV.scalar(base.val ** exp.val)
        except (ValueError, OverflowError, ZeroDivisionError):
            return TV.scalar(complex('nan'))

    # Scalar base, zero-power exp: project exp and compute
    if base.kind == TV.SCALAR and exp.kind == TV.ZERO_POWER:
        pe = exp.project()
        try:
            if base.val.real < 0 and base.val.imag == 0:
                a = abs(base.val)
                return TV.scalar(a ** pe * cmath.exp(1j * cmath.pi * pe))
            return TV.scalar(base.val ** pe)
        except (ValueError, OverflowError, ZeroDivisionError):
            return TV.scalar(complex('nan'))

    return TV.scalar(complex('nan'))


def tv_neg(a):
    """Unary negation."""
    if a.kind == TV.SCALAR:
        return TV.scalar(-a.val)
    # -0 = 0 (negative zero doesn't exist)
    if a.kind == TV.ZERO_POWER:
        return a
    return TV.scalar(-a.project())


# ============================================================
# Expression Evaluator
# ============================================================

def traction_eval(expr, env):
    """
    Evaluate a SymPy traction expression numerically using TractionValue arithmetic.

    Args:
        expr: A SymPy expression (may contain Zero, Omega, symbols, etc.)
        env: dict mapping symbol names to TV values, e.g. {'p': TV.scalar(1.5), 'q': TV.scalar(-0.7)}

    Returns:
        A TV (TractionValue)
    """
    # Atoms
    if isinstance(expr, Zero):
        return TV.zero()

    if isinstance(expr, Omega):
        return TV.omega()

    if isinstance(expr, Null):
        return TV.scalar(0.0)

    if isinstance(expr, Symbol):
        name = expr.name
        if name in env:
            return env[name]
        return TV.scalar(complex('nan'))

    if isinstance(expr, Integer):
        return TV.scalar(complex(int(expr)))

    if isinstance(expr, Rational):
        return TV.scalar(complex(float(expr)))

    if isinstance(expr, Number):
        return TV.scalar(complex(expr))

    # Pow
    if isinstance(expr, Pow):
        base = traction_eval(expr.base, env)
        exp = traction_eval(expr.exp, env)
        return tv_pow(base, exp)

    # Mul
    if isinstance(expr, Mul):
        result = TV.scalar(1.0)
        for arg in expr.args:
            result = tv_mul(result, traction_eval(arg, env))
        return result

    # Add
    if isinstance(expr, Add):
        result = TV.scalar(0.0)
        for arg in expr.args:
            result = tv_add(result, traction_eval(arg, env))
        return result

    # Log0, LogW — extensibility stubs
    if isinstance(expr, Log0):
        inner = traction_eval(expr.args[0], env)
        # log_0(y) = -ln(y) / W
        try:
            proj = inner.project()
            if abs(proj) < 1e-300:
                return TV.scalar(complex('nan'))
            return TV.scalar(cmath.log(proj) / (1j * _THETA))
        except (ValueError, ZeroDivisionError):
            return TV.scalar(complex('nan'))

    if isinstance(expr, LogW):
        inner = traction_eval(expr.args[0], env)
        # log_w(y) = ln(y) / W
        try:
            proj = inner.project()
            if abs(proj) < 1e-300:
                return TV.scalar(complex('nan'))
            return TV.scalar(-cmath.log(proj) / (1j * _THETA))
        except (ValueError, ZeroDivisionError):
            return TV.scalar(complex('nan'))

    # Fallback: try to evaluate as a number
    try:
        return TV.scalar(complex(expr.evalf()))
    except Exception:
        return TV.scalar(complex('nan'))
