"""
Traction Theory Solver
======================
SymPy-based symbolic algebra for Constructive Operational Type Theory (COTT).

Core principles:
- Totality: all operations are defined for all values
- Reversibility: no information loss

Usage:
    from traction import z, w, null, traction_simplify, log0, logw

    # z = 0 (traction zero), w = omega

    # Power rules fire automatically:
    z**0        # -> 1
    z**w        # -> -1
    z**(-1)     # -> omega
    w**(-1)     # -> 0

    # Multiplication shortcuts:
    z * w       # -> 1
    z / z       # -> 1 (via z * z^-1 = z * w = 1)

    # For complex expressions, use traction_simplify:
    traction_simplify(2 * z * 3 * w)  # -> 6
"""

from sympy import (
    Expr, S, sympify, Pow, Mul, Add,
    Integer, Rational, Symbol, Number,
    I, pi, sqrt as sp_sqrt, exp as sp_exp, simplify as sp_simplify, zoo
)

# Structure constant for 0^z = e^(-W*z), where W^2 = -i*pi
# Connects traction algebra to Lie group exponential
W_CONST = sp_sqrt(-I * pi)


# ============================================================
# Core Traction Types
# ============================================================

class Zero(Expr):
    """
    Traction zero -- not absorbing under multiplication.
    Has reciprocal omega, satisfying 0 * omega = 1.
    """
    is_commutative = True
    is_number = False
    is_zero = False
    _op_priority = 15.0

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return '0'

    def _latex(self, printer):
        return r'\mathbf{0}'

    def __repr__(self):
        return '0'

    def __str__(self):
        return '0'

    def _eval_power(self, exp):
        """
        Base-0 exponentiation rules:
            0^0     = 1         0^1     = 0
            0^(-n)  = w^n       0^w     = -1
            0^(0^x) = x         0^(w^x) = -x
        """
        if exp == S.Zero or isinstance(exp, Zero):
            return S.One
        if exp == S.One:
            return Zero()
        if isinstance(exp, Integer) and exp.is_negative:
            return Pow(Omega(), -exp)
        if isinstance(exp, Omega):
            return S.NegativeOne
        if isinstance(exp, Pow) and isinstance(exp.base, Zero):
            return exp.exp        # 0^(0^x) = x for any x
        if isinstance(exp, Pow) and isinstance(exp.base, Omega):
            return -exp.exp       # 0^(w^x) = -x for any x
        return None

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, Omega):
            return S.One
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, Omega):
            return S.One
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        return self * Pow(other, S.NegativeOne)

    def __rtruediv__(self, other):
        other = sympify(other)
        return other * Pow(self, S.NegativeOne)


class Omega(Expr):
    """
    Omega (w) -- reciprocal of zero. w = 1/0, and 0 * w = 1.
    """
    is_commutative = True
    is_number = False
    is_zero = False
    _op_priority = 15.0

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return 'w'

    def _latex(self, printer):
        return r'\omega'

    def __repr__(self):
        return 'w'

    def __str__(self):
        return 'w'

    def _eval_power(self, exp):
        """
        Base-w exponentiation rules:
            w^0     = 1         w^1     = w
            w^(-n)  = 0^n       w^w     = -1
            w^(0^x) = -x        w^(w^x) = -1/x
        """
        if exp == S.Zero or isinstance(exp, Zero):
            return S.One
        if exp == S.One:
            return Omega()
        if isinstance(exp, Integer) and exp.is_negative:
            return Pow(Zero(), -exp)
        if isinstance(exp, Omega):
            return S.NegativeOne
        if isinstance(exp, Pow) and isinstance(exp.base, Zero):
            return -exp.exp       # w^(0^x) = -x for any x
        if isinstance(exp, Pow) and isinstance(exp.base, Omega):
            return -S.One / exp.exp  # w^(w^x) = -1/x for any x
        return None

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, Zero):
            return S.One
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, Zero):
            return S.One
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        return self * Pow(other, S.NegativeOne)

    def __rtruediv__(self, other):
        other = sympify(other)
        return other * Pow(self, S.NegativeOne)


class Null(Expr):
    """
    Null -- the erasure element. Result of a - a.
    Distinct from numeric zero.
    """
    is_commutative = True
    is_number = False
    is_zero = False

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return 'null'

    def _latex(self, printer):
        return r'\emptyset'

    def __repr__(self):
        return 'null'

    def __str__(self):
        return 'null'


# ============================================================
# Singleton Aliases
# ============================================================

z = Zero()
w = Omega()
null = Null()


# ============================================================
# Simplification Engine
# ============================================================

def traction_simplify(expr):
    """
    Apply Traction Theory rewrite rules to a SymPy expression.
    Processes bottom-up: simplifies sub-expressions first.
    """
    expr = sympify(expr)

    if isinstance(expr, (Zero, Omega, Null, Number, Symbol)):
        return expr

    if isinstance(expr, Pow):
        base = traction_simplify(expr.base)
        exp = traction_simplify(expr.exp)
        # x^0 = 1 for any x (Zero() exponent may not be caught by SymPy)
        if isinstance(exp, Zero):
            return S.One
        # Collapse nested powers: (b^e1)^e2 -> b^(e1*e2) for traction bases
        if isinstance(base, Pow) and isinstance(base.base, (Zero, Omega)):
            combined_exp = traction_simplify(Mul(base.exp, exp))
            return traction_simplify(Pow(base.base, combined_exp))
        return Pow(base, exp)

    if isinstance(expr, Mul):
        return _simplify_mul(expr)

    if isinstance(expr, Add):
        return _simplify_add(expr)

    if hasattr(expr, 'args') and expr.args:
        new_args = [traction_simplify(a) for a in expr.args]
        return expr.func(*new_args)

    return expr


def _simplify_mul(expr):
    """
    Apply Traction multiplication table to a Mul expression.

    Unifies all Zero and Omega factors into a single base-0 power
    using the identity w^a = 0^(-a), then sums exponents:
        0^a * 0^b  ->  0^(a+b)
        0^a * w^b  ->  0^(a-b)
        w^a * w^b  ->  0^(-a-b)
    """
    raw_args = list(Mul.make_args(expr))
    args = [traction_simplify(a) for a in raw_args]

    # Re-flatten after simplification
    flat = []
    for a in args:
        flat.extend(Mul.make_args(a))

    # Collect base-0 exponents (unifying omega via w^a = 0^(-a))
    zero_exp = S.Zero
    others = []

    for a in flat:
        if isinstance(a, Zero):
            zero_exp += S.One
        elif isinstance(a, Omega):
            zero_exp += S.NegativeOne
        elif isinstance(a, Pow) and isinstance(a.base, Zero):
            zero_exp += a.exp
        elif isinstance(a, Pow) and isinstance(a.base, Omega):
            zero_exp -= a.exp
        else:
            others.append(a)

    # Reconstruct: 0^zero_exp (triggers _eval_power for known cases)
    if zero_exp != S.Zero:
        power = Pow(Zero(), traction_simplify(zero_exp))
        others.append(power)

    if not others:
        return S.One
    if len(others) == 1:
        return others[0]
    return Mul(*others)


def _simplify_add(expr):
    """Apply Traction addition rules to an Add expression."""
    args = [traction_simplify(a) for a in expr.args]
    result = Add(*args)

    # a - a = null (SymPy reduces cancellation to S.Zero)
    if result == S.Zero:
        return Null()

    return result


# ============================================================
# Logarithm Functions
# ============================================================

def log0(expr):
    """
    Base-0 logarithm.

        log_0(1)    = 0         log_0(0)    = 1
        log_0(0^n)  = n         log_0(n)    = 0^n
        log_0(w^n)  = -n        log_0(-n)   = w^n

    Returns None if the expression cannot be simplified.
    """
    expr = sympify(expr)
    if expr == S.One:
        return S.Zero
    if isinstance(expr, Zero):
        return S.One
    if isinstance(expr, Omega):
        return S.NegativeOne
    if isinstance(expr, Pow) and isinstance(expr.base, Zero):
        return expr.exp
    if isinstance(expr, Pow) and isinstance(expr.base, Omega):
        return -expr.exp
    if isinstance(expr, Integer):
        if expr.is_positive:
            return Pow(Zero(), expr)
        if expr.is_negative:
            return Pow(Omega(), -expr)
    return None


def logw(expr):
    """
    Base-w logarithm.

        log_w(1)    = 0         log_w(w)    = 1
        log_w(w^n)  = n         log_w(0^n)  = -n
        log_w(-n)   = 0^n

    Note: the reference doc shows log_w(w^n) = -n which contradicts
    log_w(w) = 1. This implementation uses log_w(w^n) = n for consistency.

    Returns None if the expression cannot be simplified.
    """
    expr = sympify(expr)
    if expr == S.One:
        return S.Zero
    if isinstance(expr, Omega):
        return S.One
    if isinstance(expr, Zero):
        return S.NegativeOne
    if isinstance(expr, Pow) and isinstance(expr.base, Omega):
        return expr.exp
    if isinstance(expr, Pow) and isinstance(expr.base, Zero):
        return -expr.exp
    if isinstance(expr, Integer) and expr.is_negative:
        return Pow(Zero(), -expr)
    return None


# ============================================================
# Shorthand Constructors
# ============================================================

def zpow(n):
    """Shorthand for 0^n."""
    return Pow(Zero(), sympify(n))


def wpow(n):
    """Shorthand for w^n."""
    return Pow(Omega(), sympify(n))


# ============================================================
# Identity Resolution
# ============================================================

def resolve(expr):
    """
    Force simplification by wrapping an expression through a no-op identity cycle.

    Uses 0^(0^(expr)) which equals expr by the generalized identity 0^(0^x) = x.
    The intermediate step through the zero-power domain may trigger additional
    simplifications that wouldn't fire on the original form.

    Multiple wrappers are available — all are no-ops:
        0^(0^x) = x         log_0(0^x) = x
        0^(log_0(x)) = x    log_0(log_0(x)) = x
    """
    expr = traction_simplify(sympify(expr))
    return traction_simplify(Pow(Zero(), Pow(Zero(), expr)))


def resolve_log(expr):
    """
    Resolve via log_0(0^(expr)) = expr.
    Alternative resolution path through logarithm.
    """
    expr = traction_simplify(sympify(expr))
    wrapped = Pow(Zero(), expr)  # 0^expr
    return log0(wrapped)         # log_0(0^expr) = expr


# ============================================================
# Complex Projection
# ============================================================

def project_complex(expr):
    """
    Project a traction expression to the complex numbers.

    The core mapping: 0^(t*w) -> e^(i*pi*t), derived from 0^w = -1 = e^(i*pi).

    This handles:
        Integers/rationals   ->  themselves
        0^(t*w)              ->  e^(i*pi*t)
        a * 0^(t*w)          ->  a * e^(i*pi*t)
        Products/sums        ->  distributed
        Symbols              ->  left as-is

    Integer zero-powers (0^n for n > 0) have no known general mapping
    to C yet, so they are left unevaluated.

    Returns a SymPy complex expression.
    """
    expr = traction_simplify(expr)
    return _project(expr)


def _project(expr):
    """Recursively project a traction expression to C."""
    # Atoms
    if isinstance(expr, (Integer, Rational)):
        return expr
    if isinstance(expr, Zero):
        return S.Zero
    if isinstance(expr, Omega):
        return zoo
    if isinstance(expr, Null):
        return S.Zero
    if isinstance(expr, Symbol):
        return expr

    # Powers
    if isinstance(expr, Pow):
        return _project_pow(expr)

    # Mul: project each factor
    if isinstance(expr, Mul):
        result = S.One
        for arg in expr.args:
            result = result * _project(arg)
        return sp_simplify(result)

    # Add: project each term
    if isinstance(expr, Add):
        result = S.Zero
        for arg in expr.args:
            result = result + _project(arg)
        return sp_simplify(result)

    return expr


def _project_pow(expr):
    """Project a Pow expression to C."""
    base = expr.base
    exponent = expr.exp

    # Non-traction base: project base and exponent separately
    if not isinstance(base, (Zero, Omega)):
        pb = _project(base)
        pe = _project(exponent)
        return Pow(pb, pe)

    # Convert omega base to zero base: w^a = 0^(-a)
    if isinstance(base, Omega):
        exponent = -exponent

    # Now we have 0^exponent. Extract the w-coefficient.
    exponent = traction_simplify(exponent)

    # 0^0 = 1
    if exponent == S.Zero or isinstance(exponent, Zero):
        return S.One

    # Pure w: 0^w = e^(i*pi) = -1
    if isinstance(exponent, Omega):
        return S.NegativeOne

    # Extract w-coefficient: exponent = remainder + coeff * w
    w_coeff = _omega_coeff(exponent)
    remainder = traction_simplify(exponent - w_coeff * Omega()) if w_coeff != S.Zero else exponent

    if w_coeff != S.Zero:
        # Phase: e^(i*pi*t) where t is the w-coefficient
        phase = sp_simplify(sp_exp(I * pi * w_coeff))

        if remainder == S.Zero or isinstance(remainder, Zero):
            # Pure phase: 0^(t*w) -> e^(i*pi*t)
            return phase

        # Mixed: 0^(r + t*w) -> e^(-W*r) * e^(i*pi*t)
        return sp_simplify(sp_exp(-W_CONST * remainder) * phase)

    # No w-component: 0^z -> e^(-W*z) via Lie exponential
    # where W = sqrt(-i*pi), so W^2 = -i*pi
    return sp_exp(-W_CONST * exponent)


def _omega_coeff(expr):
    """
    Extract the coefficient of w in an expression.
    E.g. w/2 -> 1/2, 3 + 2*w -> 2, w -> 1, 5 -> 0.
    """
    expr = sympify(expr)
    if isinstance(expr, Omega):
        return S.One
    if hasattr(expr, 'coeff'):
        return expr.coeff(Omega())
    return S.Zero
