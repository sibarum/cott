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
    Expr, S, sympify, Pow, Mul, Add, Function,
    Integer, Rational, Symbol, Number,
    I, pi, sqrt as sp_sqrt, exp as sp_exp, log as sp_log,
    simplify as sp_simplify, zoo
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
    _op_priority = 15.0

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


class Log0(Function):
    """
    Unevaluated base-0 logarithm.
    Stays symbolic until projected to C via log_0(y) = -ln(y)/W.
    """
    def _sympystr(self, printer):
        return f'log_0({printer.doprint(self.args[0])})'

    def _latex(self, printer):
        return r'\log_{\mathbf{0}}\left(' + printer.doprint(self.args[0]) + r'\right)'


class LogW(Function):
    """
    Unevaluated base-w logarithm.
    Stays symbolic until projected to C via log_w(y) = ln(y)/W.
    """
    def _sympystr(self, printer):
        return f'log_w({printer.doprint(self.args[0])})'

    def _latex(self, printer):
        return r'\log_{\omega}\left(' + printer.doprint(self.args[0]) + r'\right)'


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
        # Universal power-of-power: (v^a)^b -> v^(a*b)
        # In traction algebra, all operations are reversible — no branch cuts.
        if isinstance(base, Pow):
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

    # Partition into zero-class and others
    zero_exp = S.Zero
    has_definite_zero = False  # True only for bare Zero() or Pow(Zero(), positive_int)
    others = []

    for a in flat:
        if isinstance(a, Zero):
            zero_exp += S.One
            has_definite_zero = True
        elif isinstance(a, Omega):
            zero_exp += S.NegativeOne
        elif isinstance(a, Pow) and isinstance(a.base, Zero):
            zero_exp += a.exp
            if isinstance(a.exp, Integer) and a.exp.is_positive:
                has_definite_zero = True
        elif isinstance(a, Pow) and isinstance(a.base, Omega):
            zero_exp -= a.exp
        else:
            others.append(a)

    # -0 = 0: zero-class elements absorb sign.
    # Negative zero cannot exist — 0-0 = null (erasure), so -0 reverts to 0.
    # Only applies when we KNOW the element is zero-class (bare Zero or
    # positive integer power of Zero), not for symbolic exponents like 0^x
    # which might be omega-class.
    if has_definite_zero:
        new_others = []
        for a in others:
            if isinstance(a, Number) and a.is_negative:
                a = -a  # absorb sign: |-n| * 0 = n * 0
            new_others.append(a)
        others = new_others

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
    # Can't simplify: return unevaluated symbolic form
    return Log0(expr)


def logw(expr):
    """
    Base-w logarithm.

        log_w(1)    = 0         log_w(w)    = 1
        log_w(w^n)  = n         log_w(0^n)  = -n
        log_w(-n)   = 0^n

    Returns unevaluated LogW(expr) if it cannot be simplified.
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
    # Can't simplify: return unevaluated symbolic form
    return LogW(expr)


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

    # Logarithms: log_0(y) = -ln(y)/W, log_w(y) = ln(y)/W
    # Derived from 0^z = e^(-W*z) -> z = -ln(0^z)/W -> log_0(y) = -ln(y)/W
    # Substitute ω → W in the argument (same principle as exponents)
    if isinstance(expr, Log0):
        arg = expr.args[0]
        arg = arg.subs(Omega(), W_CONST) if arg.has(Omega) else arg
        proj_arg = _project(arg)
        return -sp_log(proj_arg) / W_CONST

    if isinstance(expr, LogW):
        arg = expr.args[0]
        arg = arg.subs(Omega(), W_CONST) if arg.has(Omega) else arg
        proj_arg = _project(arg)
        return sp_log(proj_arg) / W_CONST

    # Powers
    if isinstance(expr, Pow):
        return _project_pow(expr)

    # Mul: project each factor
    if isinstance(expr, Mul):
        result = S.One
        for arg in expr.args:
            result = result * _project(arg)
        return result

    # Add: project each term
    if isinstance(expr, Add):
        result = S.Zero
        for arg in expr.args:
            result = result + _project(arg)
        return result

    return expr


def _project_pow(expr):
    """Project a Pow expression to C."""
    base = expr.base
    exponent = expr.exp

    # Non-traction base: project base and exponent separately
    # In exponents, ω acts as W (the structure constant): ω = W in exponent space
    # Derived from: 0^(tω) = e^(-W·tω) = e^(iπt), so W·ω = -iπ = W², thus ω = W
    if not isinstance(base, (Zero, Omega)):
        pb = _project(base)
        exp_projected = exponent.subs(Omega(), W_CONST) if exponent.has(Omega) else exponent
        pe = _project(exp_projected)
        return Pow(pb, pe)

    # Convert omega base to zero base: w^a = 0^(-a)
    if isinstance(base, Omega):
        exponent = -exponent

    # Now we have 0^exponent.
    exponent = traction_simplify(exponent)

    # Quick exits for simple cases
    if exponent == S.Zero or isinstance(exponent, Zero):
        return S.One
    if isinstance(exponent, Omega):
        return S.NegativeOne

    # General case: 0^z = e^(-W*z)
    # Substitute ω → W in the exponent (in exponents, ω acts as W),
    # then project any remaining traction types (e.g. 0^(ω/2) → i).
    exp_with_w = exponent.subs(Omega(), W_CONST) if exponent.has(Omega) else exponent
    proj_exp = _project(exp_with_w)
    return sp_exp(-W_CONST * proj_exp)


