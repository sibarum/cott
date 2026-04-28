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
# Connects traction algebra to Lie group exponential (legacy)
W_CONST = sp_sqrt(-I * pi)

# Chebyshev evaluation angle: 0^z = e^(i*THETA*z) after omega -> pi/THETA
# At this angle, 0^omega = e^(i*pi) = -1 (always), and all 0^n lie on the unit circle.
# Default: pi/4 (matches Phase Map slider default)
CHEB_THETA = pi * 2

# In exponents, omega acts as pi/THETA (so that 0^omega = e^(i*THETA*pi/THETA) = e^(i*pi) = -1)
OMEGA_EXP_VAL = pi / CHEB_THETA


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
        if isinstance(exp, (Integer, Rational)) and exp.is_negative:
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
        if isinstance(exp, (Integer, Rational)) and exp.is_negative:
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
# Graded Algebra — Z-action
# ============================================================

class GradedElement(Expr):
    """
    Z_n(value) — an element at grade n in the operation hierarchy.

    The Z-action shifts arithmetic operations between grades:
        Z_n(a) + Z_n(b) = Z_{n-1}(a*b)      addition  → multiplication
        Z_n(a) * Z_n(b) = Z_{n+1}(a+b)      multiply  → addition
        Z_n(a) - Z_n(b) = Z_{n-1}(a/b)      subtract  → division
        Z_n(a) / Z_n(b) = Z_{n+1}(a-b)      division  → subtraction

    Exponentiation with scalars:
        b ^ Z_n(a)       = Z_{n-1}(a*b)
        Z_n(a) ^ b       = Z_{n+1}(a*b)

    Inversion rules:
        Z_n(1/a)         = -Z_{n-1}(a)       reciprocal → negation (down)
        Z_n(-a)          = 1/Z_{n+1}(a)      negation   → reciprocal (up)

    Fixed points:
        Z_n(1)  = Z_{n±1}(0)
        Z_n(0)  = Z_{n±1}(1)
        Z_n(-1) = Z_{n±1}(w)
        Z_n(w)  = Z_{n±1}(-1)

    Composition:
        Z_n(Z_{n+1}(x)) = Z_n(Z_{n-1}(x)) = x

    Z_0 is the identity: Z_0(x) = x.
    """
    is_commutative = True
    _op_priority = 16.0  # higher than Zero/Omega so our methods dispatch first

    def __new__(cls, grade, value):
        grade = sympify(grade)
        value = sympify(value)

        # Z_0(x) = x  (identity grade)
        if grade == S.Zero:
            return value

        # Fixed-point normalization: canonicalize toward lowest grade.
        # Z_n(0) = Z_{n-1}(1), so shift down:
        if (isinstance(value, Integer) and value == S.Zero) or isinstance(value, Zero):
            return GradedElement.__new__(cls, grade - 1, S.One)
        # Z_n(w) = Z_{n-1}(-1), so shift down:
        if isinstance(value, Omega):
            return GradedElement.__new__(cls, grade - 1, S.NegativeOne)
        # Z_n(-1) = Z_{n-1}(w) = Z_{n-2}(-1) = ... → collapses to grade 0 or 1.
        # -1 and w alternate: shift down by 2 each cycle until grade ≤ 1.
        if isinstance(value, Integer) and value == S.NegativeOne:
            g = int(grade)
            if g >= 2:
                # Even grade → Z_0(-1) = -1; odd grade → Z_1(-1)
                final_grade = g % 2
                if final_grade == 0:
                    return S.NegativeOne
                return Expr.__new__(cls, S.One, S.NegativeOne)

        # Composition collapse: Z_n(Z_{n+1}(x)) = x, Z_n(Z_{n-1}(x)) = x
        if isinstance(value, GradedElement):
            inner_grade = value.args[0]
            inner_value = value.args[1]
            diff = grade - inner_grade
            if diff == S.One or diff == S.NegativeOne:
                return inner_value

        return Expr.__new__(cls, grade, value)

    @property
    def grade(self):
        return self.args[0]

    @property
    def value(self):
        return self.args[1]

    def _sympystr(self, printer):
        return f'Z_{printer.doprint(self.grade)}({printer.doprint(self.value)})'

    def _latex(self, printer):
        return r'Z_{' + printer.doprint(self.grade) + r'}\left(' + printer.doprint(self.value) + r'\right)'

    def __repr__(self):
        return f'Z_{self.grade}({self.value})'

    def __str__(self):
        return f'Z_{self.grade}({self.value})'

    # --- Arithmetic: same-grade operations shift the grade ---

    def __add__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            # Z_n(a) + Z_n(b) = Z_{n-1}(a*b)
            return GradedElement(self.grade - 1, self.value * other.value)
        return Expr.__add__(self, other)

    def __radd__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, other.value * self.value)
        return Expr.__radd__(self, other)

    def __sub__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            # Z_n(a) - Z_n(b) = Z_{n-1}(a/b)
            return GradedElement(self.grade - 1, self.value / other.value)
        return Expr.__sub__(self, other)

    def __rsub__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, other.value / self.value)
        return Expr.__rsub__(self, other)

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            # Z_n(a) * Z_n(b) = Z_{n+1}(a+b)
            return GradedElement(self.grade + 1, self.value + other.value)
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, other.value + self.value)
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            # Z_n(a) / Z_n(b) = Z_{n+1}(a-b)
            return GradedElement(self.grade + 1, self.value - other.value)
        return Expr.__truediv__(self, other)

    def __rtruediv__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, other.value - self.value)
        return Expr.__rtruediv__(self, other)

    def __pow__(self, exp):
        exp = sympify(exp)
        # Z_n(a) ^ b = Z_{n+1}(a*b)
        if not isinstance(exp, GradedElement):
            return GradedElement(self.grade + 1, self.value * exp)
        return Expr.__pow__(self, exp)

    def __rpow__(self, base):
        base = sympify(base)
        # b ^ Z_n(a) = Z_{n-1}(a*b)
        if not isinstance(base, GradedElement):
            return GradedElement(self.grade - 1, self.value * base)
        return Expr.__rpow__(self, base)

    def __neg__(self):
        # Z_n(-a) = 1/Z_{n+1}(a)  →  -Z_n(a) means we negate the value
        # -Z_n(a) = Z_n(-a) = 1/Z_{n+1}(a)
        return Pow(GradedElement(self.grade + 1, self.value), S.NegativeOne)

    def _eval_power(self, exp):
        """Handle Pow(Z_n(a), b) when SymPy constructs it."""
        if not isinstance(exp, GradedElement):
            # Z_n(a) ^ b = Z_{n+1}(a*b)
            return GradedElement(self.grade + 1, self.value * exp)
        return None

    def inverse(self):
        """Z_n(1/a) = -Z_{n-1}(a), so 1/Z_n(a) needs the inverse rule."""
        # From Z_n(-a) = 1/Z_{n+1}(a), substitute a→-a:
        # Z_n(a) = 1/Z_{n+1}(-a), so 1/Z_n(a) = Z_{n+1}(-a)
        # Alternatively from Z_n(1/a) = -Z_{n-1}(a):
        # if self = Z_n(a), then Z_{n+1}(-a) inverts it
        return GradedElement(self.grade + 1, -self.value)


def Z(n, value):
    """Shorthand constructor: Z(n, x) = Z_n(x)."""
    return GradedElement(n, value)


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
        # Exponent distribution: 0^(A+B) -> 0^A * 0^B
        # Only fire when at least one term simplifies beyond a bare zero-power,
        # otherwise _simplify_mul will recombine and we get an infinite loop.
        if isinstance(base, Zero) and isinstance(exp, Add):
            terms = Add.make_args(exp)
            parts = [traction_simplify(Pow(Zero(), t)) for t in terms]
            # Check if distribution made progress: at least one part is no longer 0^(...)
            if any(not (isinstance(p, Pow) and isinstance(p.base, Zero)) for p in parts):
                return traction_simplify(Mul(*parts))
        # Exponent factoring: 0^(c * A) -> (0^A)^c when A contains traction types
        # Substitute ω -> 0^(-1) first so that e.g. 0^(2ω) becomes 0^(2·0^(-1)),
        # then factor as (0^(0^(-1)))^2 = (-1)^2 = 1.
        if isinstance(base, Zero) and isinstance(exp, Mul):
            exp_sub = exp.subs(Omega(), Pow(Zero(), S.NegativeOne))
            factors = Mul.make_args(exp_sub)
            scalars = []
            traction_parts = []
            for f in factors:
                if isinstance(f, (Integer, Rational)):
                    scalars.append(f)
                else:
                    traction_parts.append(f)
            if scalars and traction_parts:
                scalar = Mul(*scalars)
                inner = Mul(*traction_parts) if len(traction_parts) > 1 else traction_parts[0]
                inner_result = traction_simplify(Pow(Zero(), inner))
                # Skip when inner_result == 1: 1^c = 1 trivially for any c, which would
                # collapse 0^(c·0) → 1 and discard phase info (e.g. 0^(0/2) ≠ 1, since
                # i^(i/2) = e^(-π/4) under the standard projection).
                if (inner_result != S.One
                        and not (isinstance(inner_result, Pow) and isinstance(inner_result.base, Zero))):
                    # The identity fired — inner_result is a non-trivial scalar, raise to scalar power.
                    # Only apply if the final result stays in traction (Integer, Rational, etc.)
                    # to avoid escaping to classical arithmetic (e.g. (-1)^(1/7)).
                    final = traction_simplify(Pow(inner_result, scalar))
                    if isinstance(final, (Integer, Rational, Zero, Omega, Null)):
                        return final
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
    Project a traction expression to the complex numbers via Chebyshev evaluation.

    The core mapping: 0^z -> e^(i*THETA*z), where THETA is the Chebyshev angle.
    In exponents, omega -> pi/THETA so that 0^omega = e^(i*pi) = -1 always.

    All zero-powers of real exponents land on the unit circle.
    The imaginary axis (from 0^(omega/2) = i) controls magnitude.

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
        # 0 = 0^1 → e^(i*THETA)
        return sp_exp(I * CHEB_THETA)
    if isinstance(expr, Omega):
        # ω = 0^(-1) → e^(-i*THETA)
        return sp_exp(-I * CHEB_THETA)
    if isinstance(expr, Null):
        return S.Zero
    if isinstance(expr, Symbol):
        return expr

    # Logarithms: 0^z = e^(i*THETA*z) → z = ln(y)/(i*THETA) → log_0(y) = -i*ln(y)/THETA
    # log_w(y) = i*ln(y)/THETA  (since log_w = -log_0)
    if isinstance(expr, Log0):
        arg = expr.args[0]
        arg = arg.subs(Omega(), OMEGA_EXP_VAL) if arg.has(Omega) else arg
        proj_arg = _project(arg)
        return sp_log(proj_arg) / (I * CHEB_THETA)

    if isinstance(expr, LogW):
        arg = expr.args[0]
        arg = arg.subs(Omega(), OMEGA_EXP_VAL) if arg.has(Omega) else arg
        proj_arg = _project(arg)
        return -sp_log(proj_arg) / (I * CHEB_THETA)

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
    """Project a Pow expression to C using Chebyshev evaluation."""
    base = expr.base
    exponent = expr.exp

    # Non-traction base: project base and exponent separately
    if not isinstance(base, (Zero, Omega)):
        if isinstance(base, Mul) and (base.has(Zero) or base.has(Omega)):
            result = S.One
            for factor in Mul.make_args(base):
                result = result * _project(Pow(factor, exponent))
            return result
        pb = _project(base)
        exp_projected = exponent.subs(Omega(), OMEGA_EXP_VAL) if exponent.has(Omega) else exponent
        pe = _project(exp_projected)
        # Negative real bases: (-a)^n = a^n * e^(iπn)
        if isinstance(pb, Number) and pb.is_negative:
            return Pow(-pb, pe) * sp_exp(I * pi * pe)
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

    # Chebyshev projection: 0^z = e^(i*THETA*z)
    # In exponents, ω → π/THETA so that 0^ω = e^(i*π) = -1.
    # All real exponents give unit-circle values.
    exp_sub = exponent.subs(Omega(), OMEGA_EXP_VAL) if exponent.has(Omega) else exponent
    proj_exp = _project(exp_sub)
    return sp_exp(I * CHEB_THETA * proj_exp)


