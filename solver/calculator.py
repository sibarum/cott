"""
COTT Calculator — Traction Expression Evaluator
A tkinter GUI for interactively evaluating Traction Theory expressions.

Usage:
    conda activate traction
    python calculator.py
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from sympy import S, Integer, Rational, Pow, Mul, Add, Symbol, I
from sympy import symbols, lambdify
from sympy import solve as sp_solve, expand as sp_expand, factor as sp_factor
from traction import Zero, Omega, Null, Log0, LogW, z, w, null, traction_simplify, log0, logw, project_complex, GradedElement, Z
import registry
import projections  # auto-discovers and registers projection plugins

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
# Expression Parser
# ============================================================

class ParseError(Exception):
    pass


class Parser:
    """
    Recursive descent parser for traction expressions.

    Grammar:
        expr     = additive
        additive = mult (('+' | '-') mult)*
        mult     = power (('*' | '/') power)*
        power    = unary ('^' power)?
        unary    = '-' unary | primary
        primary  = '(' expr ')' | graded | funccall | number | 'w' | 'x' | 'omega'
        graded   = 'Z_' int '(' expr ')' | 'Z' '(' expr ',' expr ')'
        funccall = ('log0' | 'logw') '(' expr ')'
    """

    def __init__(self, text):
        self.text = text.replace(' ', '')
        self.pos = 0

    def peek(self):
        if self.pos < len(self.text):
            return self.text[self.pos]
        return None

    def consume(self, expected=None):
        ch = self.peek()
        if expected and ch != expected:
            raise ParseError(f"Expected '{expected}' at position {self.pos}, got '{ch}'")
        self.pos += 1
        return ch

    def match(self, s):
        if self.text[self.pos:self.pos + len(s)] == s:
            self.pos += len(s)
            return True
        return False

    def parse(self):
        result = self.expr()
        if self.pos < len(self.text):
            raise ParseError(f"Unexpected character '{self.text[self.pos]}' at position {self.pos}")
        return result

    def expr(self):
        return self.additive()

    def additive(self):
        left = self.mult()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.mult()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    def mult(self):
        left = self.power()
        while self.peek() in ('*', '/', '\u00d7', '\u00f7'):
            op = self.consume()
            right = self.power()
            if op in ('*', '\u00d7'):
                left = left * right
            else:
                left = left / right
        return left

    def power(self):
        base = self.unary()
        if self.peek() == '^':
            self.consume()
            exp = self.power()  # right-associative
            base = base ** exp
        return base

    def unary(self):
        if self.peek() == '-':
            self.consume()
            operand = self.unary()
            # -0 = 0: negative zero cannot exist (0-0 = null, not -0)
            if isinstance(operand, Zero):
                return operand
            return -operand
        return self.primary()

    def primary(self):
        ch = self.peek()
        if ch is None:
            raise ParseError("Unexpected end of expression")

        # Parenthesized expression
        if ch == '(':
            self.consume('(')
            result = self.expr()
            self.consume(')')
            return result

        # omega: 'w' or 'omega' or unicode omega
        if ch == '\u03c9':
            self.consume()
            return w
        if self.match('omega'):
            return w

        # Graded elements: Z_n(expr) or Z(n, expr)
        if ch in ('Z', 'z'):
            if self.match('Z_') or self.match('z_'):
                return self._parse_graded()
            if self.match('Z(') or self.match('z('):
                return self._parse_graded_func()

        # Function calls: log0(...), logw(...), solve(...), expand(...), factor(...)
        if self.match('log0'):
            return self._parse_func_call(log0, 'log0')
        if self.match('log\u03c9'):
            return self._parse_func_call(logw, 'log\u03c9')
        if self.match('logw'):
            return self._parse_func_call(logw, 'logw')
        if self.match('solve'):
            return self._parse_solve()
        if self.match('expand'):
            return self._parse_func_call(sp_expand, 'expand')
        if self.match('factor'):
            return self._parse_func_call(sp_factor, 'factor')

        # 'w' (must check after 'logw')
        if ch == 'w':
            self.consume()
            return w

        # Variables: p (horizontal), q (vertical), x (projection native unit),
        # c (fractal pixel coordinate)
        if ch == 'p':
            self.consume()
            return Symbol('p')
        if ch == 'q':
            self.consume()
            return Symbol('q')
        if ch == 'x':
            self.consume()
            return Symbol('x')
        if ch == 'c':
            self.consume()
            return Symbol('c')
        # Legacy: y still works as alias for q
        if ch == 'y':
            self.consume()
            return Symbol('q')

        # null
        if self.match('null'):
            return null

        # Number (integer, rational p/q, or decimal)
        if ch.isdigit() or (ch == '.' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
            return self.number()

        raise ParseError(f"Unexpected character '{ch}' at position {self.pos}")

    def _parse_graded(self):
        """Parse Z_n(expr) where n is an integer (possibly negative)."""
        # We already consumed 'Z_', now read the grade
        neg = False
        if self.peek() == '-':
            neg = True
            self.consume()
        if self.peek() is None or not self.peek().isdigit():
            raise ParseError(f"Expected digit after 'Z_' at position {self.pos}")
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        grade = int(self.text[start:self.pos])
        if neg:
            grade = -grade
        if self.peek() != '(':
            raise ParseError(f"Expected '(' after Z_{grade} at position {self.pos}")
        self.consume('(')
        arg = self.expr()
        self.consume(')')
        return Z(grade, arg)

    def _parse_graded_func(self):
        """Parse Z(n, expr) — already consumed 'Z('."""
        # Read grade expression (first argument)
        grade_expr = self.expr()
        if self.peek() != ',':
            raise ParseError(f"Expected ',' in Z(n, expr) at position {self.pos}")
        self.consume(',')
        arg = self.expr()
        self.consume(')')
        return Z(grade_expr, arg)

    def _parse_func_call(self, func, name):
        """Parse func(expr) and return the result."""
        if self.peek() != '(':
            raise ParseError(f"Expected '(' after {name} at position {self.pos}")
        self.consume('(')
        arg = self.expr()
        self.consume(')')
        return func(arg)

    def _parse_solve(self):
        """Parse solve(expr) or solve(expr, var) and return a SolutionSet."""
        if self.peek() != '(':
            raise ParseError(f"Expected '(' after solve at position {self.pos}")
        self.consume('(')
        expr = self.expr()
        var = None
        if self.peek() == ',':
            self.consume(',')
            var = self.expr()
            if not isinstance(var, Symbol):
                raise ParseError("Second argument to solve must be a variable")
        self.consume(')')
        return _do_solve(expr, var)

    def number(self):
        start = self.pos

        # Leading dot: .5 = 0.5
        if self.peek() == '.':
            self.consume('.')
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            full_str = '0.' + self.text[start + 1:self.pos]
            from fractions import Fraction
            return Rational(Fraction(full_str).limit_denominator(10**12))

        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1

        # Check for decimal: n.mmm
        if self.peek() == '.' and self.pos < len(self.text) - 1 and self.text[self.pos + 1].isdigit():
            self.consume('.')
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            full_str = self.text[start:self.pos]
            from fractions import Fraction
            return Rational(Fraction(full_str).limit_denominator(10**12))

        n = int(self.text[start:self.pos])

        # Check for rational: n/m where m is immediately after /
        # But only if / is followed by digits (not an operator context)
        if self.peek() == '/' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit():
            saved = self.pos
            self.consume('/')
            dstart = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            d = int(self.text[dstart:self.pos])
            if d == 0 or n == 0:
                # n/0 → division by zero (handled as operator, not rational)
                # 0/d → traction zero divided by d (not Rational(0,d) which collapses to numeric 0)
                self.pos = saved
                return _to_traction(n)
            return Rational(n, d)

        return _to_traction(n)


def _to_traction(n):
    """Convert a Python int to the appropriate traction type."""
    if n == 0:
        return z
    return Integer(n)


class SolutionSet:
    """Wrapper for a list of solutions returned by solve()."""
    def __init__(self, solutions, variable):
        self.solutions = solutions
        self.variable = variable

    def has(self, sym):
        """SymPy-compatible: check if any solution references sym."""
        return any(hasattr(s, 'has') and s.has(sym) for s in self.solutions)


def _guess_variable(expr):
    """Pick the solve variable from an expression's free symbols."""
    free = expr.free_symbols
    if not free:
        raise ParseError("Nothing to solve: expression has no variables")
    # Prefer x, then p, then q, then alphabetically first
    for name in ('x', 'p', 'q'):
        sym = Symbol(name)
        if sym in free:
            return sym
    return sorted(free, key=str)[0]


def _standardize_for_solve(expr):
    """Replace traction Zero/Omega with standard numeric values for solving."""
    if isinstance(expr, Zero):
        return S.Zero
    if isinstance(expr, Omega):
        return S.ComplexInfinity
    if hasattr(expr, 'args') and expr.args:
        new_args = [_standardize_for_solve(a) for a in expr.args]
        return expr.func(*new_args)
    return expr


def _do_solve(expr, var=None):
    """Solve expr = 0 for var, returning a SolutionSet."""
    expr = _standardize_for_solve(expr)
    if var is None:
        var = _guess_variable(expr)
    solutions = sp_solve(expr, var)
    return SolutionSet(solutions, var)


def parse_and_eval(text):
    """Parse a traction expression string and return the simplified result.

    Supports equation syntax: 'lhs = rhs' is solved as 'lhs - rhs = 0'.
    """
    text = text.strip()
    if not text:
        return None

    # Equation syntax: split on '=' (but not inside function calls)
    if '=' in text and not text.startswith('solve'):
        # Find top-level '=' (not inside parentheses)
        depth = 0
        eq_pos = None
        for i, ch in enumerate(text):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '=' and depth == 0:
                eq_pos = i
                break
        if eq_pos is not None:
            lhs_text = text[:eq_pos].strip()
            rhs_text = text[eq_pos + 1:].strip()
            lhs = Parser(lhs_text).parse()
            rhs = Parser(rhs_text).parse()
            return _do_solve(lhs - rhs)

    parser = Parser(text)
    result = parser.parse()
    if isinstance(result, SolutionSet):
        return result
    return traction_simplify(result)


# ============================================================
# Display Formatting
# ============================================================

def format_result(expr):
    """Format a traction expression for display."""
    if expr is None:
        return ''
    if isinstance(expr, SolutionSet):
        if not expr.solutions:
            return 'no solution'
        parts = [format_result(s) for s in expr.solutions]
        var = str(expr.variable)
        if len(parts) == 1:
            return f'{var} = {parts[0]}'
        return f'{var} \u2208 {{{", ".join(parts)}}}'
    if isinstance(expr, Zero):
        return '0'
    if isinstance(expr, Omega):
        return '\u03c9'
    if isinstance(expr, Null):
        return '\u2205'  # empty set symbol
    if isinstance(expr, GradedElement):
        grade = format_result(expr.grade)
        value = format_result(expr.value)
        return f'Z_{grade}({value})'
    if isinstance(expr, Symbol):
        return str(expr)
    if isinstance(expr, Log0):
        return f'log\u2080({format_result(expr.args[0])})'
    if isinstance(expr, LogW):
        return f'log\u03c9({format_result(expr.args[0])})'
    if expr == S.NegativeOne:
        return '-1'
    if isinstance(expr, Integer):
        return str(expr)
    if isinstance(expr, Rational):
        return f'{expr.p}/{expr.q}'
    if isinstance(expr, Pow):
        base = format_result(expr.base)
        exp = format_result(expr.exp)
        if needs_parens(expr.base):
            base = f'({base})'
        if needs_parens_exp(expr.exp):
            exp = f'({exp})'
        return f'{base}^{exp}'
    if isinstance(expr, Mul):
        args = list(expr.args)
        # -1·X displays as -X
        if args[0] == S.NegativeOne and len(args) >= 2:
            rest = Mul(*args[1:]) if len(args) > 2 else args[1]
            rest_str = format_result(rest)
            if needs_parens(rest):
                rest_str = f'({rest_str})'
            return f'-{rest_str}'
        parts = []
        for arg in args:
            s = format_result(arg)
            if isinstance(arg, Add):
                s = f'({s})'
            parts.append(s)
        return '\u00b7'.join(parts)
    if isinstance(expr, Add):
        parts = []
        for i, arg in enumerate(expr.args):
            s = format_result(arg)
            if i > 0 and not s.startswith('-'):
                s = '+' + s
            parts.append(s)
        return ''.join(parts)
    return str(expr)


def needs_parens(expr):
    """Check if an expression needs parentheses when used as a base."""
    return isinstance(expr, (Add, Mul)) or (isinstance(expr, Integer) and expr < 0)


def needs_parens_exp(expr):
    """Check if an expression needs parentheses when used as an exponent."""
    if isinstance(expr, Integer):
        return False  # Plain integers don't need parens (Integer is a subclass of Rational)
    return isinstance(expr, (Add, Mul, Rational))


def format_approx(expr):
    """
    Format a decimal approximation of an expression.
    Returns '' if the approximation adds no information (e.g. pure integers).
    Returns '' for SolutionSet (no single numeric approximation).
    Recursively approximates numeric sub-parts within traction expressions.
    """
    if expr is None or isinstance(expr, SolutionSet):
        return ''

    has_approx, text = _approx_inner(expr)
    return text if has_approx else ''


def _approx_inner(expr):
    """
    Returns (has_approximation, formatted_string).
    has_approximation is True if any part was approximated (i.e. a Rational
    was converted to decimal), meaning the output is worth displaying.
    """
    if isinstance(expr, Zero):
        return False, '0'
    if isinstance(expr, Omega):
        return False, '\u03c9'
    if isinstance(expr, Null):
        return False, '\u2205'
    if isinstance(expr, Symbol):
        return False, str(expr)
    if isinstance(expr, Integer):
        return False, str(int(expr))
    if isinstance(expr, Rational):
        return True, f'{float(expr):.6g}'

    if isinstance(expr, Pow):
        b_approx, b_str = _approx_inner(expr.base)
        e_approx, e_str = _approx_inner(expr.exp)
        if needs_parens(expr.base):
            b_str = f'({b_str})'
        if needs_parens_exp(expr.exp) or e_approx:
            e_str = f'({e_str})'
        return b_approx or e_approx, f'{b_str}^{e_str}'

    if isinstance(expr, Mul):
        parts = []
        any_approx = False
        for arg in expr.args:
            a, s = _approx_inner(arg)
            any_approx = any_approx or a
            if isinstance(arg, Add):
                s = f'({s})'
            parts.append(s)
        return any_approx, '\u00b7'.join(parts)

    if isinstance(expr, Add):
        parts = []
        any_approx = False
        for i, arg in enumerate(expr.args):
            a, s = _approx_inner(arg)
            any_approx = any_approx or a
            if i > 0 and not s.startswith('-'):
                s = '+' + s
            parts.append(s)
        return any_approx, ''.join(parts)

    # Fallback: try float conversion
    try:
        val = float(expr)
        if val != int(val):
            return True, f'{val:.6g}'
        return False, str(int(val))
    except (TypeError, ValueError):
        return False, str(expr)


def format_complex(expr):
    """
    Format the complex projection C(expr).
    Returns '' if the projection is identical to the traction result
    (e.g. pure integers), or if it can't be projected.
    """
    if expr is None:
        return ''
    try:
        proj = project_complex(expr)
    except Exception:
        return ''

    # Don't show if projection is identical to the input
    proj_str = _format_sympy_complex(proj)
    orig_str = format_result(expr)
    if proj_str == orig_str:
        return ''

    # Replace re/im decompositions with p/q for readability (matches plot axes)
    proj_str = proj_str.replace('re(x)', 'p').replace('im(x)', 'q')
    proj_str = proj_str.replace('re(p)', 'p').replace('im(p)', 'q')

    return proj_str


def _format_sympy_complex(expr):
    """Format a SymPy complex expression for display."""
    from sympy import I, im, re, cos

    # If expression still contains traction types, format as mixed
    if expr.has(Zero) or expr.has(Omega) or expr.has(Null):
        return _format_mixed(expr)

    # Try to simplify — but avoid expand() which blows up exponentials
    try:
        simplified = expr.simplify()
        # Only use simplified if it's shorter (avoid blowup)
        if len(str(simplified)) <= len(str(expr)):
            expr = simplified
    except Exception:
        pass

    # For compact expressions with symbols, format directly rather than
    # decomposing into re/im (which expands exponentials into trig).
    if expr.free_symbols and len(str(expr)) < 80:
        s = str(expr).replace('I', 'i').replace('**', '^').replace('*', '\u00b7')
        s = s.replace('sqrt(pi)', '\u221a\u03c0').replace('sqrt(-i)', 'W')
        return s

    # Check for pure real
    if expr.is_real:
        return _format_real(expr)

    # Check for pure imaginary
    if expr.is_imaginary:
        coeff = expr / I
        if coeff == S.One:
            return 'i'
        if coeff == S.NegativeOne:
            return '-i'
        return f'{_format_real(coeff)}i'

    # a + bi form
    try:
        r = re(expr).simplify()
        i = im(expr).simplify()
        if r != S.Zero and i != S.Zero:
            r_str = _format_real(r)
            if i == S.One:
                i_str = 'i'
            elif i == S.NegativeOne:
                i_str = '-i'
            elif i.is_negative:
                i_str = f'{_format_real(i)}i'
            else:
                i_str = f'+{_format_real(i)}i'
            return f'{r_str}{i_str}'
        if r != S.Zero:
            return _format_real(r)
        if i != S.Zero:
            return f'{_format_real(i)}i'
    except Exception:
        pass

    return str(expr)


def _format_mixed(expr):
    """Format an expression containing both traction types and complex numbers."""
    from sympy import I

    if isinstance(expr, Mul):
        parts = []
        for arg in Mul.make_args(expr):
            if arg == I:
                parts.append('i')
            elif arg == -I:
                parts.append('-i')
            elif isinstance(arg, (Zero, Omega, Null)) or (isinstance(arg, Pow) and isinstance(arg.base, (Zero, Omega))):
                parts.append(format_result(arg))
            else:
                parts.append(_format_sympy_complex(arg))
        return '\u00b7'.join(parts)

    if isinstance(expr, Add):
        parts = []
        for i, arg in enumerate(expr.args):
            s = _format_mixed(arg) if (arg.has(Zero) or arg.has(Omega) or arg.has(I)) else _format_real(arg)
            if i > 0 and not s.startswith('-'):
                s = '+' + s
            parts.append(s)
        return ''.join(parts)

    # Fallback
    return format_result(expr)


def _format_real(expr):
    """Format a real SymPy expression concisely."""
    if isinstance(expr, Integer):
        return str(int(expr))
    if isinstance(expr, Rational):
        return f'{expr.p}/{expr.q}'
    try:
        val = float(expr)
        if val == int(val):
            return str(int(val))
        return f'{val:.6g}'
    except (TypeError, ValueError):
        return str(expr)


# ============================================================
# Chebyshev Decomposition
# ============================================================

def _extract_zero_exponent(expr):
    """
    If expr is a zero-power (or equivalent), return the exponent.
    Returns None if not a simple zero-power.
    """
    if isinstance(expr, Zero):
        return S.One
    if isinstance(expr, Omega):
        return S.NegativeOne
    if expr is S.One or expr == Integer(1):
        return S.Zero
    if isinstance(expr, Pow):
        if isinstance(expr.base, Zero):
            return expr.exp
        if isinstance(expr.base, Omega):
            return -expr.exp
    return None


def _collect_exponent_denoms(expr):
    """Collect all exponent denominators from zero/omega powers in an expression."""
    denoms = set()
    expr = traction_simplify(expr)

    if isinstance(expr, (Zero, Omega)):
        denoms.add(1)
    elif isinstance(expr, Pow):
        if isinstance(expr.base, (Zero, Omega)) and isinstance(expr.exp, Rational):
            denoms.add(int(expr.exp.q))
        elif isinstance(expr.base, (Zero, Omega)) and isinstance(expr.exp, Integer):
            denoms.add(1)
        # Recurse into non-traction powers
        if not isinstance(expr.base, (Zero, Omega)):
            denoms.update(_collect_exponent_denoms(expr.base))
    elif isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            denoms.update(_collect_exponent_denoms(arg))

    return denoms


def _expr_to_ring(expr):
    """
    Convert a traction expression to a chebyshev_ring.Element with an
    adaptive generator. Returns (element, info_dict) or (None, None).

    Auto-detects the natural base (zero or omega) and band (rational or omega).
    Priority:
    1. Omega band: g = B^(ω/n), θ-independent universal structure
    2. Rational band: g = B^(1/n), θ-dependent
    3. Multi-band: both generators needed
    """
    from chebyshev_ring import Element, ZERO_SPEC, OMEGA_SPEC, ZERO_OMEGA_SPEC, OMEGA_OMEGA_SPEC
    from fractions import Fraction as Frac
    from math import lcm

    expr = traction_simplify(expr)

    # Classify exponents into omega and rational bands, and detect base
    rational_denoms = set()
    omega_denoms = set()
    bases_seen = set()  # 'zero', 'omega'
    _classify_exponents_full(expr, rational_denoms, omega_denoms, bases_seen)

    # Determine the natural base spec
    if bases_seen == {'omega'}:
        rat_spec = OMEGA_SPEC
        omega_spec = OMEGA_OMEGA_SPEC
    elif bases_seen == {'zero'}:
        rat_spec = ZERO_SPEC
        omega_spec = ZERO_OMEGA_SPEC
    else:
        # Mixed bases or scalars — default to zero
        rat_spec = ZERO_SPEC
        omega_spec = ZERO_OMEGA_SPEC

    # --- Priority 1: Pure omega exponents → omega-band ring ---
    if omega_denoms and not rational_denoms:
        omega_lcd = 1
        for d in omega_denoms:
            omega_lcd = lcm(omega_lcd, d)
        el = _convert_omega_band(expr, omega_lcd)
        if el is not None:
            return el, {'band': 'omega', 'omega_lcd': omega_lcd, 'spec': omega_spec}

    # --- Priority 2: Pure rational exponents → rational-band ring ---
    if rational_denoms or not omega_denoms:
        denoms = _collect_exponent_denoms(expr)
        if not denoms:
            denoms = {1}
        base_denom = 1
        for d in denoms:
            base_denom = lcm(base_denom, d)
        base_denom = max(base_denom, 2)

        el = _convert_with_scale(expr, base_denom)
        if el is not None:
            return el, {'band': 'rational', 'base_denom': base_denom, 'spec': rat_spec}

    # --- Priority 3: Mixed → multi-band ---
    mb_result = _try_multiband(expr)
    if mb_result is not None:
        el, rat_scale, omega_scale = mb_result
        return (el, rat_scale, omega_scale), {'band': 'multi', 'rat_spec': rat_spec, 'omega_spec': omega_spec}
    return None, None


def _classify_exponents_full(expr, rational_denoms, omega_denoms, bases_seen):
    """Classify exponent denominators and detect which bases appear."""
    from fractions import Fraction as Frac
    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational, Null)):
        return
    if isinstance(expr, Zero):
        rational_denoms.add(1)
        bases_seen.add('zero')
        return
    if isinstance(expr, Omega):
        rational_denoms.add(1)
        bases_seen.add('omega')
        return
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        base_name = 'zero' if isinstance(expr.base, Zero) else 'omega'
        bases_seen.add(base_name)
        exp = expr.exp
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            omega_denoms.add(coeff.denominator)
        elif isinstance(exp, (Integer, Rational)):
            rational_denoms.add(int(exp.q) if isinstance(exp, Rational) else 1)
        return
    if isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            _classify_exponents_full(arg, rational_denoms, omega_denoms, bases_seen)
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        _classify_exponents_full(expr.base, rational_denoms, omega_denoms, bases_seen)


def _convert_omega_band(expr, omega_lcd):
    """Convert an expression with omega exponents to Element.
    g = 0^(ω/omega_lcd), scale = omega_lcd.
    0^(ω·p/q) → g^(omega_lcd·p/q)."""
    from chebyshev_ring import Element
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return Element.from_int(int(expr))
    if isinstance(expr, Rational):
        return Element.from_fraction(Frac(expr.p, expr.q))
    if isinstance(expr, Null):
        return Element.zero_el()

    # Zero-powers with omega exponents
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        sign = -1 if isinstance(expr.base, Omega) else 1
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            power = coeff * omega_lcd * sign
            if power.denominator != 1:
                return None
            return Element.u_power(int(power))
        return None

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_omega_band(term, omega_lcd)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_omega_band(factor, omega_lcd)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        base_el = _convert_omega_band(expr.base, omega_lcd)
        if base_el is not None:
            return base_el ** int(expr.exp)

    return None


def _has_omega_exponent(expr):
    """Check if an expression contains zero-powers with omega in the exponent."""
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        return expr.exp.has(Omega) if hasattr(expr.exp, 'has') else False
    if isinstance(expr, (Add, Mul)):
        return any(_has_omega_exponent(arg) for arg in expr.args)
    return False


def _extract_omega_rational(exp):
    """Extract (rational_coeff, omega_power) from an exponent like ω/7 or ω²·3/4.
    Returns (coeff, omega_exp) such that the exponent = coeff * ω^omega_exp,
    or None if it doesn't have this form. For simple ω*c, returns (c, 1)."""
    from fractions import Fraction as Frac

    # ω * rational: Mul(Rational, Omega)
    if isinstance(exp, Mul):
        rational_parts = []
        omega_count = 0
        for f in Mul.make_args(exp):
            if isinstance(f, (Integer, Rational)):
                rational_parts.append(f)
            elif isinstance(f, Omega):
                omega_count += 1
            elif isinstance(f, Pow) and isinstance(f.base, Omega) and isinstance(f.exp, Integer):
                omega_count += int(f.exp)
            else:
                return None
        if omega_count > 0 and rational_parts:
            coeff = Mul(*rational_parts)
            try:
                return Frac(int(coeff.p), int(coeff.q)) if isinstance(coeff, Rational) else Frac(int(coeff)), omega_count
            except (TypeError, ValueError, AttributeError):
                return None
    # Bare omega
    if isinstance(exp, Omega):
        return Frac(1), 1
    return None


def _try_multiband(expr):
    """Try to convert an expression to MultiBandElement.
    Band 1 (g₁): rational exponents. Band 2 (g₂): omega-containing exponents."""
    from chebyshev_ring import MultiBandElement
    from fractions import Fraction as Frac
    from math import lcm

    expr = traction_simplify(expr)

    # Collect LCD for each band
    rational_denoms = set()
    omega_denoms = set()
    _classify_exponents(expr, rational_denoms, omega_denoms)

    if not omega_denoms:
        return None  # no omega exponents, single-band suffices

    # Compute scales for each band
    rat_lcd = 1
    for d in rational_denoms:
        rat_lcd = lcm(rat_lcd, d)
    rat_lcd = max(rat_lcd, 2)

    omega_lcd = 1
    for d in omega_denoms:
        omega_lcd = lcm(omega_lcd, d)

    el = _convert_multiband(expr, rat_lcd, omega_lcd)
    if el is None:
        return None
    return (el, rat_lcd, omega_lcd)


def _classify_exponents(expr, rational_denoms, omega_denoms):
    """Classify exponent denominators into rational and omega-containing bands."""
    from fractions import Fraction as Frac
    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational, Null)):
        return
    if isinstance(expr, (Zero, Omega)):
        rational_denoms.add(1)
        return
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            omega_denoms.add(coeff.denominator)
        elif isinstance(exp, (Integer, Rational)):
            rational_denoms.add(int(exp.q) if isinstance(exp, Rational) else 1)
        return
    if isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            _classify_exponents(arg, rational_denoms, omega_denoms)
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        _classify_exponents(expr.base, rational_denoms, omega_denoms)


def _convert_multiband(expr, rat_scale, omega_scale):
    """Convert a traction expression to MultiBandElement.
    g₁ handles rational exponents (scale=rat_scale), g₂ handles omega exponents (scale=omega_scale)."""
    from chebyshev_ring import MultiBandElement
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return MultiBandElement.from_int(int(expr))
    if isinstance(expr, Rational):
        return MultiBandElement.from_int(0) + MultiBandElement.from_int(int(expr.p)) * Frac(1, int(expr.q))
    if isinstance(expr, Null):
        return MultiBandElement.zero_el()

    # Zero/Omega atoms → band 1
    if isinstance(expr, Zero):
        return MultiBandElement.g1() ** rat_scale
    if isinstance(expr, Omega):
        return MultiBandElement.g1_inv() ** rat_scale

    # Zero-powers
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        sign = -1 if isinstance(expr.base, Omega) else 1

        # Check for omega-containing exponent → band 2
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            power = coeff * omega_scale * sign
            if power.denominator != 1:
                return None
            return MultiBandElement.g2() ** int(power)

        # Rational exponent → band 1
        try:
            r = Frac(int(exp.p), int(exp.q)) if isinstance(exp, Rational) else Frac(int(exp))
        except (TypeError, ValueError, AttributeError):
            return None
        power = r * rat_scale * sign
        if power.denominator != 1:
            return None
        return MultiBandElement.g1() ** int(power)

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_multiband(term, rat_scale, omega_scale)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_multiband(factor, rat_scale, omega_scale)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        base_el = _convert_multiband(expr.base, rat_scale, omega_scale)
        if base_el is not None:
            return base_el ** int(expr.exp)

    return None


def _convert_with_scale(expr, scale):
    """Convert a traction expression to Element using the given scale factor.
    0^(p/q) maps to u^(scale * p/q), which must be an integer."""
    from chebyshev_ring import Element
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return Element.from_int(int(expr))
    if isinstance(expr, Rational):
        return Element.from_fraction(Frac(expr.p, expr.q))
    if isinstance(expr, Null):
        return Element.zero_el()

    # Zero-powers: 0^(p/q) → u^(scale * p/q)
    if isinstance(expr, Zero):
        return Element.u_power(scale)
    if isinstance(expr, Omega):
        return Element.u_power(-scale)
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        try:
            r = Frac(int(exp.p), int(exp.q)) if isinstance(exp, Rational) else Frac(int(exp))
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(expr.base, Omega):
            r = -r
        power = r * scale
        if power.denominator != 1:
            return None
        return Element.u_power(int(power))

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_with_scale(term, scale)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_with_scale(factor, scale)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow):
        base_el = _convert_with_scale(expr.base, scale)
        if base_el is not None and isinstance(expr.exp, Integer):
            return base_el ** int(expr.exp)

    return None


def _decompose_ring_element(ring_el, ring_info, traction_str, complex_str, simplified):
    """Build a decomposition result dict from a chebyshev_ring Element."""
    import re
    import math
    from fractions import Fraction as Frac

    def _relabel(s):
        return re.sub(r'\bu\b', 'g', s)

    ring_str = _relabel(repr(ring_el))

    exponent = None
    exp_zero = _extract_zero_exponent(simplified)
    if exp_zero is not None:
        try:
            exponent = Rational(exp_zero)
        except (TypeError, ValueError):
            pass

    conj = ring_el.conj()
    conj_str = _relabel(repr(conj))
    norm = ring_el.norm()
    norm_str = repr(norm)

    band = ring_info.get('band', 'rational')
    base_denom = ring_info.get('base_denom')
    omega_lcd = ring_info.get('omega_lcd')
    spec = ring_info.get('spec')

    ring_label = f'Q[s][g] / (g\u00b2 \u2212 sg + 1)'

    if band == 'omega':
        n = omega_lcd
        sym = spec.symbol if spec else '0'
        g_str = spec.format_generator(n) if spec else f'0^(\u03c9/{n})'

        # s is exact traction: g + g⁻¹
        s_rational = {1: Frac(-2), 2: Frac(0), 3: Frac(1)}.get(n)
        if n == 2:
            gen_note = f'g = {g_str},  s = 0,  g\u00b2 = -1'
        else:
            gen_note = f'g = {g_str},  s = g + g\u207b\u00b9'

        if s_rational is not None:
            a_val = ring_el.a.eval_at(float(s_rational))
            b_val = ring_el.b.eval_at(float(s_rational))
            a_frac = Frac(a_val.real).limit_denominator(10000) if abs(a_val.imag) < 1e-12 else None
            b_frac = Frac(b_val.real).limit_denominator(10000) if abs(b_val.imag) < 1e-12 else None
            components = [
                ('a', str(a_frac) if a_frac is not None else repr(ring_el.a)),
                ('b', str(b_frac) if b_frac is not None else repr(ring_el.b)),
            ]
        else:
            components = [
                ('a', repr(ring_el.a)),
                ('b', repr(ring_el.b)),
            ]
    else:
        sym = spec.symbol if spec else '0'
        g_str = spec.format_generator(base_denom) if spec else f'0^(1/{base_denom})'
        gen_note = f'g = {g_str},  s = g + g\u207b\u00b9'
        components = [
            ('a', repr(ring_el.a)),
            ('b', repr(ring_el.b)),
        ]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'ring_form': True,
        'ring_str': ring_str,
        'ring_label': ring_label,
        'gen_note': gen_note,
        'components': components,
        'norm_str': norm_str,
        'conj_str': conj_str,
        'is_unit': ring_el.can_invert(),
        'complex_str': complex_str,
        'exponent': exponent,
        'base_denom': base_denom if band == 'rational' else None,
        'band': band,
        'omega_lcd': omega_lcd,
        '_ring_el': ring_el,
        '_ring_info': ring_info,
    }


def _decompose_multiband(ring_el, rat_scale, omega_scale, traction_str, complex_str, simplified):
    """Build decomposition dict for a MultiBandElement."""
    ring_str = repr(ring_el)

    exponent = None
    exp_zero = _extract_zero_exponent(simplified)
    if exp_zero is not None:
        try:
            exponent = Rational(exp_zero)
        except (TypeError, ValueError):
            pass

    norm = ring_el.norm()
    norm_str = repr(norm)
    conj_str = f'\u03c3\u2081: {repr(ring_el.sigma1())}\n  \u03c3\u2082: {repr(ring_el.sigma2())}'

    if rat_scale == 2:
        g1_str = '0^(1/2)'
    else:
        g1_str = f'0^(1/{rat_scale})'

    if omega_scale == 1:
        g2_str = '0^\u03c9'
    else:
        g2_str = f'0^(\u03c9/{omega_scale})'

    components = [
        ('a', repr(ring_el.a)),
        ('b', repr(ring_el.b)),
        ('c', repr(ring_el.c)),
        ('d', repr(ring_el.d)),
    ]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'ring_form': True,
        'ring_str': ring_str,
        'ring_label': 'Q[s\u2081,s\u2082][g\u2081,g\u2082] / (g\u2081\u00b2=s\u2081g\u2081\u22121, g\u2082\u00b2=s\u2082g\u2082\u22121)',
        'gen_note': f'g\u2081 = {g1_str},  g\u2082 = {g2_str},  s\u2081 = g\u2081+g\u2081\u207b\u00b9,  s\u2082 = g\u2082+g\u2082\u207b\u00b9',
        'components': components,
        'norm_str': norm_str,
        'conj_str': conj_str,
        'is_unit': ring_el.can_invert(),
        'complex_str': complex_str,
        'exponent': exponent,
        'base_denom': None,
    }


def _eval_ring_exact(decomp):
    """
    Compute the exact symbolic value of a ring element by substituting
    the traction expressions for g and s, then simplifying.

    Uses Horner's method with traction_simplify at each step to keep
    intermediate expressions manageable.

    Returns a formatted string of the exact traction value.
    """
    from chebyshev_ring import Element, QsPoly
    from sympy import Rational as SRat, Pow

    ring_el = decomp.get('_ring_el')
    ring_info = decomp.get('_ring_info', {})
    band = ring_info.get('band', decomp.get('band', 'rational'))

    if ring_el is None:
        return 'Ring element not available'

    if band == 'omega':
        omega_lcd = ring_info.get('omega_lcd', 2)
        # g = 0^(ω/n), g⁻¹ = 0^(-ω/n), s = g + g⁻¹
        g_expr = Pow(Zero(), Mul(Omega(), SRat(1, omega_lcd)))
        g_inv = Pow(Zero(), Mul(Omega(), SRat(-1, omega_lcd)))
        s_expr = traction_simplify(g_expr + g_inv)
    elif band == 'rational':
        base_denom = ring_info.get('base_denom', decomp.get('base_denom', 2))
        g_expr = Pow(Zero(), SRat(1, base_denom))
        g_inv = Pow(Zero(), SRat(-1, base_denom))
        s_expr = traction_simplify(g_expr + g_inv)
    else:
        return 'Multi-band exact evaluation not yet implemented'

    def eval_poly_horner(poly, x):
        """Evaluate QsPoly at traction expression x using Horner's method.
        Expands and simplifies at each step to keep expressions small."""
        from sympy import expand
        coeffs = poly.coeffs
        if not coeffs:
            return Integer(0)
        result = SRat(coeffs[-1])
        for i in range(len(coeffs) - 2, -1, -1):
            result = traction_simplify(expand(result * x) + SRat(coeffs[i]))
        return result

    a_val = eval_poly_horner(ring_el.a, s_expr)
    b_val = eval_poly_horner(ring_el.b, s_expr)

    if ring_el.b.is_zero():
        exact = a_val
    else:
        from sympy import expand
        exact = traction_simplify(expand(a_val + expand(b_val * g_expr)))

    return format_result(exact)


def _complex_at_pi2(expr):
    """Evaluate a traction expression numerically at θ = π/2.

    Maps:  0 → i,  ω → −i,  0^z → e^{izπ/2}  (with ω → 2 in exponents).
    Returns a Python complex, or None on failure.
    """
    import cmath

    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational)):
        return complex(expr)
    if isinstance(expr, Zero):
        return 1j
    if isinstance(expr, Omega):
        return -1j
    if isinstance(expr, Null):
        return 0.0

    if isinstance(expr, Pow):
        base, exponent = expr.base, expr.exp
        if isinstance(base, (Zero, Omega)):
            sign = 1.0 if isinstance(base, Zero) else -1.0
            try:
                exp_num = complex(exponent.subs(Omega(), Integer(2)).evalf())
            except Exception:
                return None
            return cmath.exp(1j * sign * cmath.pi / 2 * exp_num)
        pb = _complex_at_pi2(base)
        pe = _complex_at_pi2(exponent)
        if pb is None or pe is None:
            return None
        if pb == 0:
            return 0.0
        try:
            return pb ** pe
        except Exception:
            return None

    if isinstance(expr, Mul):
        result = complex(1)
        for arg in Mul.make_args(expr):
            v = _complex_at_pi2(arg)
            if v is None:
                return None
            result *= v
        return result

    if isinstance(expr, Add):
        result = complex(0)
        for arg in Add.make_args(expr):
            v = _complex_at_pi2(arg)
            if v is None:
                return None
            result += v
        return result

    try:
        return complex(expr.evalf())
    except Exception:
        return None


def _chebyshev_relation(n):
    """Compute T_n(s/2) as a polynomial in s (ascending Fraction coefficients).

    Uses the Chebyshev recurrence T_{k+1}(x) = 2x·T_k(x) − T_{k-1}(x)
    with x = s/2, giving T_{k+1}(s/2) = s·T_k(s/2) − T_{k-1}(s/2).

    Returns list of Fraction coefficients [c0, c1, ..., cn].
    """
    from fractions import Fraction as Frac
    if n == 0:
        return [Frac(1)]
    if n == 1:
        return [Frac(0), Frac(1, 2)]
    prev_prev = [Frac(1)]              # T_0(s/2) = 1
    prev = [Frac(0), Frac(1, 2)]      # T_1(s/2) = s/2
    for _ in range(2, n + 1):
        # s · prev  (shift coefficients right by 1)
        s_times = [Frac(0)] + prev[:]
        # subtract prev_prev
        length = max(len(s_times), len(prev_prev))
        cur = []
        for i in range(length):
            a = s_times[i] if i < len(s_times) else Frac(0)
            b = prev_prev[i] if i < len(prev_prev) else Frac(0)
            cur.append(a - b)
        prev_prev = prev
        prev = cur
    return prev


def _poly_mod(p, m):
    """Reduce polynomial p modulo m (both ascending-degree Fraction lists).

    Returns the remainder of p / m with degree < deg(m).
    """
    from fractions import Fraction as Frac
    p = list(p)
    deg_m = len(m) - 1
    lead_m = m[-1]
    while len(p) > deg_m:
        if p[-1] == Frac(0):
            p.pop()
            continue
        factor = p[-1] / lead_m
        offset = len(p) - len(m)
        for i in range(len(m)):
            p[i + offset] -= factor * m[i]
        p.pop()
    while len(p) > 1 and p[-1] == Frac(0):
        p.pop()
    return p


def _eval_frac_poly_horner(coeffs, s_expr):
    """Evaluate a polynomial (ascending-degree Fraction list) at a traction s_expr."""
    from fractions import Fraction as Frac
    from sympy import expand
    # Strip trailing zeros
    while len(coeffs) > 1 and coeffs[-1] == Frac(0):
        coeffs = coeffs[:-1]
    if not coeffs or all(c == Frac(0) for c in coeffs):
        return Integer(0)
    result = Rational(coeffs[-1].numerator, coeffs[-1].denominator)
    for i in range(len(coeffs) - 2, -1, -1):
        c = Rational(coeffs[i].numerator, coeffs[i].denominator)
        result = traction_simplify(expand(result * s_expr) + c)
    return result


def _reduce_ring_form(decomp):
    """
    Reduce the ring element using the Chebyshev identity 0 + ω = ∅
    and substitute g = 0^(1/n) to get a simplified traction expression.

    The identity 0 + ω = ∅ implies T_n(s/2) = 0 where s = g + g⁻¹,
    giving a polynomial relation on s. Reducing the ring polynomials
    mod this relation and substituting back produces a simpler form.

    For n=2: s² = 2, result has ≤ 4 terms (period-8 cycle).
    For n=3: s² = 3, similar structure with 0^(k/3) terms.
    For general n: T_n(s/2) = 0, result has ≤ 2n terms.

    Returns (formatted_string, complex_value) or None if not applicable.
    """
    from fractions import Fraction as Frac

    ring_el = decomp.get('_ring_el')
    ring_info = decomp.get('_ring_info', {})
    band = ring_info.get('band', decomp.get('band', 'rational'))
    base_denom = ring_info.get('base_denom', decomp.get('base_denom', 2))

    if ring_el is None or band != 'rational':
        return None

    n = base_denom

    # Compute the Chebyshev relation T_n(s/2) = 0
    m_poly = _chebyshev_relation(n)

    # Reduce a(s) and b(s) modulo the relation
    a_red = _poly_mod(list(ring_el.a.coeffs), m_poly)
    b_red = _poly_mod(list(ring_el.b.coeffs), m_poly)

    b_is_zero = all(c == Frac(0) for c in b_red)

    # Evaluate reduced polynomials at s = 0^(1/n) + ω^(1/n), g = 0^(1/n)
    s_expr = traction_simplify(
        Pow(Zero(), Rational(1, n)) + Pow(Omega(), Rational(1, n)))
    g_expr = Pow(Zero(), Rational(1, n))

    a_val = _eval_frac_poly_horner(a_red, s_expr)

    if b_is_zero:
        result_expr = a_val
    else:
        from sympy import expand
        b_val = _eval_frac_poly_horner(b_red, s_expr)
        result_expr = traction_simplify(expand(a_val + expand(b_val * g_expr)))

    result_str = format_result(result_expr)
    result_complex = _complex_at_pi2(result_expr)

    return result_str, result_complex


def _step_label(k, denom):
    """Format recurrence step k with denominator denom as a fraction label.
    E.g., k=3, denom=2 → 'a(3/2)'; k=4, denom=2 → 'a(2)'.
    """
    from math import gcd
    g = gcd(k, denom)
    num, den = k // g, denom // g
    if den == 1:
        return f'a({num})'
    return f'a({num}/{den})'


def _cheb_poly_str(n_steps, step_denom=2, var='s'):
    """
    Compute the modified Chebyshev polynomial via recurrence:
    T_0=2, T_1=s, T_n = s*T_{n-1} - T_{n-2}.

    n_steps: number of recurrence iterations.
    step_denom: each step represents 1/step_denom of a full unit.
        (step_denom=2 → half-integer steps, step_denom=16 → 1/16 steps, etc.)
    Returns (formatted_string, trace_list).
    """
    from sympy import Symbol as Sym, expand
    s = Sym(var)

    if n_steps == 0:
        return '2', [(_step_label(0, step_denom), '2')]
    if n_steps == 1:
        return str(s), [(_step_label(0, step_denom), '2'),
                         (_step_label(1, step_denom), str(s))]

    prev2 = Integer(2)
    prev1 = s
    trace = [(_step_label(0, step_denom), '2'),
             (_step_label(1, step_denom), str(s))]

    for k in range(2, n_steps + 1):
        curr = expand(s * prev1 - prev2)
        label = _step_label(k, step_denom)
        poly_str = str(curr).replace('**', '^').replace('*', '\u00b7')
        trace.append((label, poly_str))
        prev2 = prev1
        prev1 = curr

    final_str = str(prev1).replace('**', '^').replace('*', '\u00b7')
    return final_str, trace


def chebyshev_decompose(expr):
    """
    Analyze a traction expression for Chebyshev structure.
    Returns a dict describing the decomposition.
    """
    simplified = traction_simplify(expr)
    traction_str = format_result(simplified)

    # Try to get complex projection (both string and numeric)
    complex_val = None
    try:
        proj = project_complex(simplified)
        complex_str = format_complex(simplified)
        if not complex_str:
            complex_str = str(proj)
        complex_val = complex(proj.evalf())
    except Exception:
        complex_str = ''

    # Check for free symbols (can't decompose symbolically)
    if simplified.free_symbols:
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': 'Contains free variables \u2014 Chebyshev decomposition requires a concrete exponent',
            'complex_str': complex_str,
        }

    # --- Primary path: try to convert to a Chebyshev ring Element ---
    ring_result, ring_info = _expr_to_ring(simplified)
    if ring_result is not None:
        if isinstance(ring_result, tuple):
            # Multi-band: (element, rat_scale, omega_scale)
            el, rat_scale, omega_scale = ring_result
            return _decompose_multiband(el, rat_scale, omega_scale, traction_str, complex_str, simplified)
        else:
            return _decompose_ring_element(ring_result, ring_info, traction_str, complex_str, simplified)

    # --- Fallback: single zero-power analysis ---
    exponent = _extract_zero_exponent(simplified)

    # Handle Mul: scalar * zero-power
    coeff = None
    if exponent is None and isinstance(simplified, Mul):
        scalar_parts = []
        zero_exp = None
        for arg in Mul.make_args(simplified):
            e = _extract_zero_exponent(arg)
            if e is not None and zero_exp is None:
                zero_exp = e
            else:
                scalar_parts.append(arg)
        if zero_exp is not None:
            exponent = zero_exp
            coeff = Mul(*scalar_parts) if scalar_parts else None

    # Handle -1 = 0^2 in the integer cycle
    if exponent is None and simplified == S.NegativeOne:
        exponent = Integer(2)

    if exponent is None:
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': 'Not a zero-power expression',
            'complex_str': complex_str,
        }

    # Check if exponent is numeric
    try:
        exp_val = Rational(exponent)
    except (TypeError, ValueError):
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': f'Symbolic exponent 0^({format_result(exponent)}) \u2014 cannot compute Chebyshev polynomial',
            'complex_str': complex_str,
            'exponent': exponent,
        }

    # Format exponent string
    exp_str = format_result(exp_val)

    # Half-cycle label
    if exp_val.q == 1:
        half_label = str(int(exp_val))
    else:
        half_label = f'{exp_val.p}/{exp_val.q}'

    # Equivalence: what does 0^n simplify to?
    try:
        equiv = traction_simplify(Pow(Zero(), exp_val))
        equiv_str = format_result(equiv)
        if coeff is not None:
            equiv_str = f'{format_result(coeff)}\u00b7{equiv_str}'
    except Exception:
        equiv_str = '?'

    # Chebyshev polynomial: for exponent p/q, we need 2*exponent to be an
    # integer for the standard recurrence (u=0^(1/2), step_denom=2).
    # If not, use a finer grid: u_d = 0^(1/(2q)), step_denom = 2q.
    doubled = exp_val * 2
    if doubled.q == 1:
        # Standard half-integer case: 2*exponent is an integer
        n_steps = abs(int(doubled))
        step_denom = 2
    else:
        # Finer grid needed
        p_num = abs(exp_val.p)
        q_den = exp_val.q
        n_steps = 2 * p_num
        step_denom = 2 * q_den

    cheb_str, trace = _cheb_poly_str(n_steps, step_denom=step_denom)

    # Cap trace display
    if len(trace) > 12:
        trace = trace[:3] + [('...', '...')] + trace[-3:]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'exponent': exp_val,
        'exp_str': exp_str,
        'half_label': half_label,
        'equiv_str': equiv_str,
        'cheb_str': cheb_str,
        'trace': trace,
        'complex_str': complex_str,
        'coeff': coeff,
        'step_denom': step_denom,
    }


# ============================================================
# Phase Visualization
# ============================================================

CANVAS_SIZE = 300
GRID_RES = 150
AXIS_MARGIN = 28  # pixels for tick labels
CANVAS_TOTAL = CANVAS_SIZE + AXIS_MARGIN
DEFAULT_BOUNDS = 3.0

# Phase model: cyan, magenta, yellow, teal (4 quadrants)
PHASE_COLORS = np.array([
    [0, 255, 255],      # cyan    — phase 0        (positive real)
    [255, 0, 255],      # magenta — phase pi/2     (positive imaginary)
    [255, 255, 0],      # yellow  — phase pi       (negative real)
    [0, 128, 128],      # teal    — phase 3*pi/2   (negative imaginary)
], dtype=np.float64)



def compute_phase_grid(expr_text, grid_res=GRID_RES, bounds=3.0,
                       projection_name='complex_lie'):
    """
    Compute visualization data for a traction expression using a registered projection.

    Variable system:
        p, q — raw grid coordinates (horizontal, vertical). Always the same.
        x    — projection's native unit coordinate. Defined by each projection:
               complex_lie: x = p + q*0^(w/2)
               q_surface:   x = 0^(w*p/q)

    Returns tuple (phase, brightness, Z, log_mag) or None on failure.
    """
    parsed = parse_and_eval(expr_text)
    if parsed is None:
        return None

    p_sym = Symbol('p')
    q_sym = Symbol('q')
    x_sym = Symbol('x')
    has_p = parsed.has(p_sym)
    has_q = parsed.has(q_sym)
    has_x = parsed.has(x_sym)

    if not has_p and not has_q and not has_x:
        return None

    # Get the active projection from the registry
    proj = registry.get('projection', projection_name)
    if proj is None:
        return None

    a, b = symbols('a b', real=True)

    # Substitute variables:
    # p -> a (horizontal), q -> b (vertical), x -> projection's native unit
    subs = []
    if has_p:
        subs.append((p_sym, a))
    if has_q:
        subs.append((q_sym, b))
    if has_x:
        subs.append((x_sym, proj.native_x(a, b)))

    traction_expr = parsed.subs(subs)
    traction_expr = traction_simplify(traction_expr)

    # Graded element path: any expression containing Z_n → plot as (r, p) in r + p*w
    if traction_expr.has(GradedElement):
        return _compute_graded_grid(traction_expr, a, b, grid_res, bounds)

    # Step 1: Symbolic projection
    projected = proj.project_expr(traction_expr, a, b)
    if projected is None:
        return None

    # Step 2: Grid evaluation
    lin = np.linspace(-bounds, bounds, grid_res)
    AA, BB = np.meshgrid(lin, lin[::-1])  # flip y so up = positive

    eval_result = proj.eval_grid(projected, a, b, AA, BB,
                                 traction_expr=traction_expr)
    if eval_result is None:
        return None

    return eval_result['phase'], eval_result['brightness'], eval_result['Z'], eval_result['log_mag']


def _degrade(expr):
    """
    Recursively strip graded wrappers from an expression.

    Replaces every GradedElement(n, value) with just value,
    then maps omega -> i so that r + p*w becomes r + p*j.
    """
    from traction import Omega

    if isinstance(expr, GradedElement):
        return _degrade(expr.value)
    if expr.args:
        new_args = [_degrade(a) for a in expr.args]
        try:
            rebuilt = expr.func(*new_args)
        except Exception:
            rebuilt = expr
        return rebuilt.subs(Omega(), I)
    return expr.subs(Omega(), I)


def _split_omega(expr):
    """
    Decompose a traction expression into (r, p) where expr = r + p*w.

    Returns (r_part, w_part) as SymPy expressions.
    For expressions that don't cleanly decompose, returns (expr, S.Zero).
    """
    from traction import Omega

    if isinstance(expr, Omega):
        return S.Zero, S.One

    if isinstance(expr, Add):
        r_terms = []
        w_terms = []
        for arg in expr.args:
            if isinstance(arg, Omega):
                w_terms.append(S.One)
            elif isinstance(arg, Mul):
                # Check if any factor is Omega
                omega_count = 0
                other_factors = []
                for f in Mul.make_args(arg):
                    if isinstance(f, Omega):
                        omega_count += 1
                    else:
                        other_factors.append(f)
                if omega_count == 1:
                    w_terms.append(Mul(*other_factors) if other_factors else S.One)
                elif omega_count == 0:
                    r_terms.append(arg)
                else:
                    # Multiple omegas — treat as r part (w^2 etc.)
                    r_terms.append(arg)
            else:
                r_terms.append(arg)
        r = Add(*r_terms) if r_terms else S.Zero
        w = Add(*w_terms) if w_terms else S.Zero
        return r, w

    if isinstance(expr, Mul):
        omega_count = 0
        other_factors = []
        for f in Mul.make_args(expr):
            if isinstance(f, Omega):
                omega_count += 1
            else:
                other_factors.append(f)
        if omega_count == 1:
            return S.Zero, Mul(*other_factors) if other_factors else S.One
        # No omega or multiple omegas
        return expr, S.Zero

    # Anything else: pure real part
    return expr, S.Zero


def _compute_graded_grid(graded_expr, a, b, grid_res, bounds):
    """
    Compute visualization for an expression containing GradedElements.

    Strips graded wrappers and maps omega -> i in all values,
    so r + p*w becomes r + p*j. Axes: horizontal = r, vertical = p (omega coeff).

    Returns (phase, brightness, Z, log_mag) or None.
    """
    # Recursively unwrap all graded elements and project omega -> i
    projected = _degrade(graded_expr)
    try:
        projected = projected.expand()
    except Exception:
        pass

    try:
        f = lambdify((a, b), projected, modules=['numpy'])
    except Exception:
        return None

    lin = np.linspace(-bounds, bounds, grid_res)
    AA, BB = np.meshgrid(lin, lin[::-1])

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

    # Standard metrics from complex grid
    phase = np.angle(Z)
    phase = (phase + 2 * np.pi) % (2 * np.pi)
    mag = np.abs(Z)
    log_mag = np.log(np.maximum(mag, 1e-300))
    invalid = ~np.isfinite(Z) | (Z == 0)
    phase[invalid] = np.nan
    log_mag[invalid] = np.nan
    brightness = 0.5 + np.arctan(log_mag) / np.pi
    brightness = np.clip(brightness, 0.12, 0.95)
    brightness[invalid] = 0.0

    return phase, brightness, Z, log_mag


def phase_to_rgb(phase_grid, brightness=None):
    """Map a 2D phase grid to an RGB array with smooth quadrant interpolation."""
    # Sanitize: replace NaN/inf with 0 before any casts, track for later
    invalid = ~np.isfinite(phase_grid)
    phase_clean = np.where(invalid, 0.0, phase_grid)

    t = phase_clean / (2 * np.pi)  # normalize to [0, 1)
    segment = t * 4                 # [0, 4)
    idx = segment.astype(int) % 4
    frac = (segment - segment.astype(int))[..., np.newaxis]

    c1 = PHASE_COLORS[idx]
    c2 = PHASE_COLORS[(idx + 1) % 4]
    rgb = c1 + frac * (c2 - c1)  # float [0, 255]

    # Modulate by brightness
    if brightness is not None:
        rgb = rgb * np.where(invalid, 0.0, brightness)[..., np.newaxis]

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Invalid pixels (overflow/underflow) -> solid black
    rgb[invalid] = [0, 0, 0]

    return rgb



def continuity_to_rgb(phase_grid, log_mag):
    """
    Color continuity model: same CMYT palette as phase mode, but follows
    magnitude instead of phase. Double-cover mapping ensures |f|->0 and
    |f|->inf wrap to the same color (continuity at infinity).
    No brightness modulation — pure color from magnitude.
    """
    # Magnitude -> [0, 1) via double cover of arctan(log|f|)
    t_mag = np.arctan(log_mag) / np.pi + 0.5  # [0, 1) full magnitude range
    t_mag = (2 * t_mag) % 1.0                  # double cover
    t_mag = np.where(np.isfinite(t_mag), t_mag, 0)

    # Same 4-color quadrant interpolation as phase mode
    segment = t_mag * 4
    idx = segment.astype(int) % 4
    frac = (segment - segment.astype(int))[..., np.newaxis]

    c1 = PHASE_COLORS[idx]
    c2 = PHASE_COLORS[(idx + 1) % 4]
    rgb = np.clip(c1 + frac * (c2 - c1), 0, 255).astype(np.uint8)

    # NaN -> dark gray
    nan_mask = np.isnan(phase_grid)
    rgb[nan_mask] = [40, 40, 40]

    return rgb


# ============================================================
# Fractal Computation
# ============================================================

def compute_fractal(iter_func, grid_res, bounds, escape=2.0, max_iter=100,
                    cancel_event=None):
    """Compute an escape-time fractal on a complex grid.

    iter_func(x, c) -> next_x : vectorised numpy function for one iteration.
    c is the pixel coordinate.  x starts at traction zero (0), which
    projects to e^{iπ/2} for the numeric computation.  The iteration
    is algebraically identical to traction arithmetic at θ=π/2.

    Returns (counts, last_z, c_grid) where
      counts[r,c] = iteration at which |x| > escape (0 means did not escape)
      last_z[r,c] = value of x at escape (or at max_iter if it didn't)
      c_grid[r,c] = the c value for that pixel.
    """
    lin = np.linspace(-bounds, bounds, grid_res)
    AA, BB = np.meshgrid(lin, lin[::-1])
    c_grid = AA + 1j * BB

    # x starts at traction zero (0).  Under the θ=π/2 projection this
    # is e^{iπ/2} = 1j.  Note: 0 ≠ i = 0^(ω/2) — they are distinct
    # traction elements that share 0² = i² = −1.
    z = np.full_like(c_grid, 1j, dtype=complex)
    counts = np.zeros(c_grid.shape, dtype=int)  # 0 = did not escape
    last_z = np.zeros_like(c_grid, dtype=complex)
    mask = np.ones(c_grid.shape, dtype=bool)     # still iterating

    for n in range(1, max_iter + 1):
        if cancel_event is not None and cancel_event.is_set():
            return None
        try:
            z[mask] = iter_func(z[mask], c_grid[mask])
        except Exception:
            break
        # Clamp to avoid overflow in subsequent iterations
        bad = ~np.isfinite(z)
        z[bad] = escape + 1
        escaped = mask & (np.abs(z) > escape)
        counts[escaped] = n
        last_z[escaped] = z[escaped]
        mask &= ~escaped
        if not mask.any():
            break

    # For pixels that didn't escape, record last z
    last_z[mask] = z[mask]
    return counts, last_z, c_grid


def fractal_to_rgb(counts, last_z, max_iter, escape=2.0):
    """Colour an escape-time fractal.

    Escaped pixels: smooth HSV colouring based on iteration count.
    Interior pixels (didn't escape): black.
    """
    escaped = counts > 0
    rgb = np.zeros(counts.shape + (3,), dtype=np.uint8)

    if not escaped.any():
        return rgb

    # Smooth iteration count for anti-banding
    abs_z = np.abs(last_z[escaped])
    abs_z = np.maximum(abs_z, 1e-30)
    smooth = counts[escaped] + 1 - np.log2(np.log2(abs_z + 1e-30))
    smooth = np.maximum(smooth, 0)

    # Map to hue cycle (repeats every ~32 iterations for visual variety)
    t = smooth / 32.0
    # HSV → RGB with S=0.85, V=1
    hue = (t % 1.0) * 6.0
    sector = hue.astype(int) % 6
    frac = hue - sector

    # Build RGB from HSV sector
    v = np.ones_like(frac)
    s = np.full_like(frac, 0.85)
    p = v * (1 - s)
    q = v * (1 - s * frac)
    tt = v * (1 - s * (1 - frac))

    r = np.where(sector == 0, v, np.where(sector == 1, q, np.where(
        sector == 2, p, np.where(sector == 3, p, np.where(sector == 4, tt, v)))))
    g = np.where(sector == 0, tt, np.where(sector == 1, v, np.where(
        sector == 2, v, np.where(sector == 3, q, np.where(sector == 4, p, p)))))
    b = np.where(sector == 0, p, np.where(sector == 1, p, np.where(
        sector == 2, tt, np.where(sector == 3, v, np.where(sector == 4, v, q)))))

    rgb[escaped, 0] = (r * 255).astype(np.uint8)
    rgb[escaped, 1] = (g * 255).astype(np.uint8)
    rgb[escaped, 2] = (b * 255).astype(np.uint8)

    return rgb


def _parse_fractal_args(text):
    """Parse 'fractal(expr[, escape, maxRecursion])' from raw input text.

    Returns (expr_str, escape, max_iter) or raises ValueError with help message.
    """
    text = text.strip()
    if not text.startswith('fractal(') or not text.endswith(')'):
        raise ValueError(
            "Usage: fractal(expr[, escape, maxIter])\n"
            "  expr:    iteration body\n"
            "           c = pixel coordinate (or use p, q)\n"
            "           x = accumulator, starts at traction 0\n"
            "               (note: 0 \u2260 i. 0\u00b2 = i\u00b2 = \u22121,\n"
            "                but i = 0^(\u03c9/2), a different element)\n"
            "  escape:  escape radius (default 2)\n"
            "  maxIter: max iterations (default 100)\n\n"
            "Examples:\n"
            "  fractal(c + x^2)          — traction Mandelbrot (x\u2080 = 0)\n"
            "  fractal(c + x^2, 4, 200)  — higher escape & iterations\n"
            "  fractal(p + q*0 + x^2)    — same, using p/q coordinates\n"
            "  fractal(c*x - x^3 + c)    — custom formula")

    inner = text[len('fractal('):-1].strip()
    if not inner:
        raise ValueError("fractal() requires an expression argument")

    # Split on top-level commas (respecting parentheses)
    parts = []
    depth = 0
    current = []
    for ch in inner:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    parts.append(''.join(current).strip())

    expr_str = parts[0]
    escape = 2.0
    max_iter = 100

    if len(parts) > 1:
        try:
            escape = float(parts[1])
        except ValueError:
            raise ValueError(f"Invalid escape radius: {parts[1]}")
    if len(parts) > 2:
        try:
            max_iter = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid maxIter: {parts[2]}")

    return expr_str, escape, max_iter


# ============================================================
# Gradient Streamlines
# ============================================================

def compute_streamlines(log_mag, num_seeds=10, steps=200, step_size=0.4, normal=False):
    """
    Trace gradient flow lines of log|f| from a grid of seed points.
    Returns list of streamlines, each a list of (col, row) float pairs.
    Traces both forward and backward from each seed.

    If normal=False: tangent lines (following the gradient, zeros -> poles).
    If normal=True:  normal lines (perpendicular to gradient, constant |f| contours).
    """
    h, w = log_mag.shape
    # Gradient: dy[row,col], dx[row,col]
    gy, gx = np.gradient(log_mag)

    if normal:
        # Rotate 90°: (gx, gy) -> (-gy, gx)
        gx, gy = -gy, gx

    # Seed points: evenly spaced interior grid
    seeds = []
    for si in range(1, num_seeds + 1):
        for sj in range(1, num_seeds + 1):
            r = si * h // (num_seeds + 1)
            c = sj * w // (num_seeds + 1)
            seeds.append((float(r), float(c)))

    lines = []
    for sr, sc in seeds:
        # Trace forward
        fwd = _trace_one(gx, gy, sr, sc, steps, step_size)
        # Trace backward
        bwd = _trace_one(gx, gy, sr, sc, steps, -step_size)
        # Combine: reversed backward + forward
        line = list(reversed(bwd)) + [(sc, sr)] + fwd
        if len(line) >= 3:
            lines.append(line)

    return lines


def _trace_one(gx, gy, start_r, start_c, steps, step_size):
    """Trace a single streamline direction using midpoint (RK2) integration."""
    h, w = gx.shape
    points = []
    r, c = start_r, start_c

    for _ in range(steps):
        # Bail on NaN (from poles/zeros in the function)
        if not (np.isfinite(r) and np.isfinite(c)):
            break
        ri, ci = int(r), int(c)
        if ri < 1 or ri >= h - 1 or ci < 1 or ci >= w - 1:
            break

        # Sample gradient at current point
        vx = gx[ri, ci]
        vy = gy[ri, ci]
        if not (np.isfinite(vx) and np.isfinite(vy)):
            break
        mag = np.sqrt(vx * vx + vy * vy)
        if mag < 1e-12:
            break

        # Normalize
        nx, ny = vx / mag, vy / mag

        # Midpoint step (RK2)
        mr = r + 0.5 * step_size * ny
        mc = c + 0.5 * step_size * nx
        if not (np.isfinite(mr) and np.isfinite(mc)):
            break
        mri, mci = int(mr), int(mc)
        if mri < 1 or mri >= h - 1 or mci < 1 or mci >= w - 1:
            break

        mx = gx[mri, mci]
        my = gy[mri, mci]
        if not (np.isfinite(mx) and np.isfinite(my)):
            break
        mm = np.sqrt(mx * mx + my * my)
        if mm < 1e-12:
            break

        r += step_size * my / mm
        c += step_size * mx / mm
        points.append((c, r))  # (col, row)

    return points


# ============================================================
# GUI
# ============================================================

# Color palette matching the HTML calculator
BG_FRAME = '#808080'
BG_BODY = '#c0c0c0'
BG_DISPLAY = '#bbbbbb'
BG_RESULT = '#d4d4c8'
BG_BTN = '#c8c8c8'
BG_BTN_HOVER = '#e0e0e0'
BG_BTN_ACTIVE = '#999999'
BG_TOGGLE_ON = '#8faab8'
BG_TOGGLE_ON_HOVER = '#a0bcc8'
FG_TEXT = '#111111'
FG_DIM = '#555555'
FG_RESULT = '#1a1a2e'
BORDER_COLOR = '#888888'

# Tab Names
TOWER = 'Orient 3D'


class CalculatorApp:

    def __init__(self, root):
        self.root = root
        root.title('COTT Calculator')
        root.configure(bg=BG_FRAME)
        root.resizable(False, False)

        # Fonts
        self.font_display = tkfont.Font(family='Consolas', size=22)
        self.font_result = tkfont.Font(family='Consolas', size=18)
        self.font_btn = tkfont.Font(family='Segoe UI', size=16)
        self.font_btn_small = tkfont.Font(family='Segoe UI', size=12)
        self.font_label = tkfont.Font(family='Segoe UI', size=10)

        self.history = []
        self.viz_bounds = DEFAULT_BOUNDS
        self.viz_image = None
        self.viz_Z = None  # complex grid for hover readout
        self.viz_log_mag = None  # log magnitude grid for gradient lines
        self._viz_graded = False  # True when plotting a GradedElement
        self.show_tangent = False
        self.show_normal = False
        self.show_diamond = False
        self.color_mode = 'phase'  # 'phase' or 'continuity'
        self.projection_names = registry.names('projection')
        self.projection_index = 0  # default to first registered (complex_lie)

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        # Outer border
        border = tk.Frame(self.root, bg=BG_FRAME, padx=12, pady=12)
        border.pack()

        # ===== Top: Full-width input display =====
        top_frame = tk.Frame(border, bg=BG_BODY, bd=1, relief='solid')
        top_frame.pack(fill='x', pady=(0, 6))

        self.entry_var = tk.StringVar()
        self.entry_var.trace_add('write', lambda *_: self._on_entry_change())
        self.display_expr = tk.Entry(
            top_frame, textvariable=self.entry_var, font=self.font_display,
            bg=BG_DISPLAY, fg=FG_TEXT, justify='left',
            bd=0, highlightthickness=0, insertbackground=FG_TEXT
        )
        self.display_expr.pack(fill='x', padx=10, pady=6)

        result_frame = tk.Frame(top_frame, bg=BG_RESULT, relief='sunken', bd=1)
        result_frame.pack(fill='x')

        tk.Label(result_frame, text='=', font=self.font_btn_small,
                 bg=BG_RESULT, fg=FG_DIM, width=2).pack(side='left', padx=(4, 0))

        self.display_result_var = tk.StringVar()
        self.display_result = tk.Entry(
            result_frame, textvariable=self.display_result_var,
            font=self.font_result, bg=BG_RESULT, fg=FG_RESULT,
            justify='right', bd=0, highlightthickness=0,
            readonlybackground=BG_RESULT, state='readonly'
        )
        self.display_result.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=4)


        # ===== Tab bar =====
        tab_bar = tk.Frame(border, bg=BG_FRAME)
        tab_bar.pack(fill='x', pady=(0, 2))

        self._tab_frames = {}
        self._tab_buttons = {}
        self._active_tab = None

        for tab_name in ['Plot', 'Explain', 'Phase Map', TOWER, 'Help']:
            btn = tk.Button(
                tab_bar, text=tab_name, font=self.font_label,
                bd=0, padx=16, pady=3, bg=BG_FRAME, fg=FG_TEXT,
                activebackground=BG_BODY,
                command=lambda n=tab_name: self._select_tab(n)
            )
            btn.pack(side='left')
            self._tab_buttons[tab_name] = btn

        # Tab content container
        self._tab_container = tk.Frame(border, bg=BG_FRAME)
        self._tab_container.pack(fill='both', expand=True)

        # ===== Plot tab =====
        plot_frame = tk.Frame(self._tab_container, bg=BG_FRAME)
        self._tab_frames['Plot'] = plot_frame

        hframe = tk.Frame(plot_frame, bg=BG_FRAME)
        hframe.pack()

        # ===== Left: Calculator =====
        body = tk.Frame(hframe, bg=BG_BODY, bd=1, relief='solid')
        body.pack(side='left', padx=(0, 6))

        # Action bar
        action_frame = tk.Frame(body, bg=BG_BODY)
        action_frame.pack(anchor='w', pady=(6, 6), padx=(12, 0))

        action_buttons = [
            ('\u232b', self._backspace),      # backspace
            ('\u21ba', self._clear_entry),     # undo/clear entry
            ('C', self._clear_all),
        ]
        for label, cmd in action_buttons:
            self._make_button(action_frame, label, cmd).pack(side='left', padx=1, pady=1)

        # Number pad + operators
        pad_frame = tk.Frame(body, bg=BG_BODY)
        pad_frame.pack(anchor='w', pady=(6, 12), padx=(12, 0))

        # Layout: 5 columns x 5 rows
        layout = [
            # col 0             col 1            col 2            col 3            col 4
            [('(', '('),        (')', ')'),      ('^', '^'),      ('\u00f7', '/'), ('log\u2080','log0(')],
            [('7', '7'),        ('8', '8'),      ('9', '9'),      ('\u00d7', '*'), ('log\u03c9','log\u03c9(')],
            [('4', '4'),        ('5', '5'),      ('6', '6'),      ('\u2013', '-'), ('p', 'p')],
            [('1', '1'),        ('2', '2'),      ('3', '3'),      ('+', '+'),      ('q', 'q')],
            [('0', '0'),        ('\u03c9', '\u03c9'),('i', '0^(\u03c9/2)'),('=', None),  ('x', 'x')],
        ]

        for row_idx, row in enumerate(layout):
            for col_idx, (label, value) in enumerate(row):
                if not label:
                    # Empty placeholder — invisible spacer
                    spacer = tk.Frame(pad_frame, width=70, height=40, bg=BG_BODY)
                    spacer.grid(row=row_idx, column=col_idx, padx=1, pady=1)
                    continue
                if label == '=':
                    btn = self._make_button(pad_frame, label, self._evaluate, accent=True)
                elif value is not None:
                    small = label.startswith('log')
                    btn = self._make_button(pad_frame, label, lambda v=value: self._insert(v), small=small)
                else:
                    btn = self._make_button(pad_frame, label, None)
                btn.grid(row=row_idx, column=col_idx, padx=1, pady=1)

        # ===== Right: Phase Visualization =====
        viz_frame = tk.Frame(hframe, bg=BG_BODY, bd=1, relief='solid')
        viz_frame.pack(side='left', padx=(6, 0), anchor='n')

        self.viz_title_label = tk.Label(viz_frame, text='Phase Plot', font=self.font_label,
                                        bg=BG_BODY, fg=FG_DIM)
        self.viz_title_label.pack(pady=(8, 4))

        self.viz_canvas = tk.Canvas(viz_frame, width=CANVAS_TOTAL, height=CANVAS_TOTAL,
                                    bg='#282828', highlightthickness=0)
        self.viz_canvas.pack(padx=10, pady=(0, 4))
        self.viz_canvas.bind('<Motion>', self._on_viz_hover)
        self.viz_canvas.bind('<Leave>', self._on_viz_leave)

        # Hover gauges: 3 scale boxes (Re, Im, |f|) + 1 phase compass
        GAUGE_H = 72
        GAUGE_W = CANVAS_TOTAL
        self.gauge_canvas = tk.Canvas(viz_frame, width=GAUGE_W, height=GAUGE_H,
                                      bg='#282828', highlightthickness=0)
        self.gauge_canvas.pack(padx=10, pady=(2, 2))

        # Button row: zoom controls
        btn_row = tk.Frame(viz_frame, bg=BG_BODY)
        btn_row.pack(pady=(4, 10))

        self._make_button(btn_row, '\u2013', self._zoom_out, small=True).pack(side='left', padx=1)
        self._make_button(btn_row, '\u25cb', self._zoom_reset, small=True).pack(side='left', padx=1)
        self._make_button(btn_row, '+', self._zoom_in, small=True).pack(side='left', padx=1)
        self._make_button(btn_row, '\u2699', self._open_settings, small=True).pack(side='left', padx=(8, 1))


        # ===== Explain tab (Chebyshev Decomposition) =====
        explain_frame = tk.Frame(self._tab_container, bg=BG_BODY, bd=1, relief='solid')
        self._tab_frames['Explain'] = explain_frame

        # Horizontal split: scrollable text (left) + orbit plot (right)
        explain_hframe = tk.Frame(explain_frame, bg=BG_BODY)
        explain_hframe.pack(fill='both', expand=True, padx=6, pady=6)

        # Left: scrollable text widget
        text_frame = tk.Frame(explain_hframe, bg='#1a1a2e', bd=1, relief='sunken')
        text_frame.pack(side='left', fill='both', expand=True, padx=(0, 4))

        self.cheb_text = tk.Text(
            text_frame, wrap='word', font=tkfont.Font(family='Consolas', size=11),
            bg='#1a1a2e', fg='#eeeeee', insertbackground='#eeeeee',
            selectbackground='#3a5a8a', selectforeground='#ffffff',
            bd=0, highlightthickness=0, padx=12, pady=10,
            state='disabled', cursor='arrow'
        )
        cheb_scrollbar = tk.Scrollbar(text_frame, command=self.cheb_text.yview)
        self.cheb_text.configure(yscrollcommand=cheb_scrollbar.set)
        cheb_scrollbar.pack(side='right', fill='y')
        self.cheb_text.pack(side='left', fill='both', expand=True)

        # Text tags for styling
        self.cheb_text.tag_configure('header', font=tkfont.Font(family='Segoe UI', size=13, weight='bold'),
                                      foreground='#ffdd33', spacing1=8, spacing3=2)
        self.cheb_text.tag_configure('value', font=tkfont.Font(family='Consolas', size=12),
                                      foreground='#ffffff', spacing1=2, spacing3=2)
        self.cheb_text.tag_configure('dim', font=tkfont.Font(family='Consolas', size=10),
                                      foreground='#8899aa', spacing1=2, spacing3=2)
        self.cheb_text.tag_configure('expr', font=tkfont.Font(family='Consolas', size=14, weight='bold'),
                                      foreground='#ffffff', spacing1=4, spacing3=6)

        # Right: orbit plot (matplotlib)
        self.cheb_fig = Figure(figsize=(3.2, 4.2), dpi=96, facecolor=FG_RESULT)
        self.cheb_fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.08)
        self.cheb_ax_orbit = self.cheb_fig.add_subplot(1, 1, 1)

        self.cheb_canvas_widget = FigureCanvasTkAgg(self.cheb_fig, master=explain_hframe)
        self.cheb_canvas_widget.get_tk_widget().pack(side='left', fill='y', padx=(4, 0))

        # ===== Phase Map tab =====
        phasemap_frame = tk.Frame(self._tab_container, bg=BG_BODY, bd=1, relief='solid')
        self._tab_frames['Phase Map'] = phasemap_frame

        # Controls row
        pm_controls = tk.Frame(phasemap_frame, bg=BG_BODY)
        pm_controls.pack(fill='x', padx=10, pady=(8, 4))

        tk.Label(pm_controls, text='N:', font=self.font_label,
                 bg=BG_BODY, fg=FG_DIM).pack(side='left')
        self.pm_n_var = tk.StringVar(value='10')
        tk.Entry(pm_controls, textvariable=self.pm_n_var, width=4,
                 font=self.font_label).pack(side='left', padx=(2, 12))

        tk.Label(pm_controls, text='\u03b8 / \u03c0:', font=self.font_label,
                 bg=BG_BODY, fg=FG_DIM).pack(side='left')
        self.pm_theta_var = tk.DoubleVar(value=0.25)
        self.pm_theta_scale = tk.Scale(
            pm_controls, variable=self.pm_theta_var, from_=0.01, to=0.99,
            resolution=0.01, orient='horizontal', length=200,
            bg=BG_BODY, fg=FG_TEXT, highlightthickness=0, troughcolor=BG_DISPLAY,
            command=lambda _: self._update_phase_map()
        )
        self.pm_theta_scale.pack(side='left', padx=(2, 12))

        tk.Button(pm_controls, text='Refresh', font=self.font_btn_small,
                  bd=1, padx=8, bg='#5a7d9a', fg='white',
                  activebackground=BG_BTN_ACTIVE,
                  command=self._update_phase_map).pack(side='right', padx=4)

        # Matplotlib figure
        self.pm_fig = Figure(figsize=(8.5, 4.5), dpi=96, facecolor=BG_BODY)
        self.pm_fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.12, wspace=0.32)

        self.pm_ax_orbit = self.pm_fig.add_subplot(1, 3, 1)
        self.pm_ax_amp = self.pm_fig.add_subplot(1, 3, 2)
        self.pm_ax_table = self.pm_fig.add_subplot(1, 3, 3)

        self.pm_canvas_widget = FigureCanvasTkAgg(self.pm_fig, master=phasemap_frame)
        self.pm_canvas_widget.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=(0, 8))

        # ===== Tower tab (Level 2 interactive visualization) =====
        tower_frame = tk.Frame(self._tab_container, bg=BG_BODY, bd=1, relief='solid')
        self._tab_frames[TOWER] = tower_frame

        tower_hframe = tk.Frame(tower_frame, bg=BG_BODY)
        tower_hframe.pack(fill='both', expand=True, padx=6, pady=6)

        # Left: sphere canvas
        SPHERE_SIZE = 340
        self.tower_canvas = tk.Canvas(
            tower_hframe, width=SPHERE_SIZE, height=SPHERE_SIZE,
            bg='#1a1a2e', highlightthickness=0
        )
        self.tower_canvas.pack(side='left', padx=(0, 6))

        # Right: controls + readout
        tower_right = tk.Frame(tower_hframe, bg=BG_BODY)
        tower_right.pack(side='left', fill='both', expand=True)

        # θ_w slider (outer rotation — w level)
        tk.Label(tower_right, text='\u03b8w / \u03c0  (w = 0^{1/4})', font=self.font_label,
                 bg=BG_BODY, fg=FG_DIM).pack(anchor='w', padx=4, pady=(8, 0))
        self.tower_tw_var = tk.DoubleVar(value=0.25)
        tk.Scale(
            tower_right, variable=self.tower_tw_var, from_=0.0, to=2.0,
            resolution=0.01, orient='horizontal', length=250,
            bg=BG_BODY, fg=FG_TEXT, highlightthickness=0, troughcolor=BG_DISPLAY,
            command=lambda _: self._update_tower()
        ).pack(anchor='w', padx=4)

        # θ_u slider (inner rotation — u level)
        tk.Label(tower_right, text='\u03b8u / \u03c0  (u = 0^{1/2})', font=self.font_label,
                 bg=BG_BODY, fg=FG_DIM).pack(anchor='w', padx=4, pady=(8, 0))
        self.tower_tu_var = tk.DoubleVar(value=0.25)
        tk.Scale(
            tower_right, variable=self.tower_tu_var, from_=0.0, to=2.0,
            resolution=0.01, orient='horizontal', length=250,
            bg=BG_BODY, fg=FG_TEXT, highlightthickness=0, troughcolor=BG_DISPLAY,
            command=lambda _: self._update_tower()
        ).pack(anchor='w', padx=4)

        # Readout text
        self.tower_text = tk.Text(
            tower_right, wrap='word', font=tkfont.Font(family='Consolas', size=11),
            bg='#1a1a2e', fg='#eeeeee', insertbackground='#eeeeee',
            selectbackground='#3a5a8a', selectforeground='#ffffff',
            bd=1, relief='sunken', highlightthickness=0, padx=10, pady=8,
            state='disabled', cursor='arrow', height=12
        )
        self.tower_text.pack(fill='both', expand=True, padx=4, pady=(8, 0))
        self.tower_text.tag_configure('header', font=tkfont.Font(family='Segoe UI', size=12, weight='bold'),
                                       foreground='#ffdd33', spacing1=4, spacing3=2)
        self.tower_text.tag_configure('value', font=tkfont.Font(family='Consolas', size=11),
                                       foreground='#ffffff', spacing1=1, spacing3=1)
        self.tower_text.tag_configure('dim', font=tkfont.Font(family='Consolas', size=10),
                                       foreground='#8899aa', spacing1=1, spacing3=1)

        self._tower_sphere_size = SPHERE_SIZE

        # ===== Help tab =====
        self._build_help_tab()

        # Show Plot tab by default
        self._select_tab('Plot')

    # ===== Help Tab =====

    def _build_help_tab(self):
        """Build the Help tab with interactive example buttons."""
        HELP_BG = '#0d0d1a'  # dark background for readability

        help_frame = tk.Frame(self._tab_container, bg=HELP_BG, bd=1, relief='solid')
        self._tab_frames['Help'] = help_frame

        # Scrollable area using Canvas + Frame with fixed minimum size
        help_canvas = tk.Canvas(help_frame, bg=HELP_BG, highlightthickness=0, bd=0,
                                width=760, height=520)
        help_scrollbar = tk.Scrollbar(help_frame, orient='vertical', command=help_canvas.yview)
        help_inner = tk.Frame(help_canvas, bg=HELP_BG)

        help_inner.bind('<Configure>', lambda e: help_canvas.configure(scrollregion=help_canvas.bbox('all')))
        self._help_window_id = help_canvas.create_window((0, 0), window=help_inner, anchor='nw')
        help_canvas.configure(yscrollcommand=help_scrollbar.set)

        # Keep inner frame as wide as the canvas
        def _sync_inner_width(event):
            help_canvas.itemconfigure(self._help_window_id, width=event.width)
        help_canvas.bind('<Configure>', _sync_inner_width)

        help_scrollbar.pack(side='right', fill='y')
        help_canvas.pack(side='left', fill='both', expand=True)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            help_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        help_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        # Unbind when leaving Help tab to avoid conflicts
        self._help_canvas = help_canvas
        self._help_mousewheel_handler = _on_mousewheel

        font_section = tkfont.Font(family='Segoe UI', size=13, weight='bold')
        font_desc = tkfont.Font(family='Consolas', size=10)
        font_example = tkfont.Font(family='Consolas', size=11, weight='bold')

        def add_section(title, description, examples):
            """Add a section with header, description, and example buttons."""
            tk.Label(help_inner, text=title, font=font_section,
                     bg=HELP_BG, fg='#ffdd33', anchor='w').pack(
                fill='x', padx=16, pady=(14, 2))
            if description:
                tk.Label(help_inner, text=description, font=font_desc,
                         bg=HELP_BG, fg='#aabbcc', anchor='w', justify='left',
                         wraplength=700).pack(fill='x', padx=20, pady=(0, 6))
            for expr, hint, tab in examples:
                row = tk.Frame(help_inner, bg=HELP_BG)
                row.pack(fill='x', padx=20, pady=2)
                btn = tk.Button(
                    row, text=expr, font=font_example,
                    bg='#161630', fg='#88ccff', activebackground='#2a3a5a',
                    activeforeground='#aaddff', bd=1, relief='solid',
                    padx=10, pady=3, anchor='w', cursor='hand2',
                    command=lambda e=expr, t=tab: self._run_example(e, t)
                )
                btn.pack(side='left')
                btn.bind('<Enter>', lambda ev, b=btn: b.configure(bg='#2a3a5a'))
                btn.bind('<Leave>', lambda ev, b=btn: b.configure(bg='#161630'))
                if hint:
                    tk.Label(row, text=hint, font=font_desc,
                             bg=HELP_BG, fg='#8899bb', anchor='w').pack(
                        side='left', padx=(10, 0))

        # ---- Welcome ----
        tk.Label(help_inner, text='Welcome to the COTT Calculator',
                 font=tkfont.Font(family='Segoe UI', size=16, weight='bold'),
                 bg=HELP_BG, fg='#ffffff').pack(padx=16, pady=(14, 2), anchor='w')
        tk.Label(help_inner, font=font_desc, bg=HELP_BG, fg='#aabbcc',
                 anchor='w', justify='left', wraplength=700,
                 text='Click any example below to load it into the calculator and see the result. '
                      'Each example switches to the relevant tab automatically.').pack(
            fill='x', padx=20, pady=(0, 6))

        # ---- Foundations ----
        add_section('\u2460  Traction Foundations',
                    'In traction algebra, 0 and \u03c9 are not ordinary numbers. '
                    '0\u00b2 = \u22121, \u03c9\u00b2 = 0, and 0\u00b7\u03c9 = 1. '
                    'These identities replace imaginary numbers with pure algebra.',
                    [
                        ('0^2',           '0 squared = \u22121',                  'Explain'),
                        ('\u03c9^2',      '\u03c9 squared = 0',                   'Explain'),
                        ('0 * \u03c9',    'zero times omega = 1',                 'Explain'),
                        ('0^(1/2)',       'square root of traction zero',          'Explain'),
                        ('0^(\u03c9/2)', 'the imaginary unit i, algebraically',   'Explain'),
                        ('0^0',           'any base to its own zero = 1',         'Explain'),
                    ])

        # ---- Arithmetic ----
        add_section('\u2461  Arithmetic & Algebra',
                    'Standard operations work, but division by zero produces \u03c9-scaled '
                    'results instead of errors. Subtraction to zero yields null (\u2205).',
                    [
                        ('3 / 0',                     'division by zero \u2192 3\u00b7\u03c9',      'Plot'),
                        ('5 - 5',                     'self-cancellation \u2192 null',               'Plot'),
                        ('solve(x^2 + 1)',            'x\u00b2 = \u22121 has traction solutions',   'Plot'),
                        ('solve(x^2 - 4, x)',         'standard quadratic roots',                    'Plot'),
                        ('expand((x + 1)^3)',         'polynomial expansion',                        'Plot'),
                        ('factor(x^3 - x)',           'pulls out common factors',                    'Plot'),
                    ])

        # ---- Phase Visualization ----
        add_section('\u2462  Phase Plots',
                    'The plot tab colors the (p, q) plane by the phase and magnitude of the '
                    'expression. Cyan/Magenta/Yellow/Teal mark the four quadrants.',
                    [
                        ('p + q*0^(\u03c9/2)',        'complex plane: p + qi',                       'Plot'),
                        ('p^2 - q^2',                 'hyperbolic saddle',                           'Plot'),
                        ('p^3 - 3*p*q^2',             'real part of (p+qi)\u00b3  \u2014 harmonic',  'Plot'),
                        ('1 / (p + q*0^(\u03c9/2))',  'pole at the origin',                          'Plot'),
                        ('(p + q*0^(\u03c9/2))^5',    'quintic: five-fold phase winding',            'Plot'),
                    ])

        # ---- Traction Exponentials ----
        add_section('\u2463  Traction Exponentials',
                    '0^z is the traction exponential \u2014 it maps algebra to geometry. '
                    'Under complex-Lie projection, 0^z \u2192 e^{\u2212Wz} where W\u00b2 = \u2212i\u03c0.',
                    [
                        ('0^(p + q*0^(\u03c9/2))',    'traction plane wave e^{-W(p+qi)}',            'Plot'),
                        ('0^p * \u03c9^q',            'mixed 0/\u03c9 exponential field',            'Plot'),
                        ('0^(p*\u03c9)',              'de Broglie phase: momentum \u2192 wave',       'Plot'),
                        ('log0(p + q*0^(\u03c9/2))',  'traction logarithm \u2014 inverse of 0^z',    'Plot'),
                        ('log\u03c9(p + q*0^(\u03c9/2))', 'omega-base logarithm',                    'Plot'),
                    ])

        # ---- Physics & Relativity ----
        add_section('\u2464  Physics & Relativity',
                    'Traction algebra encodes relativistic and quantum structure natively. '
                    'The null element arises from light-cone cancellation; 0^z encodes phase.',
                    [
                        ('p^2 - q^2',                         'Minkowski interval  s\u00b2 = t\u00b2 \u2212 x\u00b2',  'Plot'),
                        ('(p + q)*(p - q)',                    'light-cone factorization',                                 'Plot'),
                        ('0^(p + q) + 0^(p - q)',             'standing wave (superposition)',                             'Plot'),
                        ('0^(p*q)',                            'interaction phase \u2014 p\u00b7q coupling',               'Plot'),
                        ('(p + q*0^(\u03c9/2))^2 + 1',       'Dirac-like: z\u00b2 + 1 with traction twist',              'Plot'),
                    ])

        # ---- Graded Elements (Z-Action) ----
        add_section('\u2465  Graded Elements  (Z-Action)',
                    'Z_n(a) lives at grade n. Addition drops grade (Z_n + Z_n = Z_{n\u22121}), '
                    'multiplication raises it. This inverts the usual algebra/geometry relationship.',
                    [
                        ('Z_2(3) + Z_2(5)',           'grade drops: Z_1(15)',                        'Explain'),
                        ('Z_2(3) * Z_2(5)',           'grade rises: Z_3(8)',                         'Explain'),
                        ('Z_1(7) - Z_1(2)',           'subtraction \u2192 Z_0(7/2)',                 'Explain'),
                        ('Z_3(p + q)',                'graded field on the plane',                   'Plot'),
                    ])

        # ---- Chebyshev Ring ----
        add_section('\u2466  Chebyshev Ring Structure',
                    'The Explain tab decomposes expressions into the Chebyshev ring \u211a[s][g]/(g\u00b2 \u2212 sg + 1), '
                    'where s\u00b2 = 2 recovers trigonometric identities purely algebraically.',
                    [
                        ('0^(1/3)',                   'third root \u2192 Chebyshev decomposition',    'Explain'),
                        ('0^(1/4)',                   'fourth root \u2192 ring element w',            'Explain'),
                        ('0^(2/3) + 0^(1/3)',         'sum of roots in the ring',                    'Explain'),
                        ('0^(1/6)',                   'sixth root structure',                         'Explain'),
                    ])

        # ---- Fractals ----
        add_section('\u2467  Fractals',
                    'fractal(expr) iterates x \u2192 expr starting from traction zero, '
                    'using each pixel as the parameter c (or p, q).',
                    [
                        ('fractal(x^2 + c)',              'Mandelbrot set',                          'Plot'),
                        ('fractal(x^3 + c)',              'cubic Mandelbrot',                        'Plot'),
                        ('fractal(x^2 + c, 4, 200)',     'higher iteration Mandelbrot',              'Plot'),
                        ('fractal(x^2 * \u03c9 + c)',    'omega-twisted Mandelbrot',                 'Plot'),
                        ('fractal(x^2 + x*\u03c9 + c)',  'traction-perturbed iteration',             'Plot'),
                    ])

        # ---- Signal Analysis ----
        add_section('\u2468  Signal Analysis & Waves',
                    'Traction exponentials act as Fourier-like basis functions. '
                    'Superpositions and interference patterns emerge from the algebra.',
                    [
                        ('0^p + 0^(2*p)',                      'harmonic superposition (1st + 2nd)',   'Plot'),
                        ('0^p * 0^q',                          'separable product: 0^{p+q}',          'Plot'),
                        ('0^p + 0^(\u03c9*p)',                 'forward + conjugate wave',             'Plot'),
                        ('(0^p - 0^(\u03c9*p)) / (0^(\u03c9/2) - 0^(1/2))',
                                                               'sine-like envelope',                   'Plot'),
                    ])

        # ---- Orient 3D ----
        add_section('\u2469  Tower Visualization',
                    'The Orient 3D tab shows the four-component tower basis (a + bt + cw + dtw). '
                    'Use the sliders to explore the sphere of orientations.',
                    [
                        ('0^(1/4)',                   'quarter-turn: fundamental tower element w',   TOWER),
                        ('0^(1/2)',                   'half-turn: the traction square root',         TOWER),
                        ('0^(3/4)',                   'three-quarter turn',                          TOWER),
                    ])

        # ---- Phase Map ----
        add_section('\u246a  Phase Map Orbits',
                    'The Phase Map tab traces discrete orbits of 0^{n/2} and \u03c9^{n/2} on the unit circle, '
                    'showing how algebraic powers generate geometric rotations.',
                    [
                        ('0^(1/3)',                   'period-6 orbit of the third root',             'Phase Map'),
                        ('0^(1/5)',                   'period-10 orbit \u2014 pentagonal symmetry',   'Phase Map'),
                        ('\u03c9^(1/4)',              'omega quarter-root orbit',                     'Phase Map'),
                    ])

        # Bottom padding
        tk.Frame(help_inner, bg=HELP_BG, height=20).pack()

    def _run_example(self, expr, tab):
        """Load an example expression, evaluate it, and switch to the target tab."""
        # Unbind mousewheel from help canvas before switching
        self._help_canvas.unbind_all('<MouseWheel>')
        self.entry_var.set(expr)
        self.display_expr.icursor('end')
        self._evaluate()
        self._select_tab(tab)
        self.display_expr.focus_set()

    def _make_button(self, parent, label, command, accent=False, small=False, toggle=False):
        """Create a button. If toggle=True, uses chisel bevel style for toggle switches."""
        bg = '#5a7d9a' if accent else BG_BTN
        fg = 'white' if accent else FG_TEXT
        relief = 'raised'
        bd = 2 if toggle else 1
        btn = tk.Button(
            parent, text=label, font=self.font_btn_small if small else self.font_btn,
            width=4, height=1, bd=bd, relief=relief,
            bg=bg, fg=fg, activebackground=BG_BTN_ACTIVE,
            command=command
        )
        btn._is_toggle = toggle
        btn._is_accent = accent
        btn._toggled = False
        btn.bind('<Enter>', lambda e, b=btn: self._btn_enter(b))
        btn.bind('<Leave>', lambda e, b=btn: self._btn_leave(b))
        return btn

    def _btn_enter(self, btn):
        if btn._toggled:
            btn.configure(bg=BG_TOGGLE_ON_HOVER)
        elif btn._is_accent:
            btn.configure(bg='#6a9dba')
        else:
            btn.configure(bg=BG_BTN_HOVER)

    def _btn_leave(self, btn):
        if btn._toggled:
            btn.configure(bg=BG_TOGGLE_ON)
        elif btn._is_accent:
            btn.configure(bg='#5a7d9a')
        else:
            btn.configure(bg=BG_BTN)

    def _set_toggle(self, btn, state):
        """Set a toggle button's visual state."""
        btn._toggled = state
        if state:
            btn.configure(relief='sunken', bg=BG_TOGGLE_ON, fg=FG_TEXT, bd=2)
        else:
            btn.configure(relief='raised', bg=BG_BTN, fg=FG_TEXT, bd=2)

    def _bind_keys(self):
        self.display_expr.bind('<Return>', lambda e: self._evaluate())
        self.display_expr.bind('<Escape>', lambda e: self._clear_all())
        # Intercept 'w' to insert ω instead
        self.display_expr.bind('w', self._insert_omega)
        # p, q, x pass through as-is (default Entry behavior)
        self.display_expr.focus_set()

    def _insert_omega(self, event):
        self.display_expr.insert('insert', '\u03c9')
        return 'break'  # prevent default 'w' insertion

    def _insert_i(self, event):
        self.display_expr.insert('insert', '0^(\u03c9/2)')
        return 'break'  # prevent default 'i' insertion

    def _insert(self, value):
        """Insert value at the current cursor position in the entry."""
        self.display_expr.insert('insert', value)
        self.display_expr.focus_set()

    def _backspace(self):
        """Delete the character before the current cursor position."""
        cursor_pos = self.display_expr.index('insert')
        if cursor_pos > 0:
            self.display_expr.delete(cursor_pos - 1, cursor_pos)
        self.display_expr.focus_set()

    def _clear_entry(self):
        self.entry_var.set('')

    def _format_display_result(self, parsed):
        """Format the result for display, applying x-substitution."""
        # SolutionSet results: display directly, no x-substitution
        if isinstance(parsed, SolutionSet):
            return format_result(parsed)

        from sympy import Symbol as Sym
        x_sym = Sym('x')
        p_sym = Sym('p')
        q_sym = Sym('q')

        result = parsed

        # If expression contains x, substitute with the active projection's native_x
        if result.has(x_sym):
            proj = registry.get('projection', self.projection_names[self.projection_index])
            if proj is not None:
                native = proj.native_x(p_sym, q_sym)
                result = result.subs(x_sym, native)
                result = traction_simplify(result)

        return f'= {format_result(result)}'

    def _set_result_text(self, text, fg=None):
        """Set the result display text (Entry widget)."""
        self.display_result.configure(state='normal')
        self.display_result_var.set(text)
        if fg:
            self.display_result.configure(fg=fg)
        self.display_result.configure(state='readonly')
        # Scroll to end so the rightmost content is visible
        self.display_result.xview_moveto(1.0)

    def _clear_all(self):
        self.entry_var.set('')
        self.history = []
        self._update_live_preview()

    def _evaluate(self):
        expr_text = self.entry_var.get().strip()
        if not expr_text:
            return

        # Fractal mode: fractal(expr[, escape, maxIter])
        if expr_text.lower().startswith('fractal'):
            try:
                self._handle_fractal(expr_text)
            except (ValueError, Exception) as e:
                self._set_result_text(str(e), fg='#aa3333')
            return

        try:
            result = parse_and_eval(expr_text)
            display_text = self._format_display_result(result)
            self._set_result_text(display_text, fg=FG_RESULT)

            # Refresh the phase plot (keeps expression in entry)
            self._refresh_viz()

            # Refresh explain tab if active
            if self._active_tab == 'Explain':
                self._run_explain()

        except (ParseError, Exception) as e:
            self._set_result_text(f'Error: {e}', fg='#aa3333')

    # ===== Fractal Rendering =====

    def _handle_fractal(self, text):
        """Parse a fractal(...) command and launch background computation.

        x starts at traction zero (0).  Under the θ=π/2 projection used for
        rasterisation, 0 projects to e^{iπ/2} — the same magnitude-1 point
        as i = 0^(ω/2), but a distinct traction element.  The numeric
        iteration is algebraically equivalent to traction arithmetic.
        Supports both 'c' and 'p'/'q' for the pixel coordinate.
        """
        import threading
        from sympy import Symbol, lambdify, I as symI

        expr_str, escape, max_iter = _parse_fractal_args(text)

        # Parse the inner expression
        c_sym = Symbol('c')
        x_sym = Symbol('x')
        p_sym = Symbol('p')
        q_sym = Symbol('q')
        try:
            parsed = Parser(expr_str).parse()
        except Exception as e:
            raise ValueError(f'Parse error in fractal expression: {e}')

        # Support p/q as alternative to c: replace p + q·i → c
        if parsed.has(p_sym) or parsed.has(q_sym):
            parsed = parsed.subs(p_sym, (c_sym + c_sym.conjugate()) / 2)  # nah, just direct
            # Simpler: p = Re(c), q = Im(c), so substitute p→Re(c), q→Im(c)
            # But for lambdify it's easier: substitute at the numpy level
            # Let's just replace p → c_real, q → c_imag via a wrapper
            pass  # handled below in the wrapper

        # Project traction types to θ=π/2: Zero→i, Omega→-i, Null→0
        projected = parsed.subs(Zero(), symI).subs(Omega(), -symI).subs(Null(), 0)

        # Determine which variables the expression uses
        has_c = projected.has(c_sym)
        has_pq = projected.has(p_sym) or projected.has(q_sym)

        if has_pq:
            # Replace p→Re(c), q→Im(c) by lambdifying with p,q then wrapping
            from sympy import re as sym_re, im as sym_im
            projected_pq = projected.subs(p_sym, sym_re(c_sym)).subs(q_sym, sym_im(c_sym))
            try:
                f_raw = lambdify((x_sym, c_sym), projected_pq, modules='numpy')
            except Exception as e:
                raise ValueError(f'Cannot compile fractal expression: {e}')
        elif has_c:
            try:
                f_raw = lambdify((x_sym, c_sym), projected, modules='numpy')
            except Exception as e:
                raise ValueError(f'Cannot compile fractal expression: {e}')
        else:
            raise ValueError(
                'Fractal expression must use c (pixel coordinate) '
                'or p/q (real/imaginary parts)')

        # Quick smoke test
        try:
            f_raw(np.array([1j]), np.array([0.1 + 0.1j]))
        except Exception as e:
            raise ValueError(f'Cannot evaluate fractal expression: {e}')

        # Store raw text for re-render on zoom
        self._fractal_raw_text = text

        self._set_result_text(
            f'fractal: {expr_str}  (x\u2080=0, esc={escape}, n={max_iter})', fg=FG_DIM)

        # Switch to Plot tab
        if self._active_tab != 'Plot':
            self._select_tab('Plot')

        # Cancel any previous fractal computation
        if hasattr(self, '_fractal_cancel'):
            self._fractal_cancel.set()

        cancel = threading.Event()
        self._fractal_cancel = cancel

        # Store fractal metadata for hover (don't clear old data yet)
        self._fractal_mode = True
        self._fractal_escape = escape
        self._fractal_max_iter = max_iter
        self._fractal_expr_str = expr_str

        self.viz_title_label.configure(
            text=f'Fractal [{expr_str}]  computing\u2026')

        def compute():
            result = compute_fractal(
                f_raw, GRID_RES, self.viz_bounds,
                escape=escape, max_iter=max_iter,
                cancel_event=cancel)
            if cancel.is_set() or result is None:
                return
            self.root.after(0, lambda: self._finish_fractal(result, cancel))

        t = threading.Thread(target=compute, daemon=True)
        t.start()

    def _finish_fractal(self, result, cancel):
        """Display a completed fractal computation."""
        if cancel.is_set():
            return

        counts, last_z, c_grid = result
        self._fractal_counts = counts
        self._fractal_last_z = last_z
        self._fractal_c_grid = c_grid

        # Store a dummy viz_Z so hover works (using last_z for coordinate mapping)
        self.viz_Z = last_z
        self.viz_log_mag = None

        escape = self._fractal_escape
        max_iter = self._fractal_max_iter
        expr_str = self._fractal_expr_str

        rgb = fractal_to_rgb(counts, last_z, max_iter, escape)
        self._render_viz(rgb)

        escaped_pct = np.count_nonzero(counts) / counts.size * 100
        self.viz_title_label.configure(
            text=f'Fractal [{expr_str}]  ({escaped_pct:.0f}% escaped)')
        self._set_result_text(
            f'fractal: {expr_str}  (x\u2080=0, esc={escape}, n={max_iter})', fg=FG_RESULT)

    def _on_entry_change(self):
        """Live preview: evaluate as you type."""
        self._update_live_preview()

    def _update_live_preview(self):
        try:
            expr_text = self.entry_var.get().strip()
            if expr_text:
                result = parse_and_eval(expr_text)
                display_text = self._format_display_result(result)
                self._set_result_text(display_text, fg=FG_DIM)
            else:
                self._set_result_text('')
        except Exception:
            self._set_result_text('...', fg=FG_DIM)

    # ===== Visualization =====

    def _refresh_viz(self):
        """Compute and render the phase plot for the current expression."""
        self._fractal_mode = False  # Clear fractal mode for normal plots
        expr_text = self.entry_var.get().strip()
        if not expr_text:
            self.viz_canvas.delete('all')
            self.display_expr.focus_set()
            return

        try:
            proj_name = self.projection_names[self.projection_index]
            result = compute_phase_grid(expr_text, bounds=self.viz_bounds,
                                       projection_name=proj_name)
            if result is None:
                # Expression has no plottable variables — show Explain tab instead
                if self._active_tab == 'Plot':
                    self._select_tab('Explain')
                return

            # Detect graded expression for title label and hover mode
            graded_label = None
            parsed = parse_and_eval(expr_text)
            if parsed is not None and parsed.has(GradedElement):
                graded_label = 'Z-action : r + p\u00b7\u03c9'
                self._viz_graded = True
            elif isinstance(parsed, GradedElement):
                graded_label = f'Z_{parsed.grade} : r + p\u00b7\u03c9'
                self._viz_graded = True
            else:
                self._viz_graded = False
            self._apply_viz_result(result, graded_label=graded_label)
        except Exception:
            self.viz_canvas.delete('all')
        self.display_expr.focus_set()

    def _apply_viz_result(self, result, graded_label=None):
        """Apply a completed visualization result (phase, brightness, Z, log_mag)."""
        phase, brightness, Z, log_mag = result
        self.viz_Z = Z
        self.viz_log_mag = log_mag

        if graded_label:
            self.viz_title_label.configure(text=f'Phase Plot [{graded_label}]')
        else:
            proj_name = self.projection_names[self.projection_index]
            label = proj_name.replace('_', ' ')
            self.viz_title_label.configure(text=f'Phase Plot [{label}]')

        if self.color_mode == 'continuity':
            rgb = continuity_to_rgb(phase, log_mag)
        else:
            rgb = phase_to_rgb(phase, brightness)
        self._render_viz(rgb)

    def _render_viz(self, rgb):
        """Render an RGB numpy array to the visualization canvas."""
        h, w, _ = rgb.shape
        zoom = max(1, CANVAS_SIZE // w)

        img = tk.PhotoImage(width=w, height=h)

        # Build image row by row
        for y in range(h):
            row = ' '.join(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in rgb[y])
            img.put(f'{{{row}}}', to=(0, y))

        # Zoom up to canvas size and display
        # Keep reference to base image — tkinter GCs it otherwise, causing missing pixels
        self._viz_image_base = img
        self.viz_image = img.zoom(zoom, zoom)
        self.viz_canvas.delete('all')
        self.viz_canvas.create_image(AXIS_MARGIN, AXIS_MARGIN, anchor='nw', image=self.viz_image)

        # Draw axis markings
        self._draw_axes()

        # Draw overlays
        if self.show_diamond:
            self._draw_diamond_grid()
        if self.viz_log_mag is not None:
            if self.show_tangent:
                self._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            if self.show_normal:
                self._draw_flow_lines(normal=True, tag='normal', color='#aaaaaa')

    def _draw_axes(self):
        """Draw tick marks and labels along the top and left edges."""
        bounds = self.viz_bounds
        tick_font = tkfont.Font(family='Consolas', size=8)

        # Compute nice tick interval
        tick_step = _nice_tick_step(bounds)
        ticks = []
        val = -bounds + (tick_step - (-bounds % tick_step)) % tick_step
        while val <= bounds:
            ticks.append(round(val, 10))
            val += tick_step

        for val in ticks:
            # Top edge (real axis)
            x = AXIS_MARGIN + (val + bounds) / (2 * bounds) * CANVAS_SIZE
            self.viz_canvas.create_line(x, AXIS_MARGIN - 4, x, AXIS_MARGIN, fill='#aaaaaa')
            label = _tick_label(val)
            self.viz_canvas.create_text(x, AXIS_MARGIN - 6, text=label,
                                        fill='#aaaaaa', font=tick_font, anchor='s')

            # Left edge (imaginary axis) — y is flipped
            y = AXIS_MARGIN + (bounds - val) / (2 * bounds) * CANVAS_SIZE
            self.viz_canvas.create_line(AXIS_MARGIN - 4, y, AXIS_MARGIN, y, fill='#aaaaaa')
            self.viz_canvas.create_text(AXIS_MARGIN - 6, y, text=label,
                                        fill='#aaaaaa', font=tick_font, anchor='e')

    def _draw_diamond_grid(self):
        """Draw a diamond grid overlay at simple fraction intervals.

        Diagonal lines where p+q = n and p-q = n form diamonds.
        Additional lines at 1/2, 1/3, 2/3, 3/4 subdivisions.
        """
        bounds = self.viz_bounds
        fractions = []
        if bounds < 25:
            fractions.append(0)
        if bounds < 16:
            fractions.append(1/2)
        if bounds < 8:
            fractions.extend([1/4, 3/4])
        if bounds < 2:
            fractions.extend([1/3, 2/3])
        # Generate all grid values: integer + fraction offsets
        values = set()
        int_min = int(-bounds) - 1
        int_max = int(bounds) + 1
        for n in range(int_min, int_max + 1):
            for f in fractions:
                v = float(n + f)
                if -bounds <= v <= bounds:
                    values.add(v)
                v = float(n - f)
                if -bounds <= v <= bounds:
                    values.add(v)

        def p2cx(p):
            return AXIS_MARGIN + (p + bounds) / (2 * bounds) * CANVAS_SIZE

        def q2cy(q):
            return AXIS_MARGIN + (bounds - q) / (2 * bounds) * CANVAS_SIZE

        # Clip boundaries
        x_min = AXIS_MARGIN
        x_max = AXIS_MARGIN + CANVAS_SIZE
        y_min = AXIS_MARGIN
        y_max = AXIS_MARGIN + CANVAS_SIZE

        for s in sorted(values):
            # Integer lines are brighter, fraction lines dimmer
            is_int = abs(s - round(s)) < 1e-9
            color = '#555555' if is_int else '#333333'
            width = 1

            # Diagonal p + q = s: line from (s-q, q) as q varies
            # At q = -bounds: p = s+bounds, cx = p2cx(s+bounds), cy = q2cy(-bounds)
            # At q = +bounds: p = s-bounds, cx = p2cx(s-bounds), cy = q2cy(+bounds)
            x1, y1 = p2cx(s - (-bounds)), q2cy(-bounds)
            x2, y2 = p2cx(s - bounds), q2cy(bounds)
            # Clip to canvas
            pts = _clip_line(x1, y1, x2, y2, x_min, y_min, x_max, y_max)
            if pts:
                self.viz_canvas.create_line(*pts, fill=color, width=width, tags='diamond')

            # Diagonal p - q = s: line from (s+q, q) as q varies
            x1, y1 = p2cx(s + (-bounds)), q2cy(-bounds)
            x2, y2 = p2cx(s + bounds), q2cy(bounds)
            pts = _clip_line(x1, y1, x2, y2, x_min, y_min, x_max, y_max)
            if pts:
                self.viz_canvas.create_line(*pts, fill=color, width=width, tags='diamond')

    def _draw_flow_lines(self, normal, tag, color):
        """Draw tangent or normal flow lines on the canvas."""
        lines = compute_streamlines(self.viz_log_mag, normal=normal)
        h, w = self.viz_log_mag.shape
        scale_x = CANVAS_SIZE / w
        scale_y = CANVAS_SIZE / h

        for line in lines:
            if len(line) < 3:
                continue
            coords = []
            for col, row in line:
                cx = AXIS_MARGIN + col * scale_x
                cy = AXIS_MARGIN + row * scale_y
                coords.extend([cx, cy])
            self.viz_canvas.create_line(
                *coords, fill=color, width=1, smooth=True, tags=tag
            )

    def _toggle_tangent(self):
        """Toggle tangent lines (gradient flow) on/off."""
        self.show_tangent = not self.show_tangent
        if self.viz_Z is not None:
            if self.show_tangent:
                self._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            else:
                self.viz_canvas.delete('tangent')
        self.display_expr.focus_set()

    def _toggle_normal(self):
        """Toggle normal lines (constant |f| contours) on/off."""
        self.show_normal = not self.show_normal
        if self.viz_Z is not None:
            if self.show_normal:
                self._draw_flow_lines(normal=True, tag='normal', color='#aaaaaa')
            else:
                self.viz_canvas.delete('normal')
        self.display_expr.focus_set()

    def _toggle_color_mode(self):
        """Toggle between Phase and Continuity color models."""
        if self.color_mode == 'phase':
            self.color_mode = 'continuity'
        else:
            self.color_mode = 'phase'
        if self.viz_Z is not None:
            self._refresh_viz()
        self.display_expr.focus_set()

    def _cycle_projection(self):
        """Cycle through registered projection plugins."""
        self.projection_index = (self.projection_index + 1) % len(self.projection_names)
        name = self.projection_names[self.projection_index]
        label = name.replace('_', ' ')
        if hasattr(self, 'viz_title_label'):
            self.viz_title_label.configure(text=f'Phase Plot [{label}]')
        if self.viz_Z is not None:
            self._refresh_viz()
        self.display_expr.focus_set()

    def _open_settings(self):
        """Open the settings window (single-instance)."""
        SettingsWindow(self)

    # ===== Tab Navigation =====

    def _select_tab(self, name):
        """Switch between Plot and Explain tabs."""
        if self._active_tab == name:
            return
        # Hide current
        if self._active_tab and self._active_tab in self._tab_frames:
            self._tab_frames[self._active_tab].pack_forget()
            self._tab_buttons[self._active_tab].configure(bg=BG_FRAME)
        # Show new
        self._tab_frames[name].pack(fill='both', expand=True)
        self._tab_buttons[name].configure(bg=BG_BODY)
        self._active_tab = name
        # Auto-run explain when switching to the Explain tab
        if name == 'Explain':
            self._run_explain()
        elif name == 'Phase Map':
            self._update_phase_map()
        elif name == TOWER:
            self._update_tower()
        # Re-bind mousewheel for Help tab scrolling
        if name == 'Help' and hasattr(self, '_help_mousewheel_handler'):
            self._help_canvas.bind_all('<MouseWheel>', self._help_mousewheel_handler)
        elif hasattr(self, '_help_canvas'):
            self._help_canvas.unbind_all('<MouseWheel>')

    # ===== Tower Tab =====

    def _update_tower(self):
        """Update the Tower tab sphere visualization and readout."""
        theta_w = self.tower_tw_var.get() * np.pi
        theta_u = self.tower_tu_var.get() * np.pi

        # Sphere position from two independent angles
        sx = np.cos(theta_w)
        sy = np.sin(theta_w) * np.cos(theta_u)
        sz = np.sin(theta_w) * np.sin(theta_u)

        self._draw_tower_sphere(sx, sy, sz)

        # Compute 4 tower components (a, b, c, d) from the sphere point.
        # We want a + bt + cw + dtw to represent this orientation.
        # Use the sphere coords as a 4D unit vector projected to 3D:
        #   a = cos(θ_w/2) * cos(θ_u/2)
        #   b = cos(θ_w/2) * sin(θ_u/2)
        #   c = sin(θ_w/2) * cos(θ_u/2)
        #   d = sin(θ_w/2) * sin(θ_u/2)
        # This ensures a² + b² + c² + d² = 1 (unit 4-sphere).
        a = np.cos(theta_w / 2) * np.cos(theta_u / 2)
        b = np.cos(theta_w / 2) * np.sin(theta_u / 2)
        c = np.sin(theta_w / 2) * np.cos(theta_u / 2)
        d = np.sin(theta_w / 2) * np.sin(theta_u / 2)

        # Convert to traction expression:
        # a + bt + cw + dtw
        # = (a+d) + (b+c)*0^(1/4) + b*0^(-1/4) + d*0^(1/2)
        const = a + d
        coeff_quarter = b + c      # coefficient of 0^(1/4)
        coeff_neg_quarter = b       # coefficient of 0^(-1/4)
        coeff_half = d              # coefficient of 0^(1/2)

        # Build display string for the traction expression
        traction_parts = []
        if abs(const) > 1e-10:
            traction_parts.append(f'{const:.4f}')
        if abs(coeff_quarter) > 1e-10:
            traction_parts.append(f'{coeff_quarter:+.4f}\u00b70^(1/4)')
        if abs(coeff_neg_quarter) > 1e-10:
            traction_parts.append(f'{coeff_neg_quarter:+.4f}\u00b70^(-1/4)')
        if abs(coeff_half) > 1e-10:
            traction_parts.append(f'{coeff_half:+.4f}\u00b70^(1/2)')
        if not traction_parts:
            traction_parts.append('0')

        traction_expr = ' '.join(traction_parts)
        # Clean up leading +
        if traction_expr.startswith('+'):
            traction_expr = traction_expr[1:]

        # Update readout
        tw = self.tower_text
        tw.configure(state='normal')
        tw.delete('1.0', 'end')

        tw.insert('end', 'Traction Expression\n', 'header')
        tw.insert('end', f'  {traction_expr}\n', 'value')

        tw.insert('end', '\n')
        tw.insert('end', 'Tower Components  a + bt + cw + dtw\n', 'header')
        tw.insert('end', f'  a = {a:+.6f}\n', 'value')
        tw.insert('end', f'  b = {b:+.6f}\n', 'value')
        tw.insert('end', f'  c = {c:+.6f}\n', 'value')
        tw.insert('end', f'  d = {d:+.6f}\n', 'value')

        tw.insert('end', '\n')
        tw.insert('end', 'Basis Expansion\n', 'header')
        tw.insert('end', '  t  = 0^(1/4) + 0^(-1/4)\n', 'dim')
        tw.insert('end', '  w  = 0^(1/4)\n', 'dim')
        tw.insert('end', '  tw = 0^(1/2) + 1\n', 'dim')

        tw.insert('end', '\n')
        tw.insert('end', 'Angles\n', 'header')
        tw.insert('end', f'  \u03b8w = {theta_w / np.pi:.4f}\u03c0  ({np.degrees(theta_w):.1f}\u00b0)\n', 'value')
        tw.insert('end', f'  \u03b8u = {theta_u / np.pi:.4f}\u03c0  ({np.degrees(theta_u):.1f}\u00b0)\n', 'value')

        tw.insert('end', '\n')
        tw.insert('end', 'Sphere\n', 'header')
        tw.insert('end', f'  x = {sx:+.4f}   y = {sy:+.4f}   z = {sz:+.4f}\n', 'value')

        tw.configure(state='disabled')

    def _draw_tower_sphere(self, dot_x, dot_y, dot_z):
        """Draw a sphere with a dot at (x, y, z) on it.
        Orthographic projection: screen_x = y, screen_y = -x (so x-axis goes into screen).
        Dot size scales with depth, opacity drops when behind."""
        canvas = self.tower_canvas
        canvas.delete('all')
        S = self._tower_sphere_size
        cx, cy = S / 2, S / 2
        R = S * 0.4  # sphere radius in pixels

        # Draw wireframe sphere
        wire_color = '#333355'

        # Equator (xz plane, projected as horizontal ellipse)
        pts = []
        for i in range(101):
            a = 2 * np.pi * i / 100
            px = cx + R * np.cos(a)
            py = cy + R * np.sin(a) * 0.3  # foreshortened
            pts.extend([px, py])
        canvas.create_line(*pts, fill=wire_color, width=1, smooth=True)

        # Front meridian (xy plane)
        pts = []
        for i in range(101):
            a = 2 * np.pi * i / 100
            px = cx + R * np.cos(a) * 0.3
            py = cy + R * np.sin(a)
            pts.extend([px, py])
        canvas.create_line(*pts, fill=wire_color, width=1, smooth=True)

        # Side meridian (yz plane — the "visible" great circle)
        pts = []
        for i in range(101):
            a = 2 * np.pi * i / 100
            px = cx + R * np.cos(a)
            py = cy + R * np.sin(a)
            pts.extend([px, py])
        canvas.create_line(*pts, fill=wire_color, width=1, smooth=True)

        # Axis lines through center
        canvas.create_line(cx - R * 1.15, cy, cx + R * 1.15, cy, fill='#222244', width=1)
        canvas.create_line(cx, cy - R * 1.15, cx, cy + R * 1.15, fill='#222244', width=1)

        # Axis labels
        label_font = tkfont.Font(family='Consolas', size=9)
        canvas.create_text(cx + R * 1.2, cy + 2, text='y', fill='#555577', font=label_font, anchor='w')
        canvas.create_text(cx + 2, cy - R * 1.2, text='x', fill='#555577', font=label_font, anchor='s')
        canvas.create_text(cx - 12, cy + R * 0.35, text='z', fill='#555577', font=label_font, anchor='e')

        # Project dot: screen coords from (x, y, z)
        # Orthographic: screen_x = y, screen_y = -x, depth = z
        screen_x = cx + dot_y * R
        screen_y = cy - dot_x * R

        # Depth-based size: larger when z > 0 (front), smaller when z < 0 (back)
        # Range: z in [-1, 1] → dot_radius in [4, 16]
        min_r, max_r = 4, 16
        dot_r = min_r + (max_r - min_r) * (dot_z + 1) / 2

        # Depth-based color: bright when front, dim when behind
        if dot_z >= 0:
            # Front face: full brightness
            r_c, g_c, b_c = 0, 200, 255
            alpha = 1.0
        else:
            # Back face: 50% opacity (blend with background #1a1a2e)
            alpha = 0.5
            r_c = int(0 * alpha + 0x1a * (1 - alpha))
            g_c = int(200 * alpha + 0x1a * (1 - alpha))
            b_c = int(255 * alpha + 0x2e * (1 - alpha))

        dot_color = f'#{r_c:02x}{g_c:02x}{b_c:02x}'

        # Draw a faint line from center to dot (projected)
        canvas.create_line(cx, cy, screen_x, screen_y, fill='#333355', width=1, dash=(3, 3))

        # Draw the dot
        canvas.create_oval(
            screen_x - dot_r, screen_y - dot_r,
            screen_x + dot_r, screen_y + dot_r,
            fill=dot_color, outline='#ffffff' if dot_z >= 0 else '#555577',
            width=2 if dot_z >= 0 else 1
        )

        # Draw a small "shadow" dot on the equator plane for depth reference
        shadow_x = cx + dot_y * R
        shadow_y = cy - dot_x * R * 0.3  # projected onto equator
        canvas.create_oval(
            shadow_x - 2, shadow_y - 2,
            shadow_x + 2, shadow_y + 2,
            fill='#333355', outline=''
        )

    # ===== Phase Map Tab =====

    def _update_phase_map(self):
        """Compute and render the Chebyshev phase map visualization."""
        try:
            N = int(self.pm_n_var.get())
        except ValueError:
            N = 10
        N = max(1, min(N, 50))

        theta = self.pm_theta_var.get() * np.pi  # θ in radians
        t = 2 * np.cos(theta)  # u + v = 2cos(θ)

        # u, v are complex conjugates on the unit circle: e^{±iθ}
        u_val = np.exp(1j * theta)
        v_val = np.exp(-1j * theta)

        # Compute half-integer indices from -N to N
        half_indices = np.arange(-2 * N, 2 * N + 1)  # n values; index = n/2
        x_vals = half_indices / 2.0  # n/2 values

        # u^n = e^{inθ}  (0^{n/2})
        zero_powers = np.array([u_val ** n for n in half_indices])
        # v^n = e^{-inθ}  (w^{n/2})
        omega_powers = np.array([v_val ** n for n in half_indices])
        # a_{n/2} = u^n + v^n = 2cos(nθ)
        a_sums = zero_powers + omega_powers  # real-valued (imaginary parts cancel)

        # --- Subplot 1: Phase Orbit (complex plane) ---
        ax1 = self.pm_ax_orbit
        ax1.clear()
        ax1.set_facecolor('#1a1a2e')
        ax1.set_title('Phase Orbit', fontsize=10, color='#cccccc')
        ax1.set_xlabel('Re', fontsize=8, color='#999999')
        ax1.set_ylabel('Im', fontsize=8, color='#999999')
        ax1.tick_params(colors='#888888', labelsize=7)
        ax1.set_aspect('equal')

        # Unit circle reference
        circ_t = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(np.cos(circ_t), np.sin(circ_t), color='#333344', linewidth=0.8, zorder=0)
        ax1.axhline(0, color='#333344', linewidth=0.5)
        ax1.axvline(0, color='#333344', linewidth=0.5)

        # Plot 0^{n/2} orbit in blue
        ax1.plot(zero_powers.real, zero_powers.imag, 'o-',
                 color='#4488ff', markersize=3, linewidth=0.8, alpha=0.7, label='$0^{n/2}$')
        # Plot w^{n/2} orbit in red
        ax1.plot(omega_powers.real, omega_powers.imag, 'o-',
                 color='#ff4444', markersize=3, linewidth=0.8, alpha=0.7, label='$\\omega^{n/2}$')

        # Highlight n=0 (both = 1)
        ax1.plot(1, 0, 'o', color='#ffffff', markersize=6, zorder=5)

        ax1.legend(fontsize=7, loc='upper left', facecolor='#1a1a2e',
                   edgecolor='#444444', labelcolor='#cccccc')

        # --- Subplot 2: Amplitude Plot ---
        ax2 = self.pm_ax_amp
        ax2.clear()
        ax2.set_facecolor('#1a1a2e')
        ax2.set_title('Half-Cycle Sums  $a_{n/2}$', fontsize=10, color='#cccccc')
        ax2.set_xlabel('n / 2', fontsize=8, color='#999999')
        ax2.set_ylabel('Value', fontsize=8, color='#999999')
        ax2.tick_params(colors='#888888', labelsize=7)

        # Re(0^{n/2}) and Re(w^{n/2})
        ax2.plot(x_vals, zero_powers.real, '-', color='#4488ff', linewidth=0.8,
                 alpha=0.5, label='Re($0^{n/2}$)')
        ax2.plot(x_vals, omega_powers.real, '-', color='#ff4444', linewidth=0.8,
                 alpha=0.5, label='Re($\\omega^{n/2}$)')
        # Sum a_{n/2} = 2cos(nθ)
        ax2.plot(x_vals, a_sums.real, 'o-', color='#ffffff', markersize=3,
                 linewidth=1.2, label='$a_{n/2}$', zorder=3)
        ax2.axhline(0, color='#333344', linewidth=0.5)

        # Highlight symmetry: mark integer vs half-integer
        int_mask = (half_indices % 2 == 0)
        ax2.plot(x_vals[int_mask], a_sums.real[int_mask], 's',
                 color='#ffcc00', markersize=4, zorder=4, label='Integer n/2')

        ax2.legend(fontsize=6, loc='upper right', facecolor='#1a1a2e',
                   edgecolor='#444444', labelcolor='#cccccc')

        # --- Subplot 3: Equivalence Table ---
        ax3 = self.pm_ax_table
        ax3.clear()
        ax3.set_facecolor('#1a1a2e')
        ax3.set_title('Equivalences', fontsize=10, color='#cccccc')
        ax3.axis('off')

        # Build table data for integer powers
        table_n = list(range(-5, 9))
        rows = []
        for n in table_n:
            half = n / 2.0
            val = 2 * np.cos(n * theta)
            label = f'{half:+.1f}' if half != int(half) else f'{int(half):+d}'
            rows.append([label, f'{val:+.3f}'])

        table = ax3.table(
            cellText=rows,
            colLabels=['n/2', '$a_{n/2}$'],
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for key, cell in table.get_celld().items():
            cell.set_facecolor('#1a1a2e')
            cell.set_edgecolor('#444466')
            cell.set_text_props(color='#cccccc')
            if key[0] == 0:  # header row
                cell.set_facecolor('#2a2a4e')
                cell.set_text_props(color='#ffcc00', fontweight='bold')

        self.pm_fig.set_facecolor(BG_BODY)
        self.pm_canvas_widget.draw()

    # ===== Explain Tab (Chebyshev Decomposition) =====

    def _run_explain(self):
        """Compute and display the Chebyshev decomposition of the current expression."""
        tw = self.cheb_text
        tw.configure(state='normal')
        tw.delete('1.0', 'end')

        expr_text = self.entry_var.get().strip()
        if not expr_text:
            tw.insert('end', 'Enter an expression', 'dim')
            tw.configure(state='disabled')
            self._draw_cheb_orbit(self.cheb_ax_orbit, decomp=None)
            self.cheb_canvas_widget.draw()
            return


        try:
            parsed = parse_and_eval(expr_text)
        except Exception as e:
            tw.insert('end', f'Parse error: {e}', 'value')
            tw.configure(state='disabled')
            self._draw_cheb_orbit(self.cheb_ax_orbit, decomp=None)
            self.cheb_canvas_widget.draw()
            return

        # Z-action path: rich diagnostic for any expression containing GradedElement
        if isinstance(parsed, GradedElement) or (hasattr(parsed, 'has') and parsed.has(GradedElement)):
            self._render_graded_explain(parsed)
            tw.configure(state='disabled')
            self._draw_cheb_orbit(self.cheb_ax_orbit, decomp=None)
            self.cheb_canvas_widget.draw()
            return

        decomp = chebyshev_decompose(parsed)
        # Store parsed expression for θ=π/2 projection
        decomp['_parsed'] = parsed
        decomp['_pi2_val'] = _complex_at_pi2(parsed)
        # Ensure complex_val is available for the orbit plot
        if 'complex_val' not in decomp:
            try:
                proj = project_complex(traction_simplify(parsed))
                decomp['complex_val'] = complex(proj.evalf())
            except Exception:
                decomp['complex_val'] = None
        self._render_cheb_text(decomp)
        tw.configure(state='disabled')
        self._draw_cheb_orbit(self.cheb_ax_orbit, decomp)
        self.cheb_canvas_widget.draw()

        # Start exact symbolic evaluation in background thread
        if decomp.get('ring_form'):
            self._start_exact_eval(decomp)

    def _start_exact_eval(self, decomp):
        """Launch background thread to compute exact symbolic value from ring element."""
        import threading

        # Cancel any previous computation
        if hasattr(self, '_exact_eval_cancel'):
            self._exact_eval_cancel.set()

        cancel = threading.Event()
        self._exact_eval_cancel = cancel

        # Show "computing..." in the text widget
        tw = self.cheb_text
        tw.configure(state='normal')
        tw.insert('end', '\n')
        tw.insert('end', 'Exact Value\n', 'header')
        tw.insert('end', '  computing...\n', 'dim')
        self._exact_eval_insert_pos = tw.index('end - 1 lines linestart')
        tw.configure(state='disabled')

        def compute():
            try:
                result = _eval_ring_exact(decomp)
                if cancel.is_set():
                    return
                # Schedule UI update on the main thread
                self.root.after(0, lambda: self._finish_exact_eval(result, cancel))
            except Exception as e:
                if cancel.is_set():
                    return
                self.root.after(0, lambda: self._finish_exact_eval(f'Error: {e}', cancel))

        t = threading.Thread(target=compute, daemon=True)
        t.start()

    def _finish_exact_eval(self, result, cancel):
        """Update the Explain tab with the exact evaluation result."""
        if cancel.is_set():
            return

        tw = self.cheb_text
        tw.configure(state='normal')

        # Find and replace the "computing..." line
        try:
            pos = self._exact_eval_insert_pos
            # Delete the "computing..." line (header is already in place)
            tw.delete(pos, 'end')
            if isinstance(result, str) and result.startswith('Error'):
                tw.insert('end', f'  {result}\n', 'dim')
            else:
                tw.insert('end', f'  {result}\n', 'value')
        except Exception:
            pass

        tw.configure(state='disabled')

    def _render_graded_explain(self, parsed):
        """Render Z-action diagnostic information into the Explain text widget."""
        from traction import Omega, Zero, traction_simplify
        from sympy import Add, Mul, Pow as SPow, I as symI, Symbol, Number, Rational

        tw = self.cheb_text
        traction_str = format_result(parsed)

        tw.insert('end', 'Z-Action Expression\n', 'header')
        tw.insert('end', f'  {traction_str}\n', 'expr')
        tw.insert('end', '\n')

        # Collect all GradedElement nodes in the expression tree
        graded_nodes = []
        def _collect_graded(e):
            if isinstance(e, GradedElement):
                graded_nodes.append(e)
            if hasattr(e, 'args'):
                for a in e.args:
                    _collect_graded(a)
        _collect_graded(parsed)

        # --- Structure summary ---
        tw.insert('end', 'Structure\n', 'header')
        if isinstance(parsed, GradedElement):
            tw.insert('end', f'  Top-level: Z_{format_result(parsed.grade)}(value)\n', 'value')
            tw.insert('end', f'  Grade:     {format_result(parsed.grade)}\n', 'value')
            tw.insert('end', f'  Value:     {format_result(parsed.value)}\n', 'value')
        else:
            tw.insert('end', f'  Top-level: {type(parsed).__name__} containing {len(graded_nodes)} graded node(s)\n', 'value')
            for i, g in enumerate(graded_nodes):
                tw.insert('end', f'  [{i+1}] Z_{format_result(g.grade)}({format_result(g.value)})\n', 'value')
        tw.insert('end', '\n')

        # --- Free symbols ---
        free = parsed.free_symbols
        if free:
            tw.insert('end', 'Free Variables\n', 'header')
            tw.insert('end', f'  {", ".join(str(s) for s in sorted(free, key=str))}\n', 'value')
            tw.insert('end', '\n')

        # --- Operation rules applied ---
        tw.insert('end', 'Z-Action Rules\n', 'header')
        tw.insert('end', '  Z_n(a) + Z_n(b) = Z_{n\u22121}(a\u00b7b)\n', 'dim')
        tw.insert('end', '  Z_n(a) \u00b7 Z_n(b) = Z_{n+1}(a+b)\n', 'dim')
        tw.insert('end', '  Z_n(a) \u2212 Z_n(b) = Z_{n\u22121}(a/b)\n', 'dim')
        tw.insert('end', '  Z_n(a) / Z_n(b) = Z_{n+1}(a\u2212b)\n', 'dim')
        tw.insert('end', '  Z_n(a) ^ b       = Z_{n+1}(a\u00b7b)\n', 'dim')
        tw.insert('end', '  b ^ Z_n(a)       = Z_{n\u22121}(a\u00b7b)\n', 'dim')
        tw.insert('end', '\n')

        # --- Omega decomposition: value = r + p*w ---
        tw.insert('end', 'Omega Decomposition\n', 'header')
        tw.insert('end', '  value = r + p\u00b7\u03c9\n', 'dim')

        # Extract the value to decompose (top-level or degrade the whole thing)
        if isinstance(parsed, GradedElement):
            val = parsed.value
        else:
            val = _degrade(parsed)

        # Try to separate into real part and omega coefficient
        r_part, w_part = _split_omega(val)
        tw.insert('end', f'  r = {format_result(r_part)}\n', 'value')
        tw.insert('end', f'  p = {format_result(w_part)}\n', 'value')

        # Try numeric evaluation if no free symbols
        if not val.free_symbols:
            try:
                val_with_i = val.subs(Omega(), symI)
                z_complex = complex(val_with_i.evalf())
                tw.insert('end', f'\n  Numeric (r + p\u00b7\u03c9):\n', 'dim')
                tw.insert('end', f'    r = {z_complex.real:+.6f}\n', 'value')
                tw.insert('end', f'    p = {z_complex.imag:+.6f}\n', 'value')
                tw.insert('end', f'    |value| = {abs(z_complex):.6f}\n', 'value')
                if z_complex != 0:
                    import math
                    angle = math.atan2(z_complex.imag, z_complex.real)
                    tw.insert('end', f'    arg = {angle/math.pi:.4f}\u03c0  ({math.degrees(angle):.1f}\u00b0)\n', 'value')
            except Exception as e:
                tw.insert('end', f'  Numeric eval error: {e}\n', 'dim')
        tw.insert('end', '\n')

        # --- Fixed-point check ---
        tw.insert('end', 'Fixed Points\n', 'header')
        tw.insert('end', '  Z_n(0)  = Z_{n\u22121}(1)       Z_n(1) = Z_{n\u22121}(0)\n', 'dim')
        tw.insert('end', '  Z_n(\u03c9) = Z_{n\u22121}(\u22121)     Z_n(\u22121) = Z_{n\u22121}(\u03c9)\n', 'dim')

        if isinstance(parsed, GradedElement) and not parsed.value.free_symbols:
            val = parsed.value
            grade = parsed.grade
            if val == Integer(0) or isinstance(val, Zero):
                tw.insert('end', f'  \u2714 Value is 0: Z_{format_result(grade)}(0) = Z_{format_result(grade - 1)}(1)\n', 'value')
            elif val == Integer(1):
                tw.insert('end', f'  \u2714 Value is 1: Z_{format_result(grade)}(1) = Z_{format_result(grade - 1)}(0)\n', 'value')
            elif val == S.NegativeOne:
                tw.insert('end', f'  \u2714 Value is \u22121: Z_{format_result(grade)}(\u22121) = Z_{format_result(grade - 1)}(\u03c9)\n', 'value')
            elif isinstance(val, Omega):
                tw.insert('end', f'  \u2714 Value is \u03c9: Z_{format_result(grade)}(\u03c9) = Z_{format_result(grade - 1)}(\u22121)\n', 'value')
            else:
                tw.insert('end', f'  Value is not a fixed point\n', 'dim')
        tw.insert('end', '\n')

        # --- Composition check ---
        tw.insert('end', 'Composition\n', 'header')
        tw.insert('end', '  Z_n(Z_{n\u00b11}(x)) = x\n', 'dim')
        if isinstance(parsed, GradedElement) and isinstance(parsed.value, GradedElement):
            inner = parsed.value
            diff = parsed.grade - inner.grade
            if diff == Integer(1) or diff == Integer(-1):
                tw.insert('end', f'  \u2714 Adjacent grades ({format_result(parsed.grade)}, {format_result(inner.grade)}): collapses to x = {format_result(inner.value)}\n', 'value')
            else:
                tw.insert('end', f'  \u2718 Non-adjacent grades ({format_result(parsed.grade)}, {format_result(inner.grade)}): gap = {format_result(diff)}, no collapse\n', 'value')
        else:
            tw.insert('end', '  (no nested graded composition)\n', 'dim')
        tw.insert('end', '\n')

        # --- Plot info ---
        tw.insert('end', 'Plot Mapping\n', 'header')
        tw.insert('end', '  \u03c9 \u2192 vertical axis (imaginary)\n', 'dim')
        tw.insert('end', '  r \u2192 horizontal axis (real)\n', 'dim')
        if free:
            tw.insert('end', f'  Grid variables: {", ".join(str(s) for s in sorted(free, key=str))}\n', 'value')
            # Show what the degrade function produces
            degraded = _degrade(parsed)
            tw.insert('end', f'  Degraded expr: {degraded}\n', 'value')
            degraded_omega = degraded.subs(symI, Symbol('\u03c9'))
            tw.insert('end', f'  As r + p\u00b7\u03c9: {degraded_omega}\n', 'value')
        else:
            tw.insert('end', '  (no free variables \u2014 constant expression, not plottable)\n', 'dim')

    def _render_cheb_text(self, decomp):
        """Render the Chebyshev decomposition into the text widget."""
        tw = self.cheb_text

        traction_str = decomp['traction_str']

        if not decomp['has_decomp']:
            tw.insert('end', traction_str + '\n', 'expr')
            tw.insert('end', '\n')
            tw.insert('end', decomp.get('note', '') + '\n', 'dim')
            return

        # --- Ring form (general expressions: sums, products, etc.) ---
        if decomp.get('ring_form'):
            tw.insert('end', traction_str + '\n', 'expr')

            tw.insert('end', '\n')
            tw.insert('end', 'Ring Decomposition\n', 'header')
            tw.insert('end', f'{decomp["ring_label"]}\n', 'dim')
            tw.insert('end', f'{decomp["gen_note"]}\n', 'dim')

            tw.insert('end', '\n')
            tw.insert('end', f'  {decomp["ring_str"]}\n', 'value')
            tw.insert('end', '\n')
            for name, val in decomp['components']:
                tw.insert('end', f'  {name} = {val}\n', 'value')

            tw.insert('end', '\n')
            tw.insert('end', 'Conjugate\n', 'header')
            tw.insert('end', f'  {decomp["conj_str"]}\n', 'value')

            tw.insert('end', '\n')
            tw.insert('end', 'Norm\n', 'header')
            tw.insert('end', f'  N = {decomp["norm_str"]}\n', 'value')
            if decomp['is_unit']:
                tw.insert('end', '  (unit \u2014 invertible)\n', 'dim')

            # Wedge products — computed purely from ring coordinates, no projection
            ring_el = decomp.get('_ring_el')
            if ring_el is not None:
                from chebyshev_ring import Element, QsPoly
                from fractions import Fraction as Frac
                import math as _math

                tw.insert('end', '\n')
                tw.insert('end', 'Wedge Products\n', 'header')
                tw.insert('end', '  P \u2227 Q = a\u2081b\u2082 \u2212 b\u2081a\u2082  (in Q[s])\n', 'dim')

                p_a, p_b = ring_el.a, ring_el.b
                p_norm = ring_el.norm()

                # Reference A: i = 0^(ω/2) → in the ring this is g (a=0, b=1)
                ref_a = Element(QsPoly.zero(), QsPoly.one())
                ref_a_label = 'i'
                wedge_a = p_a * ref_a.b - p_b * ref_a.a  # a₁·1 - b₁·0 = a₁
                ref_a_norm = ref_a.norm()

                ring_info = decomp.get('_ring_info', {})
                gen_label = decomp.get('gen_note', 'g').split(',')[0].replace('g = ', '').strip()

                def _fmt_poly(poly):
                    """Format a QsPoly: if constant, show the number; otherwise show polynomial."""
                    if poly.is_constant():
                        v = poly.constant_value()
                        return str(v) if v.denominator != 1 else str(v.numerator)
                    return repr(poly)

                def _try_sin(wedge_poly, norm_p, norm_q):
                    """Compute sin(θ) if all inputs are constant polynomials."""
                    if wedge_poly.is_constant() and norm_p.is_constant() and norm_q.is_constant():
                        w = float(wedge_poly.constant_value())
                        np_ = float(norm_p.constant_value())
                        nq = float(norm_q.constant_value())
                        if np_ > 0 and nq > 0:
                            sin_val = w / _math.sqrt(np_ * nq)
                            return sin_val
                    return None

                # Reference A: g = generator (a=0, b=1)
                tw.insert('end', f'\n  P \u2227 {gen_label}:\n', 'value')
                tw.insert('end', f'    wedge = {_fmt_poly(wedge_a)}\n', 'value')
                tw.insert('end', f'    N(P) = {_fmt_poly(p_norm)},  N({gen_label}) = {_fmt_poly(ref_a_norm)}\n', 'dim')

                sin_a = _try_sin(wedge_a, p_norm, ref_a_norm)
                if sin_a is not None:
                    tw.insert('end', f'    sin(\u03b8) = {Frac(sin_a).limit_denominator(1000)}', 'value')
                    if abs(sin_a) <= 1:
                        tw.insert('end', f'  \u2192  \u03b8 = {_math.degrees(_math.asin(sin_a)):.2f}\u00b0\n', 'dim')
                    else:
                        tw.insert('end', '\n')
                elif not wedge_a.is_constant():
                    tw.insert('end', f'    sin(\u03b8) = {_fmt_poly(wedge_a)} / \u221a({_fmt_poly(p_norm)}\u00b7{_fmt_poly(ref_a_norm)})\n', 'value')

            # Reduced Form: reduce via 0 + ω = ∅ (Chebyshev identity)
            reduced_result = _reduce_ring_form(decomp)
            if reduced_result is not None:
                reduced_str, reduced_cval = reduced_result
                decomp['_reduced_str'] = reduced_str
                decomp['_reduced_complex'] = reduced_cval
                if reduced_str != traction_str:
                    base_d = ring_info.get('base_denom', decomp.get('base_denom', 2))
                    tw.insert('end', '\n')
                    tw.insert('end', 'Reduced Form\n', 'header')
                    tw.insert('end', f'  0 + \u03c9 = \u2205,  T\u2099(s/2) = 0  (n={base_d})\n', 'dim')
                    tw.insert('end', f'  {reduced_str}\n', 'value')

            return

        # --- Single zero-power form (legacy path) ---
        equiv_str = decomp['equiv_str']
        if equiv_str != traction_str:
            tw.insert('end', f'{traction_str}  =  {equiv_str}\n', 'expr')
        else:
            tw.insert('end', traction_str + '\n', 'expr')

        # Chebyshev Polynomial
        hc_label = decomp['half_label']
        tw.insert('end', '\n')
        tw.insert('end', 'Chebyshev Polynomial\n', 'header')

        step_denom = decomp.get('step_denom', 2)
        if step_denom == 2:
            gen_note = 's = u + v,  u = 0^(1/2),  v = \u03c9^(1/2)'
        else:
            d = step_denom
            gen_note = f's = u + v,  u = 0^(1/{d}),  v = \u03c9^(1/{d})'
        tw.insert('end', gen_note + '\n', 'dim')

        cheb_str = decomp['cheb_str']
        tw.insert('end', f'a({hc_label}) = {cheb_str}\n', 'value')

        # Recurrence Trace
        trace = decomp.get('trace', [])
        if trace:
            tw.insert('end', '\n')
            tw.insert('end', 'Recurrence Trace\n', 'header')
            tw.insert('end', 'a\u2099 = (u+v)\u00b7a\u2099\u208b\u2081 \u2212 a\u2099\u208b\u2082\n', 'dim')
            for label, poly_str in trace:
                tw.insert('end', f'  {label} = {poly_str}\n', 'value')

    @staticmethod
    def _traction_plane_coords(decomp):
        """Extract (x, y) coordinates for the traction plane.
        x = scalar multiplier, y = traction power level.
        0^n → y = -n (zero-powers go down), ω^n → y = +n (omega-powers go up).
        Scalars → y = 0. Mixed → try ring (a, b) or fallback."""
        if decomp is None:
            return None, None

        simplified_str = decomp.get('traction_str', '')
        band = decomp.get('band', '')

        # Try to extract from the traction expression structure
        # Re-parse from display string (replace · with * for parser compatibility)
        try:
            clean_str = simplified_str.replace('\u00b7', '*') if simplified_str else ''
            parsed = parse_and_eval(clean_str) if clean_str else None
        except Exception:
            parsed = None

        if parsed is not None:
            parsed_s = traction_simplify(parsed)

            # Pure scalar: y=0, x=value
            if isinstance(parsed_s, (Integer, Rational)):
                return float(parsed_s), 0.0

            # Bare Zero: 0^1 → x=0, y=-1
            if isinstance(parsed_s, Zero):
                return 0.0, -1.0

            # Bare Omega: ω^1 = 0^(-1) → x=0, y=1
            if isinstance(parsed_s, Omega):
                return 0.0, 1.0

            # Zero power: 0^n → x=0, y=-n
            # Also handles omega exponents: 0^(ω·t) → y=t (omega-direction)
            if isinstance(parsed_s, Pow) and isinstance(parsed_s.base, Zero):
                exp = parsed_s.exp
                if isinstance(exp, (Integer, Rational)):
                    return 0.0, -float(exp)
                # Omega-containing exponent: 0^(ω·t) → t omega-steps
                omega_info = _extract_omega_rational(exp)
                if omega_info is not None:
                    coeff, _ = omega_info
                    return 0.0, float(coeff)

            # Omega power: ω^n → x=0, y=n
            if isinstance(parsed_s, Pow) and isinstance(parsed_s.base, Omega):
                exp = parsed_s.exp
                if isinstance(exp, (Integer, Rational)):
                    return 0.0, float(exp)
                omega_info = _extract_omega_rational(exp)
                if omega_info is not None:
                    coeff, _ = omega_info
                    return 0.0, -float(coeff)

            # Mul: coeff * 0^n or coeff * ω^n
            if isinstance(parsed_s, Mul):
                from sympy import Mul as SMul
                coeff = 1.0
                power_y = None
                for arg in SMul.make_args(parsed_s):
                    if isinstance(arg, (Integer, Rational)):
                        coeff *= float(arg)
                    elif isinstance(arg, Zero):
                        power_y = -1.0
                    elif isinstance(arg, Omega):
                        power_y = 1.0
                    elif isinstance(arg, Pow) and isinstance(arg.base, Zero):
                        if isinstance(arg.exp, (Integer, Rational)):
                            power_y = -float(arg.exp)
                    elif isinstance(arg, Pow) and isinstance(arg.base, Omega):
                        if isinstance(arg.exp, (Integer, Rational)):
                            power_y = float(arg.exp)
                if power_y is not None:
                    return coeff, power_y

        # Omega-band ring elements with numeric (a, b)
        if band == 'omega':
            components = decomp.get('components', [])
            if len(components) >= 2:
                try:
                    a_val = float(components[0][1])
                    b_val = float(components[1][1])
                    return a_val, b_val
                except (ValueError, TypeError):
                    pass

        return None, None

    def _draw_cheb_orbit(self, ax, decomp):
        """Draw the traction plane at θ = π/2.

        Axes: Re = horizontal, Im = vertical.
        Cardinal points: 1 (right), 0 (top), -1 (left), -0 (bottom).
        Plots both the exact value and reduced form on the unit circle."""
        ax.clear()
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')
        ax.tick_params(colors='#888899', labelsize=8)

        # Grid and unit circle
        ax.axhline(0, color='#333344', linewidth=0.5)
        ax.axvline(0, color='#333344', linewidth=0.5)
        circ_t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circ_t), np.sin(circ_t), color='#333344', linewidth=0.8)

        # Cardinal reference points: 1 (right), -1 (left), 0 (top), -0 (bottom)
        refs = [
            (1, 0, '1', '#ffffff'),
            (-1, 0, '\u22121', '#ff8888'),
            (0, 1, '0', '#88ff88'),
            (0, -1, '\u22120', '#88ff88'),
        ]
        for rx, ry, rlabel, rcolor in refs:
            ax.plot(rx, ry, 'o', color=rcolor, markersize=4, zorder=4, alpha=0.5)
            ox = 8 if rx >= 0 else -18
            oy = 6 if ry >= 0 else -12
            ax.annotate(rlabel, (rx, ry), textcoords='offset points',
                        xytext=(ox, oy), fontsize=9, color=rcolor, alpha=0.7)

        ax.set_title('Traction Plane  (\u03b8 = \u03c0/2)', fontsize=11, color='#eeeeee')

        if decomp is None:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            return

        traction_str = decomp.get('traction_str', '')
        max_r = 1.0  # track extent for axis limits

        # --- Plot the exact value (original expression at θ = π/2) ---
        exact_val = decomp.get('_pi2_val')
        if exact_val is not None and np.isfinite(exact_val):
            ax.plot(exact_val.real, exact_val.imag, 'o', color='#00ccff',
                    markersize=10, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.5)
            label = traction_str if len(traction_str) < 20 else traction_str[:17] + '\u2026'
            ax.annotate(label, (exact_val.real, exact_val.imag),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=10, color='#00ccff')
            max_r = max(max_r, abs(exact_val.real), abs(exact_val.imag))

        # --- Plot the reduced form (if different label) ---
        reduced_str = decomp.get('_reduced_str')
        reduced_val = decomp.get('_reduced_complex')
        if (reduced_str is not None and reduced_str != traction_str
                and reduced_val is not None and np.isfinite(reduced_val)):
            # Offset slightly if the two points coincide
            dx, dy = 0, 0
            if exact_val is not None and abs(reduced_val - exact_val) < 0.01:
                dy = -0.08  # nudge the marker down a touch
            ax.plot(reduced_val.real + dx, reduced_val.imag + dy, 's',
                    color='#ffcc00', markersize=8, zorder=6,
                    markeredgecolor='white', markeredgewidth=1.0)
            ax.annotate(reduced_str, (reduced_val.real + dx, reduced_val.imag + dy),
                        textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#ffcc00')
            max_r = max(max_r, abs(reduced_val.real), abs(reduced_val.imag))

        lim = max(1.5, max_r * 1.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    def _on_viz_hover(self, event):
        """Draw gauge readouts for the hovered pixel."""
        if self.viz_Z is None:
            return

        # Convert canvas coords to grid indices
        px = event.x - AXIS_MARGIN
        py = event.y - AXIS_MARGIN
        if px < 0 or py < 0 or px >= CANVAS_SIZE or py >= CANVAS_SIZE:
            self._clear_gauges()
            return

        h, w = self.viz_Z.shape
        col = int(px / CANVAS_SIZE * w)
        row = int(py / CANVAS_SIZE * h)
        col = min(col, w - 1)
        row = min(row, h - 1)

        # Grid coordinates (p = horizontal, q = vertical)
        bounds = self.viz_bounds
        p_val = -bounds + (col / w) * 2 * bounds
        q_val = bounds - (row / h) * 2 * bounds

        # Fractal mode: show iteration count and last z
        if getattr(self, '_fractal_mode', False) and self._fractal_counts is not None:
            self._draw_fractal_gauge(row, col, p_val, q_val)
            return

        z = self.viz_Z[row, col]
        if not np.isfinite(z):
            self._draw_overflow_gauge(p_val, q_val, z)
            return
        if z == 0:
            self._draw_overflow_gauge(p_val, q_val, complex(0))
            return

        self._draw_gauges(z.real, z.imag, abs(z), np.angle(z), p_val, q_val)

    def _on_viz_leave(self, event):
        self._clear_gauges()

    def _draw_fractal_gauge(self, row, col, p_val, q_val):
        """Draw hover info for fractal mode: coordinates, iteration count, last value."""
        gc = self.gauge_canvas
        gc.delete('all')

        counts = self._fractal_counts
        last_z = self._fractal_last_z
        max_iter = self._fractal_max_iter

        n = int(counts[row, col])
        zv = last_z[row, col]

        label_font = tkfont.Font(family='Consolas', size=9)
        small_font = tkfont.Font(family='Consolas', size=8)

        y = 8
        # Pixel coordinate (traction labels: imaginary axis = 0/-0)
        q_sign = '+' if q_val >= 0 else '\u2212'
        q_abs = abs(q_val)
        gc.create_text(8, y, text=f'c = {p_val:+.4f} {q_sign} {q_abs:.4f}\u00b70',
                        font=label_font, fill='#cccccc', anchor='nw')
        y += 16

        # Iteration count
        if n == 0:
            gc.create_text(8, y, text=f'iter: {max_iter} (bounded)',
                            font=label_font, fill='#88ff88', anchor='nw')
        else:
            gc.create_text(8, y, text=f'iter: {n} (escaped)',
                            font=label_font, fill='#ffaa44', anchor='nw')
        y += 16

        # Last z value (traction form: a + b·0)
        if np.isfinite(zv):
            zr, zi = zv.real, zv.imag
            z_sign = '+' if zi >= 0 else '\u2212'
            gc.create_text(8, y, text=f'last x = {zr:+.4f} {z_sign} {abs(zi):.4f}\u00b70',
                            font=small_font, fill='#999999', anchor='nw')
            y += 14
            gc.create_text(8, y, text=f'|x| = {abs(zv):.4f}',
                            font=small_font, fill='#999999', anchor='nw')
        else:
            gc.create_text(8, y, text=f'last x = overflow',
                            font=small_font, fill='#ff4444', anchor='nw')

    def _clear_gauges(self):
        self.gauge_canvas.delete('all')

    def _draw_overflow_gauge(self, p_val, q_val, z):
        """Draw an overflow/underflow indicator with p,q coordinates."""
        gc = self.gauge_canvas
        gc.delete('all')

        label_font = tkfont.Font(family='Consolas', size=10)
        coord_font = tkfont.Font(family='Consolas', size=8)

        if np.isinf(z):
            msg = 'overflow!'
            color = '#ff4444'
        elif z == 0:
            msg = 'underflow'
            color = '#888888'
        elif np.isnan(z):
            msg = 'undefined'
            color = '#ff8844'
        else:
            msg = 'invalid'
            color = '#888888'

        gc.create_text(CANVAS_TOTAL // 2 - 40, 36, text=msg, font=label_font,
                        fill=color, anchor='center')
        gc.create_text(CANVAS_TOTAL // 2 + 40, 28, text=f'p={p_val:+.3g}', font=coord_font,
                        fill='#bbbbbb', anchor='w')
        gc.create_text(CANVAS_TOTAL // 2 + 40, 44, text=f'q={q_val:+.3g}', font=coord_font,
                        fill='#bbbbbb', anchor='w')

    def _draw_gauges(self, re, im, mag, phase, p_val=0, q_val=0):
        """Draw 3 scale boxes (Re, Im, |f|), 1 phase compass, and p/q coordinates."""
        gc = self.gauge_canvas
        gc.delete('all')

        gh = 72   # gauge area height
        bw = 24   # box width
        bh = 46   # box height
        label_h = 12  # space for labels below boxes
        gap = 12  # gap between boxes
        compass_r = 22  # compass radius

        # Layout: [Re box] [Im box] [|f| box] ... [compass] ... [p,q text]
        total_boxes_w = 3 * bw + 2 * gap
        compass_d = compass_r * 2
        total_w = total_boxes_w + gap + compass_d
        x_start = (CANVAS_TOTAL - total_w) // 2
        y_top = 4
        y_bot = y_top + bh

        # Draw the 3 scale boxes
        if getattr(self, '_viz_graded', False):
            labels = ['r', '\u03c9', '|f|']
        else:
            labels = ['Re', 'Im', '|f|']
        values = [re, im, mag]
        for idx, (label, val) in enumerate(zip(labels, values)):
            bx = x_start + idx * (bw + gap)
            self._draw_scale_box(bx, y_top, bw, bh, val, label, is_magnitude=(idx == 2))

        # Draw the phase compass (vertically centered on boxes)
        cx = x_start + total_boxes_w + gap + compass_r
        cy = y_top + bh // 2
        self._draw_compass(cx, cy, compass_r, phase)

        # Draw p,q coordinate labels to the right of the compass
        coord_font = tkfont.Font(family='Consolas', size=8)
        tx = cx + compass_r + 10
        gc.create_text(tx, cy - 8, text=f'p={p_val:+.3g}', font=coord_font,
                        fill='#bbbbbb', anchor='w')
        gc.create_text(tx, cy + 8, text=f'q={q_val:+.3g}', font=coord_font,
                        fill='#bbbbbb', anchor='w')

    def _draw_scale_box(self, x, y, w, h, value, label, is_magnitude=False):
        """Draw a graduated scale box with logarithmic fill."""
        gc = self.gauge_canvas
        tick_font = tkfont.Font(family='Consolas', size=7)

        # Box outline
        gc.create_rectangle(x, y, x + w, y + h, outline='#666666', width=1)

        # Compute fill level: log scale mapped to [0, 1]
        if is_magnitude:
            # Magnitude: always positive, log scale
            if value <= 0:
                fill_t = 0.0
            else:
                fill_t = min(1.0, (np.log10(value + 1e-15) + 4) / 8)  # -4 to +4 range
            negative = False
        else:
            # Re/Im: signed, log scale of absolute value
            negative = value < 0
            av = abs(value)
            if av < 1e-15:
                fill_t = 0.0
            else:
                fill_t = min(1.0, (np.log10(av) + 4) / 8)

        fill_t = max(0.0, fill_t)

        # Fill color: maroon -> purple -> blue -> green(0) -> yellow -> red
        color = _scale_color(fill_t, negative=negative)

        # Fill direction: positive fills bottom-up, negative fills top-down
        fill_h = int(fill_t * h)
        if fill_h > 0:
            if negative:
                gc.create_rectangle(x + 1, y + 1, x + w - 1, y + fill_h,
                                    fill=color, outline='')
            else:
                gc.create_rectangle(x + 1, y + h - fill_h, x + w - 1, y + h - 1,
                                    fill=color, outline='')

        # Graduation lines (5 ticks)
        for i in range(1, 5):
            ty = y + int(i * h / 5)
            gc.create_line(x, ty, x + 3, ty, fill='#555555')

        # Label below
        gc.create_text(x + w // 2, y + h + 7, text=label,
                        fill='#999999', font=tick_font)

    def _draw_compass(self, cx, cy, r, phase):
        """Draw a phase compass: circle with arrow pointing at the phase angle."""
        gc = self.gauge_canvas
        tick_font = tkfont.Font(family='Consolas', size=7)

        # Circle outline
        gc.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#666666', width=1)

        # Tick marks at 0, pi/2, pi, 3pi/2
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            tx = cx + (r - 3) * np.cos(angle)
            ty = cy - (r - 3) * np.sin(angle)
            tx2 = cx + r * np.cos(angle)
            ty2 = cy - r * np.sin(angle)
            gc.create_line(tx, ty, tx2, ty2, fill='#555555')

        # Arrow from center to edge at the phase angle
        ax = cx + (r - 4) * np.cos(phase)
        ay = cy - (r - 4) * np.sin(phase)
        gc.create_line(cx, cy, ax, ay, fill='#dddddd', width=2, arrow='last',
                        arrowshape=(6, 8, 3))

    def _zoom_in(self):
        self.viz_bounds = max(0.25, self.viz_bounds / 2)
        self._refresh_viz_or_fractal()

    def _zoom_out(self):
        self.viz_bounds = min(100, self.viz_bounds * 2)
        self._refresh_viz_or_fractal()

    def _zoom_reset(self):
        self.viz_bounds = DEFAULT_BOUNDS
        self._refresh_viz_or_fractal()

    def _refresh_viz_or_fractal(self):
        """Re-render: if in fractal mode re-launch the fractal, else normal plot."""
        if getattr(self, '_fractal_mode', False) and hasattr(self, '_fractal_raw_text'):
            try:
                self._handle_fractal(self._fractal_raw_text)
            except Exception:
                self._refresh_viz()
        else:
            self._refresh_viz()


def _nice_tick_step(bounds):
    """Choose a nice tick interval for the given bounds."""
    # Target ~5-7 ticks across the full range
    raw_step = (2 * bounds) / 6
    # Round to a nice number: 0.25, 0.5, 1, 2, 5, 10, ...
    nice = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
    for n in nice:
        if n >= raw_step:
            return n
    return raw_step


def _tick_label(val):
    """Format a tick value concisely."""
    if val == int(val):
        return str(int(val))
    return f'{val:g}'


def _scale_color(t, negative=False):
    """
    Map t in [0, 1] to a color string for the scale boxes.
    Positive: green -> yellow -> red (low to high)
    Negative: green -> cyan -> blue (low to high)
    """
    t = max(0.0, min(1.0, t))
    if negative:
        stops = [
            (0.0, (0, 100, 40)),      # dark green
            (0.4, (0, 200, 120)),     # green
            (0.7, (0, 180, 200)),     # cyan
            (1.0, (0, 60, 220)),      # blue
        ]
    else:
        stops = [
            (0.0, (0, 100, 40)),      # dark green
            (0.4, (0, 200, 60)),      # green
            (0.7, (220, 200, 0)),     # yellow
            (1.0, (220, 40, 0)),      # red
        ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f'#{r:02x}{g:02x}{b:02x}'
    return '#dc2800' if not negative else '#003cdc'



def _clip_line(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Clip a line segment to a rectangle using Cohen-Sutherland. Returns (x1,y1,x2,y2) or None."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < xmin: c |= LEFT
        elif x > xmax: c |= RIGHT
        if y < ymin: c |= TOP
        elif y > ymax: c |= BOTTOM
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    for _ in range(20):
        if not (c1 | c2):
            return (x1, y1, x2, y2)
        if c1 & c2:
            return None
        c = c1 or c2
        dx, dy = x2 - x1, y2 - y1
        if c & TOP:
            x = x1 + dx * (ymin - y1) / dy if dy else x1; y = ymin
        elif c & BOTTOM:
            x = x1 + dx * (ymax - y1) / dy if dy else x1; y = ymax
        elif c & RIGHT:
            y = y1 + dy * (xmax - x1) / dx if dx else y1; x = xmax
        elif c & LEFT:
            y = y1 + dy * (xmin - x1) / dx if dx else y1; x = xmin
        if c == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)
    return None


class SettingsWindow:
    """Single-instance settings window with tabbed interface."""

    _instance = None

    def __init__(self, parent_app):
        if SettingsWindow._instance is not None:
            # Focus existing window
            SettingsWindow._instance.window.lift()
            SettingsWindow._instance.window.focus_force()
            return

        self.app = parent_app
        self.window = tk.Toplevel(parent_app.root)
        self.window.title('Settings')
        self.window.configure(bg=BG_BODY)
        self.window.resizable(False, False)
        self.window.protocol('WM_DELETE_WINDOW', self._on_close)
        SettingsWindow._instance = self

        # Tab bar
        self.tab_bar = tk.Frame(self.window, bg=BG_FRAME)
        self.tab_bar.pack(fill='x')

        self.tab_frames = {}
        self.tab_buttons = {}
        self.active_tab = None

        # Content area
        self.content = tk.Frame(self.window, bg=BG_BODY, padx=16, pady=12)
        self.content.pack(fill='both', expand=True)

        # Add tabs
        self._add_tab('Visualization', self._build_viz_tab)

        # Show first tab
        self.select_tab('Visualization')

    def _add_tab(self, name, builder):
        """Add a tab with a name and a builder function."""
        font = tkfont.Font(family='Segoe UI', size=10)
        btn = tk.Button(
            self.tab_bar, text=name, font=font, bd=0, padx=12, pady=4,
            bg=BG_FRAME, fg=FG_TEXT, activebackground=BG_BODY,
            command=lambda: self.select_tab(name)
        )
        btn.pack(side='left')
        self.tab_buttons[name] = btn

        frame = tk.Frame(self.content, bg=BG_BODY)
        builder(frame)
        self.tab_frames[name] = frame

    def select_tab(self, name):
        """Switch to the named tab."""
        if self.active_tab == name:
            return
        # Hide current
        if self.active_tab and self.active_tab in self.tab_frames:
            self.tab_frames[self.active_tab].pack_forget()
            self.tab_buttons[self.active_tab].configure(bg=BG_FRAME, relief='flat')
        # Show new
        self.tab_frames[name].pack(fill='both', expand=True)
        self.tab_buttons[name].configure(bg=BG_BODY, relief='flat')
        self.active_tab = name

    def _build_viz_tab(self, parent):
        """Build the Visualization settings tab."""
        font = tkfont.Font(family='Segoe UI', size=10)
        font_small = tkfont.Font(family='Segoe UI', size=9)

        # Section: Projection
        tk.Label(parent, text='Projection', font=font, bg=BG_BODY, fg=FG_TEXT,
                 anchor='w').pack(fill='x', pady=(0, 4))

        proj_frame = tk.Frame(parent, bg=BG_BODY)
        proj_frame.pack(fill='x', pady=(0, 12))

        self.proj_var = tk.StringVar(value=self.app.projection_names[self.app.projection_index])
        for name in self.app.projection_names:
            entry = registry.get_entry('projection', name)
            desc = entry.get('description', name) if entry else name
            label = name.replace('_', ' ').title()
            rb = tk.Radiobutton(
                proj_frame, text=f'{label}', font=font_small,
                variable=self.proj_var, value=name,
                bg=BG_BODY, activebackground=BG_BODY,
                command=self._on_projection_change
            )
            rb.pack(anchor='w')
            tk.Label(proj_frame, text=f'  {desc}', font=font_small,
                     bg=BG_BODY, fg=FG_DIM).pack(anchor='w', padx=(20, 0))

        # Section: Color Mode
        tk.Label(parent, text='Color Mode', font=font, bg=BG_BODY, fg=FG_TEXT,
                 anchor='w').pack(fill='x', pady=(0, 4))

        color_frame = tk.Frame(parent, bg=BG_BODY)
        color_frame.pack(fill='x', pady=(0, 12))

        self.color_var = tk.StringVar(value=self.app.color_mode)
        for mode, desc in [('phase', 'Phase (CMYT quadrants + brightness)'),
                           ('continuity', 'Continuity (magnitude, double-cover)')]:
            rb = tk.Radiobutton(
                color_frame, text=desc, font=font_small,
                variable=self.color_var, value=mode,
                bg=BG_BODY, activebackground=BG_BODY,
                command=self._on_color_change
            )
            rb.pack(anchor='w')

        # Section: Overlays
        tk.Label(parent, text='Overlays', font=font, bg=BG_BODY, fg=FG_TEXT,
                 anchor='w').pack(fill='x', pady=(0, 4))

        self.tangent_var = tk.BooleanVar(value=self.app.show_tangent)
        tk.Checkbutton(parent, text='Tangent lines (gradient flow)',
                       variable=self.tangent_var, font=font_small,
                       bg=BG_BODY, activebackground=BG_BODY,
                       command=self._on_tangent_change).pack(anchor='w')

        self.normal_var = tk.BooleanVar(value=self.app.show_normal)
        tk.Checkbutton(parent, text='Normal lines (constant |f| contours)',
                       variable=self.normal_var, font=font_small,
                       bg=BG_BODY, activebackground=BG_BODY,
                       command=self._on_normal_change).pack(anchor='w')

        self.diamond_var = tk.BooleanVar(value=self.app.show_diamond)
        tk.Checkbutton(parent, text='Diamond grid (simple fraction diagonals)',
                       variable=self.diamond_var, font=font_small,
                       bg=BG_BODY, activebackground=BG_BODY,
                       command=self._on_diamond_change).pack(anchor='w')

        # Section: Grid
        tk.Label(parent, text='Grid', font=font, bg=BG_BODY, fg=FG_TEXT,
                 anchor='w').pack(fill='x', pady=(12, 4))

        bounds_frame = tk.Frame(parent, bg=BG_BODY)
        bounds_frame.pack(fill='x')
        tk.Label(bounds_frame, text='Bounds:', font=font_small, bg=BG_BODY).pack(side='left')
        self.bounds_var = tk.StringVar(value=str(self.app.viz_bounds))
        bounds_entry = tk.Entry(bounds_frame, textvariable=self.bounds_var, width=8,
                                font=font_small)
        bounds_entry.pack(side='left', padx=4)
        bounds_entry.bind('<Return>', self._on_bounds_change)

    def _on_projection_change(self):
        name = self.proj_var.get()
        idx = self.app.projection_names.index(name) if name in self.app.projection_names else 0
        self.app.projection_index = idx
        label = name.replace('_', ' ')
        self.app.viz_title_label.configure(text=f'Phase Plot [{label}]')
        if self.app.viz_Z is not None:
            self.app._refresh_viz()

    def _on_color_change(self):
        self.app.color_mode = self.color_var.get()
        if self.app.viz_Z is not None:
            self.app._refresh_viz()

    def _on_tangent_change(self):
        self.app.show_tangent = self.tangent_var.get()
        if self.app.viz_Z is not None:
            if self.app.show_tangent:
                self.app._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            else:
                self.app.viz_canvas.delete('tangent')

    def _on_normal_change(self):
        self.app.show_normal = self.normal_var.get()
        if self.app.viz_Z is not None:
            if self.app.show_normal:
                self.app._draw_flow_lines(normal=True, tag='normal', color='#aaaaaa')
            else:
                self.app.viz_canvas.delete('normal')

    def _on_diamond_change(self):
        self.app.show_diamond = self.diamond_var.get()
        if self.app.viz_Z is not None:
            if self.app.show_diamond:
                self.app._draw_diamond_grid()
            else:
                self.app.viz_canvas.delete('diamond')

    def _on_bounds_change(self, event=None):
        try:
            val = float(self.bounds_var.get())
            if 0.1 <= val <= 200:
                self.app.viz_bounds = val
                if self.app.viz_Z is not None:
                    self.app._refresh_viz()
        except ValueError:
            pass

    def _on_close(self):
        SettingsWindow._instance = None
        self.window.destroy()


def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
