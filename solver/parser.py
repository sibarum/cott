"""
Traction expression parser and evaluator.

Recursive descent parser for traction expressions with support for
equation solving, graded elements, and function calls.
"""

from sympy import S, Integer, Rational, Pow, Mul, Add, Symbol
from sympy import solve as sp_solve, expand as sp_expand, factor as sp_factor
from traction import Zero, Omega, Null, Log0, LogW, z, w, null, traction_simplify, log0, logw, GradedElement, Z


class ParseError(Exception):
    pass


class SolutionSet:
    """Wrapper for a list of solutions returned by solve()."""
    def __init__(self, solutions, variable):
        self.solutions = solutions
        self.variable = variable

    def has(self, sym):
        """SymPy-compatible: check if any solution references sym."""
        return any(hasattr(s, 'has') and s.has(sym) for s in self.solutions)


def _to_traction(n):
    """Convert a Python int to the appropriate traction type."""
    if n == 0:
        return z
    return Integer(n)


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
