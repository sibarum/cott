"""
Traction expression parser and evaluator.

Recursive descent parser for traction expressions with support for
equation solving, graded elements, and function calls.
"""

import re
from sympy import S, Integer, Rational, Pow, Mul, Add, Symbol, Function, Expr
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


class FunctionDef:
    """Returned by parse_and_eval when a function definition is processed."""
    def __init__(self, name, params, body):
        self.name = name
        self.params = params  # list of parameter names (strings)
        self.body = body      # unevaluated SymPy expression


# ============================================================
# User-defined function registry
# ============================================================
# Maps function name -> (param_names: list[str], body: Expr)
# Body is stored unsimplified.
_user_functions = {}

# Reserved prefixes that cannot be used as function names
_RESERVED_PREFIXES = ('omega', 'log')
# Reserved names (built-in functions and keywords)
_RESERVED_NAMES = frozenset({
    'solve', 'expand', 'factor', 'null', 'fractal',
    'p', 'q', 'x', 'c', 't', 'w', 'y', 'Z', 'z', 'i',
})

_FUNC_NAME_RE = re.compile(r'^[a-zA-Z][a-zA-Z]*$')


def define_function(name, params, body):
    """Register a user-defined function. Returns a FunctionDef."""
    if not _FUNC_NAME_RE.match(name):
        raise ParseError(f"Invalid function name '{name}'")
    if name in _RESERVED_NAMES:
        raise ParseError(f"'{name}' is a reserved name")
    for prefix in _RESERVED_PREFIXES:
        if name.startswith(prefix):
            raise ParseError(f"Function name cannot start with '{prefix}'")
    # Validate param names
    for p in params:
        if not _FUNC_NAME_RE.match(p):
            raise ParseError(f"Invalid parameter name '{p}'")
    _user_functions[name] = (params, body)
    return FunctionDef(name, params, body)


def get_user_functions():
    """Return a copy of the user-defined function registry."""
    return dict(_user_functions)


def delete_user_function(name):
    """Remove a user-defined function."""
    _user_functions.pop(name, None)


def _apply_user_function(name, args):
    """Apply a user-defined function to arguments, substituting at eval time."""
    if name not in _user_functions:
        raise ParseError(f"Undefined function '{name}'")
    params, body = _user_functions[name]
    if len(args) != len(params):
        raise ParseError(f"'{name}' expects {len(params)} argument(s), got {len(args)}")
    # Substitute each parameter:
    #   _FuncRef       → replace _UserCall nodes with actual named function calls
    #   expression     → if param is used as a function (_UserCall), substitute free vars
    #   plain value    → direct symbol substitution
    result = body
    for pname, arg in zip(params, args):
        if isinstance(arg, _FuncRef):
            result = _substitute_func_param(result, pname, arg.name)
        elif _has_user_calls(result, pname):
            result = _substitute_expr_as_func(result, pname, arg)
        else:
            result = result.subs(Symbol(pname), arg)
    return result


class _FuncRef:
    """A reference to a user-defined function, used when passing functions as arguments."""
    def __init__(self, name):
        self.name = name


class _UserCall(Expr):
    """SymPy Expr node representing an unevaluated call to a user-defined function.

    Stored in function bodies when a parameter is called as a function,
    e.g. in g(f,x)=f(x), the body contains _UserCall('f', Symbol('x')).
    When g is applied, _substitute_func_param replaces 'f' with the actual function name,
    or _substitute_expr_as_func substitutes an expression's free variables.

    Data is stored via SymPy's args mechanism so it survives expression rebuilds.
    args[0] = Symbol(func_param_name), args[1:] = call arguments.
    """
    def __new__(cls, name_or_str, *call_args):
        # Accepts two conventions:
        # 1. Our code:     _UserCall('f', arg1, arg2)     — name_or_str is str
        # 2. SymPy rebuild: _UserCall(name_sym, arg1, ...) — name_or_str is Symbol
        if isinstance(name_or_str, str):
            name_sym = Symbol(f'__usercall_{name_or_str}')
        else:
            name_sym = name_or_str
        return Expr.__new__(cls, name_sym, *call_args)

    @property
    def _func_param(self):
        return str(self.args[0]).replace('__usercall_', '', 1)

    @property
    def _call_args(self):
        return self.args[1:]

    @property
    def free_symbols(self):
        result = set()
        for a in self._call_args:
            if hasattr(a, 'free_symbols'):
                result |= a.free_symbols
        result.add(Symbol(self._func_param))
        return result

    def _sympystr(self, printer):
        args_str = ', '.join(printer.doprint(a) for a in self._call_args)
        return f'{self._func_param}({args_str})'

    def __repr__(self):
        from formatting import format_result
        args_str = ', '.join(format_result(a) for a in self._call_args)
        return f'{self._func_param}({args_str})'

    def __str__(self):
        return self.__repr__()


def _has_user_calls(expr, param_name):
    """Check whether expr contains any _UserCall nodes for param_name."""
    if isinstance(expr, _UserCall) and expr._func_param == param_name:
        return True
    if hasattr(expr, 'args') and expr.args:
        return any(_has_user_calls(a, param_name) for a in expr.args)
    if isinstance(expr, _UserCall):
        return any(_has_user_calls(a, param_name) for a in expr._call_args)
    return False


def _substitute_func_param(expr, param_name, actual_func_name):
    """Replace _UserCall nodes for param_name with actual calls to actual_func_name."""
    if isinstance(expr, _UserCall) and expr._func_param == param_name:
        # Replace with actual function call
        new_args = tuple(_substitute_func_param(a, param_name, actual_func_name) for a in expr._call_args)
        return _apply_user_function(actual_func_name, list(new_args))
    if hasattr(expr, 'args') and expr.args:
        new_args = [_substitute_func_param(a, param_name, actual_func_name) for a in expr.args]
        try:
            return expr.func(*new_args)
        except Exception:
            return expr
    return expr


def _substitute_expr_as_func(expr, param_name, func_expr):
    """Replace _UserCall nodes for param_name by substituting func_expr's free variables.

    When an expression like x^2+3*x is passed where a function is expected,
    calling it with argument(s) means substituting the expression's free variables.
    Single free variable: f(a) substitutes that variable with a.
    Multiple free variables: matched positionally in sorted (alphabetical) order.
    """
    if isinstance(expr, _UserCall) and expr._func_param == param_name:
        # Recurse into call args first
        call_args = tuple(_substitute_expr_as_func(a, param_name, func_expr) for a in expr._call_args)
        # Find free variables in the expression, sorted alphabetically
        free = sorted(func_expr.free_symbols, key=str)
        if len(call_args) > len(free):
            raise ParseError(
                f"Expression has {len(free)} free variable(s) but was called with {len(call_args)} argument(s)")
        result = func_expr
        for var, arg in zip(free, call_args):
            result = result.subs(var, arg)
        return result
    if hasattr(expr, 'args') and expr.args:
        new_args = [_substitute_expr_as_func(a, param_name, func_expr) for a in expr.args]
        try:
            return expr.func(*new_args)
        except Exception:
            return expr
    return expr


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
        primary  = '(' expr ')' | graded | funccall | number | 'ω' | 'omega' | var
        graded   = 'Z_' int '(' expr ')' | 'Z' '(' expr ',' expr ')'
        funccall = ('log0' | 'logw') '(' expr ')'
    """

    def __init__(self, text, func_params=None):
        self.text = text.replace(' ', '')
        self.pos = 0
        # func_params: set of parameter names being defined (for function bodies)
        self._func_params = func_params or set()

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

        # Single-letter variables: only match when not followed by more letters
        # (so 'poly' doesn't get eaten as 'p' + 'oly')
        # 'w' is a regular variable — use ω (unicode) or 'omega' for Omega
        if ch in ('p', 'q', 'x', 'c', 't', 'w', 'y') and not self._next_is_alpha(1):
            self.consume()
            if ch == 'y':
                return Symbol('q')  # legacy alias
            return Symbol(ch)

        # null
        if self.match('null'):
            return null

        # Number (integer, rational p/q, or decimal)
        if ch.isdigit() or (ch == '.' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
            return self.number()

        # User-defined function call or function-parameter call: name(args)
        # Also handles bare function-parameter references: passing f as argument
        if ch.isalpha():
            ident = self._read_identifier()
            if ident:
                # Check if this is a call: name(...)
                if self.peek() == '(':
                    # Function parameter used as a function in a body definition
                    if ident in self._func_params:
                        return self._parse_param_call(ident)
                    # User-defined function call
                    if ident in _user_functions:
                        return self._parse_user_func_call(ident)
                    raise ParseError(f"Undefined function '{ident}'")
                # Bare identifier: could be a function reference as argument
                if ident in self._func_params:
                    return Symbol(ident)
                # Or a known function passed by name
                if ident in _user_functions:
                    return _FuncRef(ident)
                raise ParseError(f"Unknown identifier '{ident}'")

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

    def _next_is_alpha(self, offset=0):
        """Check if the character at pos+offset is alphabetic."""
        idx = self.pos + offset
        return idx < len(self.text) and self.text[idx].isalpha()

    def _read_identifier(self):
        """Read a multi-character alphabetic identifier from current position.
        Returns the identifier string, or None if not at an alpha character.
        Backtracks the parser position if needed (caller already checked peek)."""
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            self.pos += 1
        if self.pos == start:
            return None
        return self.text[start:self.pos]

    def _parse_user_func_call(self, name):
        """Parse name(arg1, arg2, ...) for a user-defined function."""
        self.consume('(')
        args = []
        if self.peek() != ')':
            args.append(self.expr())
            while self.peek() == ',':
                self.consume(',')
                args.append(self.expr())
        self.consume(')')
        # Resolve _FuncRef arguments
        return _apply_user_function(name, args)

    def _parse_param_call(self, param_name):
        """Parse param_name(arg1, ...) inside a function body — creates a _UserCall node."""
        self.consume('(')
        args = []
        if self.peek() != ')':
            args.append(self.expr())
            while self.peek() == ',':
                self.consume(',')
                args.append(self.expr())
        self.consume(')')
        return _UserCall(param_name, *args)

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


# Regex to detect function definition: name(param1,param2,...)=
_FUNC_DEF_RE = re.compile(
    r'^([a-zA-Z][a-zA-Z]*)\(([a-zA-Z][a-zA-Z]*(?:,[a-zA-Z][a-zA-Z]*)*)?\)\s*='
)


def _is_func_definition(text):
    """Check if text matches function definition syntax. Returns (name, params, body_text) or None.
    Raises ParseError for reserved names so the user gets clear feedback."""
    m = _FUNC_DEF_RE.match(text)
    if not m:
        return None
    name = m.group(1)
    # Reject reserved names/prefixes with clear error
    if name in _RESERVED_NAMES:
        raise ParseError(f"'{name}' is a reserved name")
    for prefix in _RESERVED_PREFIXES:
        if name.startswith(prefix):
            raise ParseError(f"Function name cannot start with '{prefix}'")
    params = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
    body_text = text[m.end():].strip()
    if not body_text:
        return None
    return (name, params, body_text)


def parse_and_eval(text, allow_definition=False):
    """Parse a traction expression string and return the simplified result.

    Supports equation syntax: 'lhs = rhs' is solved as 'lhs - rhs = 0'.
    If allow_definition=True, also handles function definitions: name(params)=body.
    """
    text = text.strip().replace('\u00b7', '*')
    if not text:
        return None

    # Function definition: name(params)=body
    if allow_definition and '=' in text:
        defn = _is_func_definition(text)
        if defn is not None:
            name, params, body_text = defn
            # Parse the body with parameter names visible (not simplified)
            parser = Parser(body_text, func_params=set(params))
            body = parser.parse()
            return define_function(name, params, body)

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
            # Don't treat function definitions as equations in live preview
            if _FUNC_DEF_RE.match(text):
                return None
            lhs = Parser(lhs_text).parse()
            rhs = Parser(rhs_text).parse()
            return _do_solve(lhs - rhs)

    parser = Parser(text)
    result = parser.parse()
    if isinstance(result, SolutionSet):
        return result
    return traction_simplify(result)
