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
from sympy import S, Integer, Rational, Pow, Mul, Add, Symbol
from sympy import I as symI, symbols, lambdify
from traction import Zero, Omega, Null, z, w, null, traction_simplify, log0, logw, project_complex


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
        primary  = '(' expr ')' | funccall | number | 'w' | 'x' | 'omega'
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

        # Function calls: log0(...), logw(...)
        if self.match('log0'):
            return self._parse_func_call(log0, 'log0')
        if self.match('log\u03c9'):
            return self._parse_func_call(logw, 'log\u03c9')
        if self.match('logw'):
            return self._parse_func_call(logw, 'logw')

        # 'w' (must check after 'logw')
        if ch == 'w':
            self.consume()
            return w

        # Variable x
        if ch == 'x':
            self.consume()
            return Symbol('x')

        # null
        if self.match('null'):
            return null

        # Number (integer or rational p/q)
        if ch.isdigit():
            return self.number()

        raise ParseError(f"Unexpected character '{ch}' at position {self.pos}")

    def _parse_func_call(self, func, name):
        """Parse func(expr) and return the result."""
        if self.peek() != '(':
            raise ParseError(f"Expected '(' after {name} at position {self.pos}")
        self.consume('(')
        arg = self.expr()
        self.consume(')')
        result = func(arg)
        if result is None:
            raise ParseError(f"{name} cannot simplify the given argument")
        return result

    def number(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
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
            if d == 0:
                # n/0 should be handled as division, not a rational
                self.pos = saved
                return _to_traction(n)
            return Rational(n, d)

        return _to_traction(n)


def _to_traction(n):
    """Convert a Python int to the appropriate traction type."""
    if n == 0:
        return z
    return Integer(n)


def parse_and_eval(text):
    """Parse a traction expression string and return the simplified result."""
    text = text.strip()
    if not text:
        return None
    parser = Parser(text)
    result = parser.parse()
    return traction_simplify(result)


# ============================================================
# Display Formatting
# ============================================================

def format_result(expr):
    """Format a traction expression for display."""
    if expr is None:
        return ''
    if isinstance(expr, Zero):
        return '0'
    if isinstance(expr, Omega):
        return '\u03c9'
    if isinstance(expr, Null):
        return '\u2205'  # empty set symbol
    if isinstance(expr, Symbol):
        return str(expr)
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
        parts = []
        for arg in expr.args:
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
    Recursively approximates numeric sub-parts within traction expressions.
    """
    if expr is None:
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

    # Replace re(x)/im(x) with x/y for readability (matches plot axes)
    proj_str = proj_str.replace('re(x)', 'x').replace('im(x)', 'y')

    return proj_str


def _format_sympy_complex(expr):
    """Format a SymPy complex expression for display."""
    from sympy import I, im, re, cos

    # If expression still contains traction types, format as mixed
    if expr.has(Zero) or expr.has(Omega) or expr.has(Null):
        return _format_mixed(expr)

    # Try to simplify to a+bi form
    try:
        expr = expr.rewrite(cos).simplify().expand()
    except Exception:
        pass

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
# Phase Visualization
# ============================================================

CANVAS_SIZE = 300
GRID_RES = 150
AXIS_MARGIN = 28  # pixels for tick labels
CANVAS_TOTAL = CANVAS_SIZE + AXIS_MARGIN
DEFAULT_BOUNDS = 3.0

# Phase quadrant colors: cyan, magenta, yellow, teal
PHASE_COLORS = np.array([
    [0, 255, 255],      # cyan    — phase 0        (positive real)
    [255, 0, 255],      # magenta — phase pi/2     (positive imaginary)
    [255, 255, 0],      # yellow  — phase pi       (negative real)
    [0, 128, 128],      # teal    — phase 3*pi/2   (negative imaginary)
], dtype=np.float64)


def compute_phase_grid(expr_text, grid_res=GRID_RES, bounds=3.0):
    """
    Compute a phase grid for a traction expression containing x.

    Substitutes x -> a + b*0^(w/2) (i.e. a + b*i in the complex projection),
    evaluates on a grid, and returns phase angles in [0, 2*pi).
    Returns None if the expression has no x or can't be evaluated.
    """
    parsed = parse_and_eval(expr_text)
    if parsed is None:
        return None

    x_sym = Symbol('x')
    if not parsed.has(x_sym):
        return None

    # Project traction elements to complex numbers
    proj = project_complex(parsed)

    # Substitute x -> a + b*i
    a, b = symbols('a b', real=True)
    complex_expr = proj.subs(x_sym, a + b * symI)
    try:
        complex_expr = complex_expr.expand()
    except Exception:
        pass

    # Lambdify for fast numeric evaluation
    try:
        f = lambdify((a, b), complex_expr, modules=['numpy'])
    except Exception:
        return None

    # Create grid (flip y so up = positive imaginary)
    lin = np.linspace(-bounds, bounds, grid_res)
    AA, BB = np.meshgrid(lin, lin[::-1])

    # Evaluate
    try:
        Z = f(AA, BB)
        if np.isscalar(Z) or (isinstance(Z, np.ndarray) and Z.ndim == 0):
            Z = np.full_like(AA, complex(Z), dtype=complex)
        Z = np.asarray(Z, dtype=complex)
    except Exception:
        return None

    # Phase in [0, 2*pi)
    phase = np.angle(Z)
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    # Mask invalid points (zero, inf, nan)
    invalid = ~np.isfinite(Z) | (np.abs(Z) < 1e-15)
    phase[invalid] = np.nan

    # Log-magnitude brightness: |z|=1 -> 0.5, |z|->inf -> ~1, |z|->0 -> ~0
    mag = np.abs(Z)
    log_mag = np.log(mag + 1e-15)
    brightness = 0.5 + np.arctan(log_mag) / np.pi  # maps to (0, 1)
    # Clamp to [0.12, 0.95] so colors stay visible
    brightness = np.clip(brightness, 0.12, 0.95)
    brightness[invalid] = 0.0

    return phase, brightness


def phase_to_rgb(phase_grid, brightness=None):
    """Map a 2D phase grid to an RGB array with smooth quadrant interpolation."""
    t = phase_grid / (2 * np.pi)  # normalize to [0, 1)
    segment = t * 4               # [0, 4)
    idx = segment.astype(int) % 4
    frac = (segment - segment.astype(int))[..., np.newaxis]

    c1 = PHASE_COLORS[idx]
    c2 = PHASE_COLORS[(idx + 1) % 4]
    rgb = c1 + frac * (c2 - c1)  # float [0, 255]

    # Modulate by brightness
    if brightness is not None:
        rgb = rgb * brightness[..., np.newaxis]

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # NaN -> dark gray
    nan_mask = np.isnan(phase_grid)
    rgb[nan_mask] = [40, 40, 40]

    return rgb


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
FG_TEXT = '#111111'
FG_DIM = '#555555'
FG_RESULT = '#1a1a2e'
BORDER_COLOR = '#888888'


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
        # Approximation mode: cycle through ~, ≃, ≈
        self.approx_modes = ['~', '\u2243', '\u2248']
        self.approx_index = 2  # default to ≃

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        # Outer border
        border = tk.Frame(self.root, bg=BG_FRAME, padx=12, pady=12)
        border.pack()

        # Horizontal container: calculator + visualization
        hframe = tk.Frame(border, bg=BG_FRAME)
        hframe.pack()

        # ===== Left: Calculator =====
        body = tk.Frame(hframe, bg=BG_BODY, bd=1, relief='solid')
        body.pack(side='left', padx=(0, 6))

        # Title
        title = tk.Label(body, text='COTT Calculator', font=self.font_label,
                         bg=BG_BODY, fg=FG_DIM)
        title.pack(pady=(8, 0))

        # Display area
        display_frame = tk.Frame(body, bg=BORDER_COLOR, padx=1, pady=1)
        display_frame.pack(padx=10, pady=(4, 0), fill='x')

        display_inner = tk.Frame(display_frame, bg=BG_DISPLAY)
        display_inner.pack(fill='both')

        self.entry_var = tk.StringVar()
        self.entry_var.trace_add('write', lambda *_: self._on_entry_change())
        self.display_expr = tk.Entry(
            display_inner, textvariable=self.entry_var, font=self.font_display,
            bg=BG_DISPLAY, fg=FG_TEXT, justify='right',
            bd=0, highlightthickness=0, insertbackground=FG_TEXT
        )
        self.display_expr.pack(fill='x', padx=10, pady=6)

        self.display_result = tk.Label(
            display_inner, text='', font=self.font_result,
            bg=BG_RESULT, fg=FG_RESULT, anchor='e', padx=10, pady=6,
            width=20, height=1, relief='sunken', bd=1
        )
        self.display_result.pack(fill='x')

        # Approximation display
        self.display_approx = tk.Label(
            display_inner, text='', font=self.font_result,
            bg=BG_DISPLAY, fg=FG_DIM, anchor='e', padx=10, pady=4,
            width=20, height=1
        )
        self.display_approx.pack(fill='x')

        # History label
        self.display_history = tk.Label(
            display_inner, text='', font=self.font_label,
            bg=BG_DISPLAY, fg=FG_DIM, anchor='e', padx=10,
            width=20
        )
        self.display_history.pack(fill='x')

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
            [('4', '4'),        ('5', '5'),      ('6', '6'),      ('\u2013', '-'), ('x', 'x')],
            [('1', '1'),        ('2', '2'),      ('3', '3'),      ('+', '+'),      ('i', '0^(\u03c9/2)')],
            [('0', '0'),        ('\u03c9', '\u03c9'),('\u2248', None),('=', None),  ('', None)],
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
                elif label in ('~', '\u2243', '\u2248'):
                    self.approx_btn = self._make_button(pad_frame, label, self._cycle_approx_mode)
                    btn = self.approx_btn
                elif value is not None:
                    small = label.startswith('log')
                    btn = self._make_button(pad_frame, label, lambda v=value: self._insert(v), small=small)
                else:
                    btn = self._make_button(pad_frame, label, None)
                btn.grid(row=row_idx, column=col_idx, padx=1, pady=1)

        # ===== Right: Phase Visualization =====
        viz_frame = tk.Frame(hframe, bg=BG_BODY, bd=1, relief='solid')
        viz_frame.pack(side='left', padx=(6, 0), anchor='n')

        viz_title = tk.Label(viz_frame, text='Phase Plot', font=self.font_label,
                             bg=BG_BODY, fg=FG_DIM)
        viz_title.pack(pady=(8, 4))

        self.viz_canvas = tk.Canvas(viz_frame, width=CANVAS_TOTAL, height=CANVAS_TOTAL,
                                    bg='#282828', highlightthickness=0)
        self.viz_canvas.pack(padx=10, pady=(0, 4))

        # Button row: zoom controls + plot
        btn_row = tk.Frame(viz_frame, bg=BG_BODY)
        btn_row.pack(pady=(4, 10))

        self._make_button(btn_row, '\u2013', self._zoom_out, small=True).pack(side='left', padx=1)
        self._make_button(btn_row, '\u25cb', self._zoom_reset, small=True).pack(side='left', padx=1)
        self._make_button(btn_row, '+', self._zoom_in, small=True).pack(side='left', padx=1)

    def _make_button(self, parent, label, command, accent=False, small=False):
        bg = '#5a7d9a' if accent else BG_BTN
        fg = 'white' if accent else FG_TEXT
        btn = tk.Button(
            parent, text=label, font=self.font_btn_small if small else self.font_btn,
            width=4, height=1, bd=1, relief='raised',
            bg=bg, fg=fg, activebackground=BG_BTN_ACTIVE,
            command=command
        )
        btn.bind('<Enter>', lambda e, b=btn, bg0=bg: b.configure(bg=BG_BTN_HOVER if not accent else '#6a9dba'))
        btn.bind('<Leave>', lambda e, b=btn, bg0=bg: b.configure(bg=bg0))
        return btn

    def _bind_keys(self):
        self.display_expr.bind('<Return>', lambda e: self._evaluate())
        self.display_expr.bind('<Escape>', lambda e: self._clear_all())
        # Intercept 'w' to insert ω instead
        self.display_expr.bind('w', self._insert_omega)
        self.display_expr.bind('i', self._insert_i)
        # Allow 'x' through as-is (default Entry behavior handles it)
        # Keep focus on the entry
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

    def _format_secondary(self, result):
        """Get the secondary display text based on the current approximation mode."""
        mode = self.approx_modes[self.approx_index]
        if mode == '\u2248':  # ≈ — complex projection
            return format_complex(result)
        else:  # ~ or ≃ — decimal approximation
            return format_approx(result)

    def _cycle_approx_mode(self):
        self.approx_index = (self.approx_index + 1) % len(self.approx_modes)
        symbol = self.approx_modes[self.approx_index]
        self.approx_btn.configure(text=symbol)
        self._update_live_preview()
        self.display_expr.focus_set()

    def _clear_all(self):
        self.entry_var.set('')
        self.history = []
        self.display_history.configure(text='')
        self.display_approx.configure(text='')
        self._update_live_preview()

    def _evaluate(self):
        expr_text = self.entry_var.get().strip()
        if not expr_text:
            return
        try:
            result = parse_and_eval(expr_text)
            result_str = format_result(result)

            # Save to history
            self.history.append(f'{expr_text} = {result_str}')
            if len(self.history) > 3:
                self.history.pop(0)

            self.display_result.configure(text=f'= {result_str}', fg=FG_RESULT)
            self.display_history.configure(text='\n'.join(self.history[-2:]))

            # Update secondary display (approximation or complex projection)
            secondary = self._format_secondary(result)
            if secondary:
                symbol = self.approx_modes[self.approx_index]
                self.display_approx.configure(text=f'{symbol} {secondary}', fg=FG_DIM)
            else:
                self.display_approx.configure(text='')

            # Refresh the phase plot (keeps expression in entry)
            self._refresh_viz()

        except (ParseError, Exception) as e:
            self.display_result.configure(text=f'Error: {e}', fg='#aa3333')
            self.display_approx.configure(text='')

    def _on_entry_change(self):
        """Live preview: evaluate as you type."""
        self._update_live_preview()

    def _update_live_preview(self):
        try:
            expr_text = self.entry_var.get().strip()
            if expr_text:
                result = parse_and_eval(expr_text)
                result_str = format_result(result)
                self.display_result.configure(text=f'= {result_str}', fg=FG_DIM)
                secondary = self._format_secondary(result)
                symbol = self.approx_modes[self.approx_index]
                if secondary:
                    self.display_approx.configure(text=f'{symbol} {secondary}')
                else:
                    self.display_approx.configure(text='')
            else:
                self.display_result.configure(text='')
                self.display_approx.configure(text='')
        except Exception:
            self.display_result.configure(text='...', fg=FG_DIM)
            self.display_approx.configure(text='')

    # ===== Visualization =====

    def _refresh_viz(self):
        """Compute and render the phase plot for the current expression."""
        expr_text = self.entry_var.get().strip()
        if not expr_text:
            self.viz_canvas.delete('all')
            self.display_expr.focus_set()
            return

        try:
            result = compute_phase_grid(expr_text, bounds=self.viz_bounds)
            if result is None:
                self.viz_canvas.delete('all')
                self.display_expr.focus_set()
                return

            phase, brightness = result
            rgb = phase_to_rgb(phase, brightness)
            self._render_viz(rgb)
        except Exception:
            self.viz_canvas.delete('all')
        self.display_expr.focus_set()

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
        self.viz_image = img.zoom(zoom, zoom)
        self.viz_canvas.delete('all')
        self.viz_canvas.create_image(AXIS_MARGIN, AXIS_MARGIN, anchor='nw', image=self.viz_image)

        # Draw axis markings
        self._draw_axes()

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

    def _zoom_in(self):
        self.viz_bounds = max(0.25, self.viz_bounds / 2)
        self._refresh_viz()

    def _zoom_out(self):
        self.viz_bounds = min(100, self.viz_bounds * 2)
        self._refresh_viz()

    def _zoom_reset(self):
        self.viz_bounds = DEFAULT_BOUNDS
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


def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
