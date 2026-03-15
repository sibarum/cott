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
from sympy import symbols, lambdify
from traction import Zero, Omega, Null, Log0, LogW, z, w, null, traction_simplify, log0, logw, project_complex
import registry
import projections  # auto-discovers and registers projection plugins


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

        # Variables: p (horizontal), q (vertical), x (projection native unit)
        if ch == 'p':
            self.consume()
            return Symbol('p')
        if ch == 'q':
            self.consume()
            return Symbol('q')
        if ch == 'x':
            self.consume()
            return Symbol('x')
        # Legacy: y still works as alias for q
        if ch == 'y':
            self.consume()
            return Symbol('q')

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
        return func(arg)

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

# Phase model: cyan, magenta, yellow, teal (4 quadrants)
PHASE_COLORS = np.array([
    [0, 255, 255],      # cyan    — phase 0        (positive real)
    [255, 0, 255],      # magenta — phase pi/2     (positive imaginary)
    [255, 255, 0],      # yellow  — phase pi       (negative real)
    [0, 128, 128],      # teal    — phase 3*pi/2   (negative imaginary)
], dtype=np.float64)



def compute_phase_grid(expr_text, grid_res=GRID_RES, bounds=3.0, projection_name='complex_lie'):
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

    # Step 1: Symbolic projection
    projected = proj.project_expr(traction_expr, a, b)
    if projected is None:
        return None

    # Step 2: Grid evaluation
    lin = np.linspace(-bounds, bounds, grid_res)
    AA, BB = np.meshgrid(lin, lin[::-1])  # flip y so up = positive

    result = proj.eval_grid(projected, a, b, AA, BB)
    if result is None:
        return None

    # Return the standard tuple that the rest of the calculator expects
    return result['phase'], result['brightness'], result['Z'], result['log_mag']


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
        self.show_tangent = False
        self.show_normal = False
        self.show_diamond = False
        self.color_mode = 'phase'  # 'phase' or 'continuity'
        self.projection_names = registry.names('projection')
        self.projection_index = 0  # default to first registered (complex_lie)
        # Approximation mode: cycle through ~, ≃, ≈
        self.approx_modes = ['~', '\u2243', '\u2248']
        self.approx_index = 2  # default to ≃

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

        self.display_result = tk.Label(
            top_frame, text='', font=self.font_result,
            bg=BG_RESULT, fg=FG_RESULT, anchor='e', padx=10, pady=4,
            height=1, relief='sunken', bd=1
        )
        self.display_result.pack(fill='x')

        # Hidden stubs for removed display elements (code still references them)
        self.display_approx = tk.Label(top_frame)
        self.display_history = tk.Label(top_frame)

        # Horizontal container: calculator + visualization
        hframe = tk.Frame(border, bg=BG_FRAME)
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
            [('0', '0'),        ('\u03c9', '\u03c9'),('\u2248', None),('=', None),  ('x', 'x')],
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

        # Hidden stub buttons for settings window _set_toggle references
        self.tangent_btn = tk.Button(btn_row); self.tangent_btn._toggled = False; self.tangent_btn._is_toggle = True; self.tangent_btn._is_accent = False
        self.normal_btn = tk.Button(btn_row); self.normal_btn._toggled = False; self.normal_btn._is_toggle = True; self.normal_btn._is_accent = False
        self.color_btn = tk.Button(btn_row); self.color_btn._toggled = False; self.color_btn._is_toggle = True; self.color_btn._is_accent = False
        self.proj_btn = tk.Button(btn_row); self.proj_btn._toggled = False; self.proj_btn._is_toggle = True; self.proj_btn._is_accent = False

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
            proj_name = self.projection_names[self.projection_index]
            result = compute_phase_grid(expr_text, bounds=self.viz_bounds, projection_name=proj_name)
            if result is None:
                self.viz_canvas.delete('all')
                self.display_expr.focus_set()
                return

            phase, brightness, Z, log_mag = result
            self.viz_Z = Z  # store complex grid for hover readout
            self.viz_log_mag = log_mag
            if self.color_mode == 'continuity':
                rgb = continuity_to_rgb(phase, log_mag)
            else:
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
        fractions = [0, 1/4, 1/3, 1/2, 2/3, 3/4]
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
        self._set_toggle(self.tangent_btn, self.show_tangent)
        if self.viz_Z is not None:
            if self.show_tangent:
                self._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            else:
                self.viz_canvas.delete('tangent')
        self.display_expr.focus_set()

    def _toggle_normal(self):
        """Toggle normal lines (constant |f| contours) on/off."""
        self.show_normal = not self.show_normal
        self._set_toggle(self.normal_btn, self.show_normal)
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
        self._set_toggle(self.color_btn, self.color_mode == 'continuity')
        if self.viz_Z is not None:
            self._refresh_viz()
        self.display_expr.focus_set()

    def _cycle_projection(self):
        """Cycle through registered projection plugins."""
        self.projection_index = (self.projection_index + 1) % len(self.projection_names)
        name = self.projection_names[self.projection_index]
        label = name.replace('_', ' ')
        self.proj_btn.configure(text='P')
        if hasattr(self, 'viz_title_label'):
            self.viz_title_label.configure(text=f'Phase Plot [{label}]')
        if self.viz_Z is not None:
            self._refresh_viz()
        self.display_expr.focus_set()

    def _open_settings(self):
        """Open the settings window (single-instance)."""
        SettingsWindow(self)

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

        z = self.viz_Z[row, col]
        if not np.isfinite(z):
            self._clear_gauges()
            return

        self._draw_gauges(z.real, z.imag, abs(z), np.angle(z), p_val, q_val)

    def _on_viz_leave(self, event):
        self._clear_gauges()

    def _clear_gauges(self):
        self.gauge_canvas.delete('all')

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
        self.app._set_toggle(self.app.color_btn, self.app.color_mode == 'continuity')
        if self.app.viz_Z is not None:
            self.app._refresh_viz()

    def _on_tangent_change(self):
        self.app.show_tangent = self.tangent_var.get()
        self.app._set_toggle(self.app.tangent_btn, self.app.show_tangent)
        if self.app.viz_Z is not None:
            if self.app.show_tangent:
                self.app._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            else:
                self.app.viz_canvas.delete('tangent')

    def _on_normal_change(self):
        self.app.show_normal = self.normal_var.get()
        self.app._set_toggle(self.app.normal_btn, self.app.show_normal)
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
