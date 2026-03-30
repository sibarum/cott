"""
Phase visualization pipeline.

Computes phase grids, RGB mappings, and graded-element visualizations
for traction expressions over 2D coordinate grids.
"""

import numpy as np
from sympy import S, Integer, Pow, Mul, Add, Symbol, I, symbols, lambdify
from traction import Zero, Omega, Null, GradedElement, traction_simplify
from parser import parse_and_eval
import registry

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
                       projection_name='complex_lie', t_value=None):
    """
    Compute visualization data for a traction expression using a registered projection.

    Variable system:
        p, q — raw grid coordinates (horizontal, vertical). Always the same.
        x    — projection's native unit coordinate. Defined by each projection.
               e.g. complex_lie: x = p + q*0^(w/2)
        t    — time parameter (scalar, 0 to 1). Substituted before grid eval.

    Returns tuple (phase, brightness, Z, log_mag) or None on failure.
    """
    parsed = parse_and_eval(expr_text)
    if parsed is None:
        return None

    p_sym = Symbol('p')
    q_sym = Symbol('q')
    x_sym = Symbol('x')
    t_sym = Symbol('t')
    has_p = parsed.has(p_sym)
    has_q = parsed.has(q_sym)
    has_x = parsed.has(x_sym)
    has_t = parsed.has(t_sym)

    if not has_p and not has_q and not has_x:
        return None

    # Substitute t with its scalar value before grid evaluation
    if has_t:
        from sympy import Rational
        tv = Rational(t_value) if t_value is not None else Rational(0)
        parsed = parsed.subs(t_sym, tv)

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
