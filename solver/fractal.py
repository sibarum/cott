"""
Fractal computation engine.

Computes escape-time fractals on complex grids with traction-aware
initial conditions and smooth HSV colouring.
"""

import numpy as np


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


def parse_fractal_args(text):
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
