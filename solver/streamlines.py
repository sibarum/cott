"""
Gradient streamline computation.

Traces gradient flow lines and normal lines of log|f|
for overlay on phase plots.
"""

import numpy as np


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
