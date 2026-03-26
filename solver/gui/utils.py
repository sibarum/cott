"""
GUI utility functions: tick formatting, color scales, line clipping.
"""


def nice_tick_step(bounds):
    """Choose a nice tick interval for the given bounds."""
    # Target ~5-7 ticks across the full range
    raw_step = (2 * bounds) / 6
    # Round to a nice number: 0.25, 0.5, 1, 2, 5, 10, ...
    nice = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
    for n in nice:
        if n >= raw_step:
            return n
    return raw_step


def tick_label(val):
    """Format a tick value concisely."""
    if val == int(val):
        return str(int(val))
    return f'{val:g}'


def scale_color(t, negative=False):
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


def clip_line(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
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
