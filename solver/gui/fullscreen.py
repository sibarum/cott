"""
Full-screen plot viewer with pan, zoom, and progressive rendering.
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import threading

from traction import Zero, Omega, Null, GradedElement, traction_simplify
import registry
from parser import Parser, parse_and_eval
from visualization import (compute_phase_grid, phase_to_rgb, continuity_to_rgb,
                           CANVAS_SIZE, GRID_RES, AXIS_MARGIN, CANVAS_TOTAL, DEFAULT_BOUNDS)
from fractal import compute_fractal, fractal_to_rgb, parse_fractal_args
from streamlines import compute_streamlines
from gui.constants import *
from gui.utils import nice_tick_step, tick_label, scale_color, clip_line


class FullScreenViewer:
    """Full-screen plot viewer with pan, zoom, and progressive rendering.

    Interaction model: during drag/zoom the existing rendered image is
    repositioned instantly with no re-computation.  A coordinate grid
    overlay provides spatial reference.  Rendering is debounced and
    fires 200 ms after the last interaction event.
    """

    _instance = None

    def __init__(self, app):
        # Single-instance guard
        if FullScreenViewer._instance is not None:
            try:
                FullScreenViewer._instance.win.lift()
                return
            except tk.TclError:
                FullScreenViewer._instance = None
        FullScreenViewer._instance = self

        self.app = app
        self.bounds = app.viz_bounds
        self.center_p = 0.0
        self.center_q = 0.0

        # View at time of last completed render (for repositioning)
        self._rendered_center_p = 0.0
        self._rendered_center_q = 0.0
        self._rendered_bounds = app.viz_bounds

        # Drag state
        self._drag_start = None
        self._drag_center_start = None

        # Rendering state
        self._render_cancel = None
        self._render_after_id = None

        # Build window
        self.win = tk.Toplevel(app.root)
        self.win.title('COTT \u2014 Full Screen Viewer')
        self.win.configure(bg='#000000')
        self.win.attributes('-fullscreen', True)
        self.win.protocol('WM_DELETE_WINDOW', self._close)
        self.win.lift()
        self.win.focus_force()

        # Canvas fills entire window
        self.canvas = tk.Canvas(self.win, bg='#000000', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)

        # Controls legend
        self._legend_visible = True

        # Dual-layer image buffers (low-res persists behind high-res)
        # Each layer: base PhotoImage, cached scaled version, view params, cache key
        self._photo_low_base = None     # raw low-res PhotoImage
        self._photo_low_scaled = None   # current display-scaled version
        self._low_scale_key = None      # (zoom_numer, zoom_denom) cache key
        self._low_center_p = 0.0        # view params when low-res was rendered
        self._low_center_q = 0.0
        self._low_bounds = app.viz_bounds
        self._has_low = False

        self._photo_high_base = None
        self._photo_high_scaled = None
        self._high_scale_key = None
        self._high_center_p = 0.0
        self._high_center_q = 0.0
        self._high_bounds = app.viz_bounds
        self._has_high = False

        # Spinner state
        self._spinner_after_id = None
        self._spinner_frame = 0
        self._rendering = False

        # Zoom velocity state
        self._zoom_velocity = 0.0       # log-scale velocity (positive = zoom in)
        self._zoom_mouse = (0, 0)       # mouse position to zoom toward
        self._zoom_anim_id = None       # after() id for animation tick

        # Bindings
        self.win.bind('<Escape>', lambda e: self._close())
        self.win.bind('<Key-h>', lambda e: self._toggle_legend())
        self.canvas.bind('<ButtonPress-1>', self._on_drag_start)
        self.canvas.bind('<B1-Motion>', self._on_drag_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_drag_end)
        self.canvas.bind('<MouseWheel>', self._on_scroll_zoom)
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Leave>', self._on_leave)
        self.canvas.bind('<Configure>', self._on_resize)

        # Initial render after window maps
        self.win.after(50, self._start_render)

    # ----------------------------------------------------------------
    # Geometry helpers
    # ----------------------------------------------------------------

    def _canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            w = self.win.winfo_screenwidth()
            h = self.win.winfo_screenheight()
        return w, h

    def _view_bounds(self, center_p=None, center_q=None, bounds=None):
        """Return (p_min, p_max, q_min, q_max) for a given view."""
        if center_p is None:
            center_p = self.center_p
        if center_q is None:
            center_q = self.center_q
        if bounds is None:
            bounds = self.bounds
        cw, ch = self._canvas_size()
        aspect = cw / max(ch, 1)
        half_q = bounds
        half_p = bounds * aspect
        return (center_p - half_p, center_p + half_p,
                center_q - half_q, center_q + half_q)

    # ----------------------------------------------------------------
    # Coordinate grid overlay
    # ----------------------------------------------------------------

    def _draw_grid(self):
        """Draw a coordinate grid matching the current view."""
        self.canvas.delete('grid')
        cw, ch = self._canvas_size()
        p_min, p_max, q_min, q_max = self._view_bounds()

        tick_step = nice_tick_step(self.bounds)
        grid_color = '#1a1a2e'
        label_color = '#444466'
        font = tkfont.Font(family='Consolas', size=9)

        # Vertical lines (constant p)
        p = p_min - (p_min % tick_step)
        if p < p_min:
            p += tick_step
        while p <= p_max:
            sx = (p - p_min) / (p_max - p_min) * cw
            is_origin = abs(p) < tick_step * 0.01
            color = '#2a2a44' if is_origin else grid_color
            width = 2 if is_origin else 1
            self.canvas.create_line(sx, 0, sx, ch, fill=color, width=width, tags='grid')
            self.canvas.create_text(sx + 4, ch - 4, text=tick_label(round(p, 10)),
                                    font=font, fill=label_color, anchor='sw', tags='grid')
            p += tick_step

        # Horizontal lines (constant q)
        q = q_min - (q_min % tick_step)
        if q < q_min:
            q += tick_step
        while q <= q_max:
            sy = (q_max - q) / (q_max - q_min) * ch
            is_origin = abs(q) < tick_step * 0.01
            color = '#2a2a44' if is_origin else grid_color
            width = 2 if is_origin else 1
            self.canvas.create_line(0, sy, cw, sy, fill=color, width=width, tags='grid')
            self.canvas.create_text(4, sy - 4, text=tick_label(round(q, 10)),
                                    font=font, fill=label_color, anchor='sw', tags='grid')
            q += tick_step

    # ----------------------------------------------------------------
    # Image repositioning (no re-render)
    # ----------------------------------------------------------------

    def _scale_and_place(self, tag, base_img, scaled_cache, scale_key_cache,
                         rendered_cp, rendered_cq, rendered_bounds):
        """Scale and position an image layer to match the current view.

        Uses rational zoom/subsample (denominators 1–4) for smooth steps,
        cached by (numer, denom) key so unchanged frames are free.
        Returns (new_scaled, new_key).
        """
        if base_img is None:
            return scaled_cache, scale_key_cache

        cw, ch = self._canvas_size()
        base_w = base_img.width()
        if base_w < 1:
            return scaled_cache, scale_key_cache

        bounds_ratio = rendered_bounds / max(self.bounds, 1e-12)
        ratio = cw * bounds_ratio / base_w

        # Best rational approximation with small denominator
        best_n, best_d, best_err = 1, 1, float('inf')
        for d in range(1, 5):
            n = max(1, round(ratio * d))
            err = abs(n / d - ratio)
            if err < best_err:
                best_n, best_d, best_err = n, d, err

        key = (best_n, best_d)
        if key != scale_key_cache:
            scaled_cache = base_img.zoom(best_n, best_n)
            if best_d > 1:
                scaled_cache = scaled_cache.subsample(best_d, best_d)
            scale_key_cache = key

        # Position: map rendered center to current screen coords
        cp_min, cp_max, cq_min, cq_max = self._view_bounds()
        cur_p_range = cp_max - cp_min
        cur_q_range = cq_max - cq_min

        scr_cx = (rendered_cp - cp_min) / cur_p_range * cw
        scr_cy = (cq_max - rendered_cq) / cur_q_range * ch

        img_w = scaled_cache.width() if scaled_cache else 0
        img_h = scaled_cache.height() if scaled_cache else 0
        ox = int(scr_cx - img_w / 2)
        oy = int(scr_cy - img_h / 2)

        self.canvas.create_image(ox, oy, anchor='nw',
                                 image=scaled_cache, tags=tag)
        return scaled_cache, scale_key_cache

    def _reposition_images(self):
        """Rescale and reposition both image layers to match the current view."""
        self.canvas.delete('img_low')
        self.canvas.delete('img_high')

        if self._has_low:
            self._photo_low_scaled, self._low_scale_key = self._scale_and_place(
                'img_low', self._photo_low_base, self._photo_low_scaled,
                self._low_scale_key,
                self._low_center_p, self._low_center_q, self._low_bounds)

        if self._has_high:
            self._photo_high_scaled, self._high_scale_key = self._scale_and_place(
                'img_high', self._photo_high_base, self._photo_high_scaled,
                self._high_scale_key,
                self._high_center_p, self._high_center_q, self._high_bounds)

        # Layer order: low under high, grid and overlays on top
        if self._has_low and self._has_high:
            self.canvas.tag_raise('img_high', 'img_low')
        self.canvas.tag_raise('grid')
        self.canvas.tag_raise('spinner')
        self.canvas.tag_raise('legend')
        self.canvas.tag_raise('coord')

    # ----------------------------------------------------------------
    # Rendering pipeline
    # ----------------------------------------------------------------

    def _cancel_pending(self):
        if self._render_cancel is not None:
            self._render_cancel.set()
        if self._render_after_id is not None:
            self.win.after_cancel(self._render_after_id)
            self._render_after_id = None
        if self._zoom_anim_id is not None:
            self.win.after_cancel(self._zoom_anim_id)
            self._zoom_anim_id = None
            self._zoom_velocity = 0.0

    def _schedule_render(self):
        """Debounce: schedule a render 200ms from now, cancelling any prior."""
        self._cancel_pending()
        self._render_after_id = self.win.after(1000, self._start_render)

    def _start_render(self):
        """Begin progressive render: low-res immediately, high-res after."""
        # If expression was cleared, close the viewer
        if not self.app.entry_var.get().strip():
            self._close()
            return
        self._cancel_pending()
        self._show_spinner()
        self._do_render(quality='low')

    def _do_render(self, quality='high'):
        """Render the current view in a background thread."""
        import threading

        cw, ch = self._canvas_size()
        if cw < 2 or ch < 2:
            return

        # Pick a uniform zoom divisor so base_w * zoom == cw and base_h * zoom == ch.
        # This avoids gaps from mismatched integer scaling.
        if quality == 'low':
            divisor = max(4, min(cw, ch) // 150)
        else:
            divisor = max(1, min(cw, ch) // 800)
        res_x = max(1, cw // divisor)
        res_y = max(1, ch // divisor)
        res = (res_x, res_y, divisor)

        app = self.app
        is_fractal = getattr(app, '_fractal_mode', False) and hasattr(app, '_fractal_raw_text')

        cancel = threading.Event()
        self._render_cancel = cancel

        # Snapshot current view and all tkinter state on the main thread
        # (StringVar.get() and widget reads are not thread-safe)
        snap_center_p = self.center_p
        snap_center_q = self.center_q
        snap_bounds = self.bounds
        snap_expr = app.entry_var.get().strip()
        snap_proj_name = app.projection_names[app.projection_index]
        snap_color_mode = app.color_mode
        snap_fractal_text = getattr(app, '_fractal_raw_text', '') if is_fractal else ''

        def compute():
            try:
                if is_fractal:
                    rgb = self._compute_fractal_rgb(res, cw, ch, cancel,
                                                   snap_center_p, snap_center_q, snap_bounds,
                                                   snap_fractal_text)
                else:
                    rgb = self._compute_phase_rgb(res, cw, ch,
                                                 snap_center_p, snap_center_q, snap_bounds,
                                                 snap_expr, snap_proj_name, snap_color_mode)
                if cancel.is_set() or rgb is None:
                    return
                self.win.after(0, lambda: self._finish_render(
                    rgb, cw, ch, quality, cancel,
                    snap_center_p, snap_center_q, snap_bounds))
            except Exception:
                self.win.after(0, self._hide_spinner)

        t = threading.Thread(target=compute, daemon=True)
        t.start()

    def _finish_render(self, rgb, cw, ch, quality, cancel,
                       snap_center_p, snap_center_q, snap_bounds):
        """Display a completed render result into the appropriate layer."""
        if cancel.is_set():
            return
        try:
            self.win.winfo_exists()
        except tk.TclError:
            return

        h, w, _ = rgb.shape

        img = tk.PhotoImage(width=w, height=h)
        for y in range(h):
            row = ' '.join(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in rgb[y])
            img.put(f'{{{row}}}', to=(0, y))

        if quality == 'low':
            self._photo_low_base = img
            self._photo_low_scaled = None
            self._low_scale_key = None  # invalidate cache — reposition will rebuild
            self._low_center_p = snap_center_p
            self._low_center_q = snap_center_q
            self._low_bounds = snap_bounds
            self._has_low = True
            # Clear stale high-res (it's from the old view)
            self._has_high = False
            self.canvas.delete('img_high')
        else:
            self._photo_high_base = img
            self._photo_high_scaled = None
            self._high_scale_key = None
            self._high_center_p = snap_center_p
            self._high_center_q = snap_center_q
            self._high_bounds = snap_bounds
            self._has_high = True

        # Position both layers and overlays
        self.canvas.delete('grid')
        self._reposition_images()
        self._draw_legend()

        if quality == 'low':
            # Schedule high-res refinement
            self._render_after_id = self.win.after(200, lambda: self._do_render(quality='high'))
        else:
            # High-res done — hide spinner
            self._hide_spinner()

    # ----------------------------------------------------------------
    # Rendering spinner
    # ----------------------------------------------------------------

    _SPINNER_CHARS = ['\u25dc', '\u25dd', '\u25de', '\u25df']  # ◜ ◝ ◞ ◟

    def _show_spinner(self):
        """Show an animated rendering indicator in the top-left."""
        self._rendering = True
        self._spinner_frame = 0
        self._tick_spinner()

    def _hide_spinner(self):
        """Remove the rendering indicator."""
        self._rendering = False
        if self._spinner_after_id is not None:
            self.win.after_cancel(self._spinner_after_id)
            self._spinner_after_id = None
        self.canvas.delete('spinner')

    def _tick_spinner(self):
        """Advance the spinner animation by one frame."""
        if not self._rendering:
            return
        self.canvas.delete('spinner')

        ch = self._SPINNER_CHARS[self._spinner_frame % len(self._SPINNER_CHARS)]
        font_icon = tkfont.Font(family='Segoe UI', size=16)
        font_label = tkfont.Font(family='Consolas', size=10)

        # Background pill
        self.canvas.create_rectangle(
            12, 12, 140, 42, fill='#000000', stipple='gray50',
            outline='#444444', tags='spinner')
        self.canvas.create_text(
            24, 27, text=ch, font=font_icon, fill='#88ccff',
            anchor='w', tags='spinner')
        self.canvas.create_text(
            48, 27, text='Rendering\u2026', font=font_label, fill='#888888',
            anchor='w', tags='spinner')

        self._spinner_frame += 1
        self._spinner_after_id = self.win.after(150, self._tick_spinner)

    # ----------------------------------------------------------------
    # Compute helpers (phase / graded / fractal)
    # ----------------------------------------------------------------

    def _compute_phase_rgb(self, res, cw, ch, center_p, center_q, bounds,
                           expr_text, proj_name, color_mode):
        """Compute phase grid RGB for a specific view.

        All tkinter-derived values (expr_text, proj_name, color_mode) must be
        snapshotted on the main thread and passed in — StringVar.get() is not
        thread-safe.
        """
        if not expr_text:
            return None

        p_min, p_max, q_min, q_max = self._view_bounds(center_p, center_q, bounds)
        res_x, res_y, _divisor = res

        lin_p = np.linspace(p_min, p_max, res_x)
        lin_q = np.linspace(q_max, q_min, res_y)

        from sympy import Symbol, symbols
        p_sym, q_sym, x_sym = Symbol('p'), Symbol('q'), Symbol('x')
        a, b = symbols('a b', real=True)

        parsed = parse_and_eval(expr_text)
        if parsed is None:
            return None

        proj = registry.get('projection', proj_name)
        if proj is None:
            return None

        has_p = parsed.has(p_sym)
        has_q = parsed.has(q_sym)
        has_x = parsed.has(x_sym)
        if not has_p and not has_q and not has_x:
            return None

        subs = []
        if has_p:
            subs.append((p_sym, a))
        if has_q:
            subs.append((q_sym, b))
        if has_x:
            subs.append((x_sym, proj.native_x(a, b)))

        traction_expr = traction_simplify(parsed.subs(subs))

        if traction_expr.has(GradedElement):
            return self._compute_graded_rgb(traction_expr, a, b, lin_p, lin_q)

        projected = proj.project_expr(traction_expr, a, b)
        if projected is None:
            return None

        AA, BB = np.meshgrid(lin_p, lin_q)
        eval_result = proj.eval_grid(projected, a, b, AA, BB,
                                     traction_expr=traction_expr)
        if eval_result is None:
            return None

        phase = eval_result['phase']
        brightness = eval_result['brightness']
        log_mag = eval_result.get('log_mag')

        if color_mode == 'continuity':
            return continuity_to_rgb(phase, log_mag)
        else:
            return phase_to_rgb(phase, brightness)

    def _compute_graded_rgb(self, traction_expr, a, b, lin_p, lin_q):
        from sympy import lambdify
        from traction import Omega

        stripped = traction_expr
        if isinstance(stripped, GradedElement):
            stripped = stripped.value
        projected = stripped.subs(Omega(), 1j)

        try:
            f = lambdify((a, b), projected, modules='numpy')
        except Exception:
            return None

        AA, BB = np.meshgrid(lin_p, lin_q)
        try:
            Z = np.asarray(f(AA, BB), dtype=complex)
        except Exception:
            return None

        phase = np.angle(Z) % (2 * np.pi)
        brightness = 1 - 1 / (1 + np.abs(Z) ** 0.4)
        return phase_to_rgb(phase, brightness)

    def _compute_fractal_rgb(self, res, cw, ch, cancel, center_p, center_q, bounds,
                             fractal_text):
        expr_str, escape, max_iter = parse_fractal_args(fractal_text)

        from sympy import Symbol, lambdify, I as symI
        c_sym, x_sym = Symbol('c'), Symbol('x')
        p_sym, q_sym = Symbol('p'), Symbol('q')

        parsed = Parser(expr_str).parse()
        projected = parsed.subs(Zero(), symI).subs(Omega(), -symI).subs(Null(), 0)

        has_pq = projected.has(p_sym) or projected.has(q_sym)
        if has_pq:
            from sympy import re as sym_re, im as sym_im
            projected = projected.subs(p_sym, sym_re(c_sym)).subs(q_sym, sym_im(c_sym))

        f_raw = lambdify((x_sym, c_sym), projected, modules='numpy')

        p_min, p_max, q_min, q_max = self._view_bounds(center_p, center_q, bounds)
        res_x, res_y, _divisor = res

        lin_p = np.linspace(p_min, p_max, res_x)
        lin_q = np.linspace(q_max, q_min, res_y)
        AA, BB = np.meshgrid(lin_p, lin_q)
        c_grid = AA + 1j * BB

        z = np.full_like(c_grid, 1j, dtype=complex)
        counts = np.zeros(c_grid.shape, dtype=int)
        last_z = np.zeros_like(c_grid, dtype=complex)
        mask = np.ones(c_grid.shape, dtype=bool)

        for n in range(1, max_iter + 1):
            if cancel is not None and cancel.is_set():
                return None
            try:
                z[mask] = f_raw(z[mask], c_grid[mask])
            except Exception:
                break
            bad = ~np.isfinite(z)
            z[bad] = escape + 1
            escaped = mask & (np.abs(z) > escape)
            counts[escaped] = n
            last_z[escaped] = z[escaped]
            mask &= ~escaped
            if not mask.any():
                break

        last_z[mask] = z[mask]
        return fractal_to_rgb(counts, last_z, max_iter, escape)

    # ----------------------------------------------------------------
    # Controls legend
    # ----------------------------------------------------------------

    def _draw_legend(self):
        self.canvas.delete('legend')
        if not self._legend_visible:
            return

        cw, _ = self._canvas_size()
        lines = [
            'Drag        Pan',
            'Scroll      Zoom',
            'Esc         Close',
            'H           Toggle this legend',
        ]
        x = cw - 16
        y = 16
        font = tkfont.Font(family='Consolas', size=10)

        line_h = 18
        box_w = 280
        box_h = len(lines) * line_h + 16
        self.canvas.create_rectangle(
            x - box_w, y, x, y + box_h,
            fill='#000000', stipple='gray50', outline='#444444', tags='legend')

        y += 10
        for line in lines:
            self.canvas.create_text(
                x - box_w + 12, y, text=line, font=font,
                fill='#cccccc', anchor='nw', tags='legend')
            y += line_h

    def _toggle_legend(self):
        self._legend_visible = not self._legend_visible
        self._draw_legend()

    # ----------------------------------------------------------------
    # Hover readout
    # ----------------------------------------------------------------

    def _on_hover(self, event):
        if self._drag_start is not None:
            return  # suppress readout during drag
        self.canvas.delete('coord')

        cw, ch = self._canvas_size()
        p_min, p_max, q_min, q_max = self._view_bounds()
        p_val = p_min + (event.x / cw) * (p_max - p_min)
        q_val = q_max - (event.y / ch) * (q_max - q_min)

        font = tkfont.Font(family='Consolas', size=10)
        text = f'p = {p_val:+.4f}   q = {q_val:+.4f}   bounds = {self.bounds:.4g}'
        self.canvas.create_text(
            12, ch - 12, text=text, font=font,
            fill='#aaaaaa', anchor='sw', tags='coord')

    def _on_leave(self, event):
        self.canvas.delete('coord')

    # ----------------------------------------------------------------
    # Pan (click-drag): reposition image, draw grid, debounce render
    # ----------------------------------------------------------------

    def _on_drag_start(self, event):
        self._cancel_pending()
        self._drag_start = (event.x, event.y)
        self._drag_center_start = (self.center_p, self.center_q)
        self.canvas.configure(cursor='fleur')

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return

        cw, ch = self._canvas_size()
        p_min, p_max, q_min, q_max = self._view_bounds(
            self._drag_center_start[0], self._drag_center_start[1], self.bounds)
        p_range = p_max - p_min
        q_range = q_max - q_min

        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]

        self.center_p = self._drag_center_start[0] - dx / cw * p_range
        self.center_q = self._drag_center_start[1] + dy / ch * q_range

        self._draw_grid()
        self._reposition_images()

    def _on_drag_end(self, event):
        self._drag_start = None
        self.canvas.configure(cursor='crosshair')
        self._schedule_render()

    # ----------------------------------------------------------------
    # Scroll zoom with velocity / inertia
    # ----------------------------------------------------------------

    _ZOOM_IMPULSE = 0.18       # log-scale impulse per scroll tick
    _ZOOM_FRICTION = 0.5      # velocity multiplier per animation frame
    _ZOOM_TICK_MS = 60
    _ZOOM_STOP_THRESHOLD = 0.003  # velocity below this = stopped

    def _on_scroll_zoom(self, event):
        """Accumulate zoom velocity from scroll events."""
        self._cancel_pending()
        # Add impulse: scroll-up → negative log-velocity → zoom in (bounds shrinks)
        impulse = -self._ZOOM_IMPULSE if event.delta > 0 else self._ZOOM_IMPULSE
        self._zoom_velocity += impulse
        self._zoom_mouse = (event.x, event.y)

        # Start animation if not already running
        if self._zoom_anim_id is None:
            self._zoom_anim_tick()

    def _zoom_anim_tick(self):
        """One frame of zoom inertia animation."""
        self._zoom_anim_id = None

        if abs(self._zoom_velocity) < self._ZOOM_STOP_THRESHOLD:
            # Velocity depleted — stop animating and schedule render
            self._zoom_velocity = 0.0
            self._schedule_render()
            return

        # Apply velocity: factor = e^velocity ≈ 1 + velocity for small values
        import math
        factor = math.exp(self._zoom_velocity)

        # Zoom toward last mouse position
        cw, ch = self._canvas_size()
        mx, my = self._zoom_mouse
        p_min, p_max, q_min, q_max = self._view_bounds()

        mp = p_min + (mx / cw) * (p_max - p_min)
        mq = q_max - (my / ch) * (q_max - q_min)

        self.center_p = mp + (self.center_p - mp) * factor
        self.center_q = mq + (self.center_q - mq) * factor
        self.bounds = max(0.01, min(500, self.bounds * factor))

        # Apply friction
        self._zoom_velocity *= self._ZOOM_FRICTION

        # Update visuals (no render — just reposition)
        self._draw_grid()
        self._reposition_images()

        # Schedule next frame
        self._zoom_anim_id = self.win.after(self._ZOOM_TICK_MS, self._zoom_anim_tick)

    # ----------------------------------------------------------------
    # Resize
    # ----------------------------------------------------------------

    def _on_resize(self, event):
        self._cancel_pending()
        self._render_after_id = self.win.after(200, self._start_render)

    # ----------------------------------------------------------------
    # Close
    # ----------------------------------------------------------------

    def _close(self):
        self._cancel_pending()
        self._hide_spinner()
        FullScreenViewer._instance = None
        self.win.destroy()


