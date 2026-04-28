"""
COTT Calculator - Traction Expression Evaluator
A tkinter GUI for interactively evaluating Traction Theory expressions.

Usage:
    conda activate traction
    python calculator.py

This module serves as the entry point. All logic is in submodules:
    parser.py          — Expression parser and evaluator
    formatting.py      — Display formatting
    decomposition.py   — Chebyshev ring decomposition engine
    visualization.py   — Phase grid computation and RGB mapping
    fractal.py         — Escape-time fractal computation
    streamlines.py     — Gradient streamline computation
    gui/               — GUI package (app, fullscreen, settings, constants, utils)
"""

import tkinter as tk

# Re-export public API for backward compatibility
from parser import Parser, ParseError, SolutionSet, parse_and_eval, FunctionDef
from parser import get_user_functions, delete_user_function, define_function
from formatting import format_result, format_approx, format_complex, format_numeric_approx
from decomposition import chebyshev_decompose, _eval_ring_exact, _reduce_ring_form, _complex_at_pi2
from visualization import (compute_phase_grid, phase_to_rgb, magnitude_to_rgb, blended_to_rgb,
                           CANVAS_SIZE, GRID_RES, AXIS_MARGIN, CANVAS_TOTAL, DEFAULT_BOUNDS, PHASE_COLORS)
from fractal import compute_fractal, fractal_to_rgb, parse_fractal_args as _parse_fractal_args
from streamlines import compute_streamlines
from gui.app import CalculatorApp
from gui.fullscreen import FullScreenViewer
from gui.settings import SettingsWindow
from gui.constants import *
from gui.utils import nice_tick_step as _nice_tick_step, tick_label as _tick_label
from gui.utils import scale_color as _scale_color, clip_line as _clip_line


def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
