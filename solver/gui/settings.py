"""
Settings window with tabbed interface.
"""

import tkinter as tk
from tkinter import font as tkfont

import registry
from gui.constants import *


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
        if self.app.viz_Z is not None:
            self.app._refresh_viz()

    def _on_tangent_change(self):
        self.app.show_tangent = self.tangent_var.get()
        if self.app.viz_Z is not None:
            if self.app.show_tangent:
                self.app._draw_flow_lines(normal=False, tag='tangent', color='#ffffff')
            else:
                self.app.viz_canvas.delete('tangent')

    def _on_normal_change(self):
        self.app.show_normal = self.normal_var.get()
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

