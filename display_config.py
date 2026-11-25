"""Keep the studio's design coordinates independent of Windows display zoom."""
import os
import tkinter as tk
from tkinter import font as tkfont


def configure_windows_dpi():
    """Call before creating any Tk windows; Windows must not bitmap-scale them."""
    if os.name != 'nt':
        return
    import ctypes
    try:
        set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_bool
        if set_context(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    # Also covers interpreters whose manifest already fixed process awareness.
    try:
        set_thread = ctypes.windll.user32.SetThreadDpiAwarenessContext
        set_thread.argtypes = [ctypes.c_void_p]
        set_thread.restype = ctypes.c_void_p
        set_thread(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass


def configure_studio_display(root):
    # Font points and the existing pixel-based layout share a stable 96-DPI
    # coordinate system. Changing Windows zoom must not grow only the text.
    if os.name == 'nt':
        root.tk.call('tk', 'scaling', 96 / 72)


class DesignCanvas(tk.Canvas):
    """Draw in original design units, while the widget uses physical pixels."""
    design_scale = 1.0

    def winfo_width(self):
        return super().winfo_width() / self.design_scale

    def winfo_height(self):
        return super().winfo_height() / self.design_scale

    def _create(self, itemType, args, kw):
        def flatten(values):
            for value in values:
                if isinstance(value, (tuple, list)):
                    yield from flatten(value)
                else:
                    yield value
        coords = [float(v) * self.design_scale for v in flatten(args)]
        kw = dict(kw)
        for key in ('width', 'activewidth', 'disabledwidth'):
            if key in kw:
                kw[key] *= self.design_scale
        if 'font' in kw:
            family, size, *style = kw['font']
            kw['font'] = (family, -max(1, round(abs(size) * (4/3 if size > 0 else 1) * self.design_scale)), *style)
        return super()._create(itemType, coords, kw)

    def configure(self, cnf=None, **kw):
        if isinstance(cnf, dict):
            kw = {**cnf, **kw}
            cnf = None
        if 'scrollregion' in kw:
            kw['scrollregion'] = tuple(v * self.design_scale for v in kw['scrollregion'])
        return super().configure(cnf, **kw)

    config = configure

    def canvasx(self, screenx, gridspacing=None):
        return super().canvasx(screenx, gridspacing) / self.design_scale


class StudioLayout:
    """Scale spacing, controls, fonts and drawings together from a 1440×920 design.

    Always derive dimensions from the original values, avoiding rounding drift
    when resizing repeatedly or moving between displays.
    """
    def __init__(self, root, redraw, viewport):
        self.root, self.redraw, self.viewport = root, redraw, viewport
        self.scale = 1.0
        self.pending = None
        self.widgets = []
        self.fonts = []
        for name in tkfont.names(root):
            font = tkfont.nametofont(name, root)
            size = font.actual('size')
            self.fonts.append((font, abs(size) * (4/3 if size > 0 else 1)))
        self._collect(root)
        root.bind('<Configure>', self._resize, add='+')

    def _collect(self, parent):
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Toplevel):
                continue
            values = {}
            keys = widget.keys()
            options = ['padx', 'pady', 'borderwidth', 'highlightthickness', 'wraplength']
            if isinstance(widget, (tk.Frame, tk.Canvas, tk.Scale)):
                options += ['width', 'height', 'length', 'sliderlength']
            for key in options:
                if key in keys and not (widget is self.viewport and key == 'height'):
                    values[key] = widget.winfo_fpixels(str(widget.cget(key)))
            font = None
            if 'font' in keys:
                original = widget.cget('font')
                if str(original) not in tkfont.names(self.root):
                    font = tkfont.Font(root=self.root, font=original).actual()
            manager = widget.winfo_manager()
            spacing = {}
            if manager in ('pack', 'grid'):
                info = getattr(widget, manager + '_info')()
                for key in ('padx', 'pady', 'ipadx', 'ipady'):
                    if key in info:
                        value = info[key]
                        parts = (value,) if isinstance(value, (int, float)) else self.root.tk.splitlist(value)
                        spacing[key] = tuple(float(v) for v in parts)
            self.widgets.append((widget, values, font, manager, spacing))
            self._collect(widget)

    def _resize(self, event):
        if event.widget is not self.root:
            return
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(80, self.fit)

    def fit(self):
        self.pending = None
        scale = max(.5, min(self.root.winfo_width()/1440, self.root.winfo_height()/920))
        if abs(scale - self.scale) < .005:
            return
        self.scale = scale
        for font, size in self.fonts:
            font.configure(size=-max(1, round(size * scale)))
        for widget, values, font, manager, spacing in self.widgets:
            if isinstance(widget, DesignCanvas):
                widget.design_scale = scale
            options = {key: round(value * scale) for key, value in values.items()}
            if font:
                size = font['size']
                options['font'] = (font['family'], -max(1, round(abs(size)*(4/3 if size > 0 else 1)*scale)),
                                   font['weight'], font['slant'])
            if options:
                widget.configure(**options)
            if spacing:
                getattr(widget, manager + '_configure')(**{
                    key: tuple(round(v * scale) for v in vals) for key, vals in spacing.items()})
        self.root.update_idletasks()
        self.redraw()

