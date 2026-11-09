"""Modern dark "studio" skin for the AI Music Co-Creator.

This module only replaces the *presentation* of ``gui.MusicApp``.  Every
behaviour — note-guided generation (LSTM or rule-based fallback), model
selection, opt-in LoRA, playback, undo/clear, session autosave, interaction-log
and MIDI export — is inherited unchanged from ``gui.MusicApp``.

Launch with ``python -B UI/modern_main.py`` (the original UI stays on ``main.py``).
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from pathlib import Path

# The UI lives in UI/; make the project root (gui.py, config.py, ml/ ...) importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from config import LOG_DIR, NOTES
from gui import PITCH, MusicApp
from midi_player import MidiPlayer

# --------------------------------------------------------------------------- palette
BG = "#0b0b11"
SIDEBAR = "#0e0e15"
PANEL = "#111118"
CARD = "#17171f"
CARD_HI = "#1e1e29"
BORDER = "#25252f"
TEXT = "#f3f1f9"
MUTED = "#8d8b9c"
FAINT = "#57556a"
PURPLE = "#8b5cf6"
VIOLET = "#a855f7"
PINK = "#ec4899"
HUMAN = "#38bdf8"   # human cue (blue, as in the original legend)
AI = "#b26bff"      # AI continuation (purple, as in the original legend)
OK_GREEN = "#4ade80"
AMBER = "#fbbf24"

NOTE_HUES = {"C": "#8b5cf6", "D": "#6366f1", "E": "#3b82f6", "F": "#06b6d4",
             "G": "#d946ef", "A": "#ec4899", "B": "#f43f5e"}
SOLFEGE = {"C": "Do", "D": "Re", "E": "Mi", "F": "Fa", "G": "Sol", "A": "La", "B": "Si"}
DURATIONS = (("0.5 s", "0.5 seconds"), ("1 s", "1 second"), ("2 s", "2 seconds"))
MODELS = (("◆", "Baseline", "LSTM 3×512", "LSTM 3x512 (baseline)"),
          ("◈", "Experimental", "LSTM 4×768", "LSTM 4x768 (experimental)"))
BLACK_KEYS = {1, 3, 6, 8, 10}
FAMILY = "Segoe UI"


# --------------------------------------------------------------------------- drawing helpers
def _rgb(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    return int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two ``#rrggbb`` colours (t=0 → a, t=1 → b)."""
    t = max(0.0, min(1.0, t))
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t),
                              round(ba + (bb - ba) * t))


def round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, splinesteps=16, **kw)


def gradient_round_rect(c: tk.Canvas, x1, y1, x2, y2, r, c1: str, c2: str, tags=()):
    """Horizontal gradient clipped to a rounded rectangle (also draws circles)."""
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    span = max(1, x2 - x1)
    for x in range(int(x1), int(math.ceil(x2))):
        cx = x + 0.5
        dx = (x1 + r - cx) if cx < x1 + r else (cx - (x2 - r)) if cx > x2 - r else 0
        if dx > r:
            continue
        inset = r - math.sqrt(max(0.0, r * r - dx * dx)) if dx > 0 else 0
        c.create_line(x, y1 + inset, x, y2 - inset + 0.5, fill=mix(c1, c2, (x - x1) / span), tags=tags)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def draw_cover(c: tk.Canvas, x0, y0, size, hue: str, glyph: str = "♪"):
    """Procedural 'album art': dark gradient, radial glow and a neon star."""
    top, bottom = mix(hue, "#000000", 0.78), "#07060b"
    for i in range(int(size)):
        c.create_line(x0, y0 + i, x0 + size, y0 + i, fill=mix(top, bottom, i / size))
    cx, cy = x0 + size / 2, y0 + size * 0.47
    base = mix(top, bottom, 0.5)
    radius = size * 0.5
    for step in range(26, 0, -1):
        t = step / 26
        r = radius * t
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="",
                      fill=mix(mix(hue, "#ffffff", 0.1), base, 0.35 + 0.65 * t ** 0.7))
    outer, inner = size * 0.36, size * 0.36 * 0.46
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = outer if k % 2 == 0 else inner
        pts += [cx + rr * math.cos(ang), cy + rr * math.sin(ang)]
    for k in range(6, 0, -1):
        c.create_polygon(pts, fill="", outline=mix(hue, base, k / 7), width=k * 2.4, joinstyle="round")
    c.create_polygon(pts, fill=mix(hue, "#000000", 0.84), outline=mix(hue, "#ffffff", 0.6),
                     width=2, joinstyle="round")
    c.create_text(cx, cy + size * 0.02, text=glyph, fill=mix(hue, "#ffffff", 0.75),
                  font=(FAMILY, max(10, int(size * 0.16)), "bold"))
    # floor glow
    for step in range(10, 0, -1):
        t = step / 10
        c.create_oval(cx - size * 0.32 * t, y0 + size * 0.86 - 5 * t, cx + size * 0.32 * t,
                      y0 + size * 0.86 + 5 * t, outline="", fill=mix(hue, bottom, 0.25 + 0.75 * t))


def draw_phrase_art(c: tk.Canvas, w, h, hue: str, pitches: list[int], letter: str):
    """Thumbnail for one human → AI exchange: gradient plus the AI pitch contour."""
    dark = mix(hue, "#000000", 0.82)
    mid = mix(hue, "#000000", 0.45)
    for x in range(int(w)):
        c.create_line(x, 0, x, h, fill=mix(dark, mid, (x / max(1, w)) ** 1.6))
    c.create_text(w - 14, h - 8, text=letter, anchor="se", fill=mix(hue, "#ffffff", 0.35),
                  font=(FAMILY, int(h * 0.42), "bold"))
    if not pitches:
        return
    lo, hi = min(pitches) - 2, max(pitches) + 2
    n = len(pitches)
    pts = []
    for i, p in enumerate(pitches):
        x = 14 + (w - 28) * (i / max(1, n - 1) if n > 1 else 0.5)
        y = h - 14 - (h - 28) * (p - lo) / max(1, hi - lo)
        pts += [x, y]
    if len(pts) >= 4:
        c.create_line(*pts, fill=mix(hue, dark, 0.35), width=7, smooth=True, capstyle="round")
        c.create_line(*pts, fill=mix(hue, "#ffffff", 0.55), width=2, smooth=True, capstyle="round")
    for i in range(0, len(pts), 2):
        c.create_oval(pts[i] - 3, pts[i + 1] - 3, pts[i] + 3, pts[i + 1] + 3, fill="#ffffff", outline="")


# --------------------------------------------------------------------------- canvas controls
class _CanvasControl(tk.Canvas):
    """A canvas-drawn control that honours ``configure(state=...)`` like tk.Button.

    ``gui.MusicApp._set_busy`` disables controls through ``configure(state=...)``,
    so these widgets can be placed in ``_busy_controls``/``_note_buttons``.
    """

    def __init__(self, parent, width, height, command=None, bg=None):
        self._command = command
        self._state = "normal"
        self._hover = False
        self._pressed = False
        if bg is None:
            try:
                bg = parent.cget("bg")
            except tk.TclError:
                bg = BG
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2" if command else "")
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Configure>", lambda _e: self.redraw())

    # tk.Button-compatible state handling
    def configure(self, cnf=None, **kw):
        if isinstance(cnf, dict):
            kw = {**cnf, **kw}
            cnf = None
        if "state" in kw:
            self._state = "disabled" if str(kw.pop("state")) == "disabled" else "normal"
            super().configure(cursor="hand2" if (self._command and self._state == "normal") else "")
            self.redraw()
            if not kw:
                return None
        return super().configure(cnf, **kw)

    config = configure

    def cget(self, key):
        return self._state if key == "state" else super().cget(key)

    def invoke(self):
        if self._state == "normal" and self._command:
            self._command()

    def _set_hover(self, value):
        self._hover = value
        if not value:
            self._pressed = False
        self.redraw()

    def _press(self, _event):
        if self._state == "normal":
            self._pressed = True
            self.redraw()

    def _release(self, event):
        was = self._pressed
        self._pressed = False
        self.redraw()
        if was and 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self.invoke()

    def size(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1:
            w = int(float(super().cget("width")))
        if h <= 1:
            h = int(float(super().cget("height")))
        return w, h

    def redraw(self):
        self.delete("all")
        self.draw(*self.size())

    def draw(self, w, h):  # pragma: no cover - abstract
        raise NotImplementedError


class PillButton(_CanvasControl):
    """Rounded button. Variants: ghost, primary (gradient), chip (selectable), tag."""

    def __init__(self, parent, text, command=None, variant="ghost", font=None, icon=None,
                 height=34, width=None, radius=None, padx=14, badge=None):
        self._text, self._icon, self._badge = text, icon, badge
        self._variant, self._radius, self._selected = variant, radius, False
        self._font = font or (FAMILY, 10)
        self._badge_font = (FAMILY, 9, "bold")
        measure = tkfont.Font(root=parent, font=self._font).measure(self._label())
        if badge:
            measure += tkfont.Font(root=parent, font=self._badge_font).measure(badge) + 30
        super().__init__(parent, width or measure + 2 * padx, height, command)

    def _label(self):
        return f"{self._icon}  {self._text}" if self._icon else self._text

    def set_text(self, text, badge=None):
        self._text = text
        if badge is not None:
            self._badge = badge
        self.redraw()

    def set_selected(self, selected: bool):
        self._selected = selected
        self.redraw()

    def draw(self, w, h):
        r = self._radius if self._radius is not None else h / 2
        off = self._state == "disabled"
        v = self._variant
        fg = TEXT
        if v in ("primary", "tag"):
            c1, c2 = (PURPLE, PINK)
            if self._hover and not off:
                c1, c2 = mix(c1, "#ffffff", 0.12), mix(c2, "#ffffff", 0.12)
            if self._pressed:
                c1, c2 = mix(c1, "#000000", 0.15), mix(c2, "#000000", 0.15)
            if off:
                c1, c2, fg = mix(c1, CARD, 0.6), mix(c2, CARD, 0.6), mix(TEXT, CARD, 0.4)
            gradient_round_rect(self, 0, 0, w, h, r, c1, c2)
        elif v == "chip":
            if self._selected:
                fill, outline, fg = "#241a3d", VIOLET, "#ead9ff"
            else:
                fill = CARD_HI if self._hover and not off else CARD
                outline, fg = BORDER, (TEXT if self._hover and not off else MUTED)
            if off:
                fg = FAINT
            round_rect(self, 1, 1, w - 1, h - 1, r, fill=fill, outline=outline)
        else:
            fill = "#252531" if self._hover and not off else "#1b1b24"
            if self._pressed:
                fill = "#2d2d3b"
            round_rect(self, 1, 1, w - 1, h - 1, r, fill=fill, outline="#2c2c38")
            fg = FAINT if off else TEXT
        cx = w / 2
        if self._badge:
            bw = tkfont.Font(root=self, font=self._badge_font).measure(self._badge) + 18
            bx2, by1, by2 = w - 8, h / 2 - 11, h / 2 + 11
            round_rect(self, bx2 - bw, by1, bx2, by2, 11, fill=mix(PINK, "#000000", 0.3) if not off else CARD,
                       outline=mix(PINK, "#ffffff", 0.25) if not off else BORDER)
            self.create_text(bx2 - bw / 2, h / 2, text=self._badge, fill=fg, font=self._badge_font)
            cx = (w - bw - 8) / 2
        self.create_text(cx, h / 2, text=self._label(), fill=fg, font=self._font)


class CircleButton(_CanvasControl):
    """Round transport button drawing play/pause/stop icons as vector shapes."""

    def __init__(self, parent, icon, command=None, size=40, kind="ghost"):
        self._icon, self._kind = icon, kind
        super().__init__(parent, size, size, command)

    def set_icon(self, icon):
        if icon != self._icon:
            self._icon = icon
            self.redraw()

    def draw(self, w, h):
        off = self._state == "disabled"
        s = min(w, h)
        if self._kind == "gradient":
            c1, c2 = (mix(PURPLE, "#ffffff", 0.12), mix(PINK, "#ffffff", 0.12)) if self._hover else (PURPLE, PINK)
            if off:
                c1, c2 = mix(c1, CARD, 0.6), mix(c2, CARD, 0.6)
            gradient_round_rect(self, 0, 0, s, s, s / 2, c1, c2)
            fg = "#ffffff"
        elif self._kind == "white":
            fill = "#e9e6f2" if self._hover else "#ffffff"
            self.create_oval(1, 1, s - 1, s - 1, fill=mix(fill, CARD, 0.5) if off else fill, outline="")
            fg = "#0b0b11"
        else:
            fill = "#252531" if self._hover and not off else "#1b1b24"
            self.create_oval(1, 1, s - 1, s - 1, fill=fill, outline="#2c2c38")
            fg = FAINT if off else TEXT
        cx, cy, u = s / 2, s / 2, s * 0.16
        if self._icon == "play":
            self.create_polygon(cx - u * 0.75, cy - u * 1.1, cx - u * 0.75, cy + u * 1.1,
                                cx + u * 1.15, cy, fill=fg, outline=fg, joinstyle="round")
        elif self._icon == "pause":
            self.create_rectangle(cx - u * 0.95, cy - u, cx - u * 0.3, cy + u, fill=fg, outline="")
            self.create_rectangle(cx + u * 0.3, cy - u, cx + u * 0.95, cy + u, fill=fg, outline="")
        elif self._icon == "stop":
            self.create_rectangle(cx - u * 0.85, cy - u * 0.85, cx + u * 0.85, cy + u * 0.85, fill=fg, outline="")
        else:
            self.create_text(cx, cy, text=self._icon, fill=fg, font=(FAMILY, int(s * 0.32)))


class NotePad(_CanvasControl):
    def __init__(self, parent, note, command, width=64, height=58):
        self.note, self._selected = note, False
        super().__init__(parent, width, height, command)

    def set_selected(self, selected):
        self._selected = selected
        self.redraw()

    def draw(self, w, h):
        off = self._state == "disabled"
        hue = NOTE_HUES.get(self.note, VIOLET)
        if self._selected:
            fill, outline, width = mix(hue, CARD, 0.82), hue, 2
        elif self._hover and not off:
            fill, outline, width = CARD_HI, mix(hue, BORDER, 0.4), 1
        else:
            fill, outline, width = CARD, BORDER, 1
        if self._pressed:
            fill = mix(hue, CARD, 0.65)
        round_rect(self, 2, 2, w - 2, h - 2, 12, fill=fill, outline=outline, width=width)
        fg = FAINT if off else TEXT
        self.create_text(w / 2, h / 2 - 7, text=self.note, fill=fg, font=(FAMILY, 17, "bold"))
        self.create_text(w / 2, h / 2 + 15, text=SOLFEGE.get(self.note, "random"),
                         fill=FAINT if off else (mix(hue, "#ffffff", 0.4) if self._selected else MUTED),
                         font=(FAMILY, 8))


class ModelTile(_CanvasControl):
    def __init__(self, parent, icon, title, subtitle, command, width=140, height=66):
        self._icon, self._title, self._subtitle, self._selected = icon, title, subtitle, False
        super().__init__(parent, width, height, command)

    def set_selected(self, selected):
        self._selected = selected
        self.redraw()

    def draw(self, w, h):
        off = self._state == "disabled"
        if self._selected:
            fill, outline, width = "#1d1630", VIOLET, 2
        else:
            fill, outline, width = (CARD_HI if self._hover and not off else CARD), BORDER, 1
        round_rect(self, 2, 2, w - 2, h - 2, 12, fill=fill, outline=outline, width=width)
        fg = FAINT if off else TEXT
        self.create_text(w / 2, 19, text=self._icon, fill=VIOLET if self._selected else MUTED, font=(FAMILY, 12))
        self.create_text(w / 2, 38, text=self._title, fill=fg, font=(FAMILY, 9, "bold"))
        self.create_text(w / 2, 53, text=self._subtitle, fill=FAINT if off else MUTED, font=(FAMILY, 8))


class NavItem(_CanvasControl):
    def __init__(self, parent, icon, text, command=None, active=False, width=200, height=36):
        self._icon, self._text, self._active = icon, text, active
        super().__init__(parent, width, height, command)

    def draw(self, w, h):
        if self._active:
            round_rect(self, 0, 0, w, h, 10, fill=CARD_HI, outline="")
        elif self._hover:
            round_rect(self, 0, 0, w, h, 10, fill="#16161f", outline="")
        fg = TEXT if self._active or self._hover else "#c9c6d6"
        self.create_text(16, h / 2, text=self._icon, anchor="w", fill=VIOLET if self._active else MUTED,
                         font=(FAMILY, 11))
        self.create_text(42, h / 2, text=self._text, anchor="w", fill=fg, font=(FAMILY, 10))


class GradientSlider(_CanvasControl):
    """Slider bound to a Tk variable with a purple→pink filled track."""

    def __init__(self, parent, variable, from_, to, resolution, width=280, height=26):
        self.variable, self.from_, self.to, self.resolution = variable, from_, to, resolution
        super().__init__(parent, width, height, command=None)
        self.configure(cursor="hand2")
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonPress-1>", self._drag, add="+")
        variable.trace_add("write", lambda *_: self.redraw())

    def _release(self, _event):
        self._pressed = False
        self.redraw()

    def _drag(self, event):
        if self._state != "normal":
            return
        w, _ = self.size()
        frac = max(0.0, min(1.0, (event.x - 9) / max(1, w - 18)))
        value = self.from_ + frac * (self.to - self.from_)
        value = round(round(value / self.resolution) * self.resolution, 4)
        if isinstance(self.variable, tk.IntVar):
            value = int(round(value))
        if value != self.variable.get():
            self.variable.set(value)

    def draw(self, w, h):
        off = self._state == "disabled"
        try:
            value = float(self.variable.get())
        except (tk.TclError, ValueError):
            value = self.from_
        frac = (value - self.from_) / max(1e-9, self.to - self.from_)
        x1, x2, cy = 9, w - 9, h / 2
        kx = x1 + frac * (x2 - x1)
        round_rect(self, x1, cy - 2, x2, cy + 2, 2, fill="#272733", outline="")
        if kx - x1 > 2:
            c1, c2 = (PURPLE, PINK) if not off else (mix(PURPLE, CARD, 0.6), mix(PINK, CARD, 0.6))
            gradient_round_rect(self, x1, cy - 2, kx, cy + 2, 2, c1, mix(c1, c2, frac))
        r = 8 if (self._hover or self._pressed) and not off else 7
        self.create_oval(kx - r, cy - r, kx + r, cy + r, fill="#ffffff" if not off else MUTED,
                         outline=VIOLET if not off else FAINT, width=2)


# --------------------------------------------------------------------------- playback tracking
class TrackedMidiPlayer(MidiPlayer):
    """``MidiPlayer`` that additionally remembers what is playing for the progress UI."""

    def __init__(self, on_status=None):
        super().__init__(on_status)
        self.origin = 0.0
        self.length = 0.0
        self._t0 = None
        self._paused_at = None
        self._paused_total = 0.0

    def play(self, events):
        events = list(events)
        super().play(events)
        if events and self.playback_mode != "unavailable":
            self.origin = min(e.start for e in events)
            self.length = max(e.start + e.duration for e in events) - self.origin
            self._t0, self._paused_at, self._paused_total = time.monotonic(), None, 0.0

    def pause(self):
        super().pause()
        if self._t0 is not None and self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self):
        if self._paused_at is not None:
            self._paused_total += time.monotonic() - self._paused_at
            self._paused_at = None
        super().resume()

    def stop(self):
        super().stop()
        self._t0 = self._paused_at = None

    @property
    def is_playing(self) -> bool:
        return self.position() is not None and self._paused_at is None

    def position(self):
        """Absolute composition time of the playhead, or None when idle."""
        if self._t0 is None:
            return None
        now = self._paused_at if self._paused_at is not None else time.monotonic()
        elapsed = now - self._t0 - self._paused_total
        if elapsed > self.length + 0.05:
            self._t0 = None
            return None
        return self.origin + max(0.0, elapsed)


# --------------------------------------------------------------------------- application
class ModernMusicApp(MusicApp):
    """Same co-creative workflow as ``gui.MusicApp`` with a modern studio layout."""

    def __init__(self, adapter_hint=None) -> None:
        self._selected_note = "C"
        self._last_playhead = None
        self._history_cards: list[tk.Widget] = []
        super().__init__(adapter_hint)
        self.root.title("AI Music Co-Creator • Studio")
        self.root.geometry("1400x900")
        self.root.minsize(1180, 760)
        # Swap in a player that also reports its position (same MIDI/tone back-end).
        self.player.close()
        self.player = TrackedMidiPlayer(self._set_status_threadsafe)
        self.playback_label.config(text=f"Output · {self.player.playback_mode}")
        self._refresh()
        self.root.after(60, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        global FAMILY
        families = set(tkfont.families(self.root))
        FAMILY = next((f for f in ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans") if f in families),
                      "TkDefaultFont")
        self.root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=CARD, background=CARD_HI, foreground=TEXT,
                        arrowcolor=TEXT, bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", TEXT)])
        style.configure("Roll.Horizontal.TScrollbar", troughcolor=CARD, background="#2f2f3e",
                        bordercolor=CARD, lightcolor="#2f2f3e", darkcolor="#2f2f3e", arrowcolor=MUTED,
                        gripcount=0)
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", PURPLE)

        self.player_bar = tk.Frame(self.root, bg=SIDEBAR, height=86)
        self.player_bar.pack(side="bottom", fill="x")
        self.player_bar.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="bottom", fill="x")
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=SIDEBAR, width=232)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")
        create = tk.Frame(body, bg=PANEL, width=340)
        create.pack(side="left", fill="y")
        create.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")
        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_create(create)
        self._build_main(main)
        self._build_player_bar(self.player_bar)

        for var in (self.duration, self.bpm, self.temperature, self.model_choice):
            var.trace_add("write", lambda *_: self._sync_controls())
        self._sync_controls()

    @staticmethod
    def _section(parent, text, right=None, bg=PANEL, pady=(16, 8)):
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=pady)
        tk.Label(row, text=text, bg=bg, fg=MUTED, font=(FAMILY, 8, "bold")).pack(side="left")
        if right:
            tk.Label(row, text=right, bg=bg, fg=FAINT, font=(FAMILY, 8)).pack(side="right")
        return row

    def _button(self, parent, label, command):  # used by LoRAControls as well
        return PillButton(parent, label, command, variant="ghost", font=(FAMILY, 9), height=30, padx=12)

    def _build_sidebar(self, side):
        inner = tk.Frame(side, bg=SIDEBAR)
        inner.pack(fill="both", expand=True, padx=16, pady=18)
        brand = tk.Frame(inner, bg=SIDEBAR)
        brand.pack(fill="x", pady=(0, 18))
        logo = tk.Canvas(brand, width=36, height=36, bg=SIDEBAR, highlightthickness=0)
        logo.pack(side="left")
        gradient_round_rect(logo, 0, 0, 36, 36, 10, PURPLE, PINK)
        logo.create_text(18, 18, text="♫", fill="#ffffff", font=(FAMILY, 15, "bold"))
        names = tk.Frame(brand, bg=SIDEBAR)
        names.pack(side="left", padx=10)
        tk.Label(names, text="Co-Creator", bg=SIDEBAR, fg=TEXT, font=(FAMILY, 13, "bold")).pack(anchor="w")
        tk.Label(names, text="Human × AI music studio", bg=SIDEBAR, fg=MUTED, font=(FAMILY, 8)).pack(anchor="w")

        for icon, text, command, active in (("⌂", "Studio", None, True),
                                            ("▤", "Session logs", self._open_sessions_folder, False),
                                            ("⤓", "Export MIDI", self._export_midi, False),
                                            ("≡", "Export interaction log", self._export_log, False)):
            NavItem(inner, icon, text, command, active=active).pack(fill="x", pady=1)

        self._section(inner, "AI ENGINE", bg=SIDEBAR, pady=(22, 6))
        lora_row = tk.Frame(inner, bg=SIDEBAR)
        lora_row.pack(fill="x")
        self._build_lora_controls(lora_row)
        self.lora.toggle.configure(bg=SIDEBAR, fg=TEXT, activebackground=SIDEBAR, activeforeground=TEXT,
                                   selectcolor=CARD, highlightthickness=0, font=(FAMILY, 9))
        tk.Label(inner, text="LoRA always starts OFF. Enable it, pick a compatible adapter, then Apply.",
                 bg=SIDEBAR, fg=FAINT, font=(FAMILY, 8), justify="left", wraplength=196,
                 anchor="w").pack(anchor="w", fill="x", pady=(4, 0))

        self._section(inner, "SHORTCUTS", bg=SIDEBAR, pady=(20, 6))
        for keys, what in (("C D E F G A B", "choose a note"), ("H", "also plays B")):
            row = tk.Frame(inner, bg=SIDEBAR)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=keys, bg=SIDEBAR, fg="#d8ccff", font=(FAMILY, 9, "bold")).pack(side="left")
            tk.Label(row, text=what, bg=SIDEBAR, fg=MUTED, font=(FAMILY, 8)).pack(side="right")

        card = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.pack(side="bottom", fill="x")
        accent = tk.Canvas(card, height=3, bg=CARD, highlightthickness=0)
        accent.pack(fill="x")
        accent.bind("<Configure>", lambda e: (accent.delete("all"),
                                              gradient_round_rect(accent, 0, 0, e.width, 3, 0, PURPLE, PINK)))
        pad = tk.Frame(card, bg=CARD)
        pad.pack(fill="x", padx=12, pady=(8, 12))
        tk.Label(pad, text="Model status", bg=CARD, fg=TEXT, font=(FAMILY, 10, "bold")).pack(anchor="w")
        self.model_label = tk.Label(pad, text="", bg=CARD, fg=MUTED, font=(FAMILY, 8), justify="left",
                                    wraplength=176, anchor="w")
        self.model_label.pack(anchor="w", fill="x", pady=(4, 0))
        self._refresh_model_label()

    def _build_create(self, panel):
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=20, pady=18)
        head = tk.Frame(inner, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text="Create", bg=PANEL, fg=TEXT, font=(FAMILY, 16, "bold")).pack(side="left")
        CircleButton(head, "?", self._show_help, size=26).pack(side="right")
        sub = tk.Frame(inner, bg=PANEL)
        sub.pack(fill="x", pady=(2, 10))
        tk.Label(sub, text="Guide the next musical phrase", bg=PANEL, fg=MUTED, font=(FAMILY, 9)).pack(side="left")
        surprise = tk.Label(sub, text="✦ Surprise me", bg=PANEL, fg=VIOLET, font=(FAMILY, 9), cursor="hand2")
        surprise.pack(side="right")
        surprise.bind("<Button-1>", lambda _e: self._surprise())

        prompt = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        prompt.pack(fill="x")
        self.prompt_text = tk.Label(prompt, text="", bg=CARD, fg="#bdb9cc", font=(FAMILY, 10), justify="left",
                                    anchor="nw", wraplength=272, height=3)
        self.prompt_text.pack(fill="x", padx=12, pady=(10, 4))
        meta = tk.Frame(prompt, bg=CARD)
        meta.pack(fill="x", padx=12, pady=(0, 8))
        self.prompt_meta = tk.Label(meta, text="", bg=CARD, fg=FAINT, font=(FAMILY, 8))
        self.prompt_meta.pack(side="left")
        tk.Label(meta, text="⌨  C–B", bg=CARD, fg=FAINT, font=(FAMILY, 8)).pack(side="right")

        self._section(inner, "YOUR NOTE", right="human cue")
        pads = tk.Frame(inner, bg=PANEL)
        pads.pack(fill="x")
        self._pads: dict[str, NotePad] = {}
        for index, note in enumerate(NOTES):
            pad = NotePad(pads, note, lambda n=note: self._choose_note(n))
            pad.grid(row=index // 4, column=index % 4, padx=3, pady=3, sticky="nsew")
            self._pads[note] = pad
            self._note_buttons.append(pad)
        rnd = NotePad(pads, "?", self._surprise)
        rnd.grid(row=1, column=3, padx=3, pady=3, sticky="nsew")
        self._note_buttons.append(rnd)
        for col in range(4):
            pads.grid_columnconfigure(col, weight=1)

        self._section(inner, "MODEL", right="AI engine")
        tiles = tk.Frame(inner, bg=PANEL)
        tiles.pack(fill="x")
        self._model_tiles: dict[str, ModelTile] = {}
        for col, (icon, title, sub_text, value) in enumerate(MODELS):
            tile = ModelTile(tiles, icon, title, sub_text, lambda v=value: self._pick_model(v))
            tile.grid(row=0, column=col, padx=3, sticky="nsew")
            tiles.grid_columnconfigure(col, weight=1)
            self._model_tiles[value] = tile
            self._busy_controls.append(tile)

        self._section(inner, "CONTINUATION", right="AI autonomy")
        chips = tk.Frame(inner, bg=PANEL)
        chips.pack(fill="x")
        self._duration_chips: dict[str, PillButton] = {}
        for label, value in DURATIONS:
            chip = PillButton(chips, label, lambda v=value: self.duration.set(v), variant="chip",
                              font=(FAMILY, 9), height=28, padx=16)
            chip.pack(side="left", padx=(0, 6))
            self._duration_chips[value] = chip
            self._busy_controls.append(chip)

        self.bpm_value = self._slider_block(inner, "Tempo", self.bpm, 60, 160, 1, ("Slow", "Fast"))
        self.temp_value = self._slider_block(inner, "Temperature", self.temperature, 0.2, 1.5, 0.05,
                                             ("Focused", "Adventurous"))

        self.generate_button = PillButton(inner, "Generate continuation", self._generate_selected,
                                          variant="primary", font=(FAMILY, 11, "bold"), height=46,
                                          radius=12, badge="♪ C")
        self.generate_button.pack(fill="x", pady=(18, 6))
        self._busy_controls.append(self.generate_button)
        tk.Label(inner, textvariable=self.status, bg=PANEL, fg=MUTED, font=(FAMILY, 8),
                 wraplength=290, justify="center").pack(fill="x")

    def _slider_block(self, parent, title, variable, lo, hi, step, ends):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(14, 0))
        tk.Label(row, text=title, bg=PANEL, fg="#d6d3e2", font=(FAMILY, 9)).pack(side="left")
        value = tk.Label(row, text="", bg=PANEL, fg=TEXT, font=(FAMILY, 9, "bold"))
        value.pack(side="right")
        slider = GradientSlider(parent, variable, lo, hi, step)
        slider.pack(fill="x", pady=(2, 0))
        self._busy_controls.append(slider)
        foot = tk.Frame(parent, bg=PANEL)
        foot.pack(fill="x")
        tk.Label(foot, text=ends[0], bg=PANEL, fg=FAINT, font=(FAMILY, 7)).pack(side="left")
        tk.Label(foot, text=ends[1], bg=PANEL, fg=FAINT, font=(FAMILY, 7)).pack(side="right")
        return value

    def _build_main(self, main):
        inner = tk.Frame(main, bg=BG)
        inner.pack(fill="both", expand=True, padx=22, pady=18)

        head = tk.Frame(inner, bg=BG)
        head.pack(fill="x")
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="Your Creation", bg=BG, fg=TEXT, font=(FAMILY, 16, "bold")).pack(anchor="w")
        tk.Label(titles, text="HUMAN  →  AI  →  HUMAN  →  AI   ·   one human decision per AI phrase",
                 bg=BG, fg=MUTED, font=(FAMILY, 8)).pack(anchor="w")
        self.engine_chip = tk.Label(head, text="", bg=CARD, fg=TEXT, font=(FAMILY, 9), padx=12, pady=6,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.engine_chip.pack(side="right")
        self._refresh_model_label()

        # featured "current session" card
        featured = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        featured.pack(fill="x", pady=(14, 0))
        self.cover = tk.Canvas(featured, width=170, height=170, bg=CARD, highlightthickness=0)
        self.cover.pack(side="left", padx=16, pady=16)
        info = tk.Frame(featured, bg=CARD)
        info.pack(side="left", fill="both", expand=True, padx=(4, 18), pady=14)
        top = tk.Frame(info, bg=CARD)
        top.pack(fill="x")
        PillButton(top, "CURRENT SESSION", None, variant="tag", font=(FAMILY, 7, "bold"), height=20,
                   padx=10).pack(side="left")
        self.updated_label = tk.Label(top, text="", bg=CARD, fg=FAINT, font=(FAMILY, 8))
        self.updated_label.pack(side="right")
        self.title_label = tk.Label(info, text="Untitled co-creation", bg=CARD, fg=TEXT, font=(FAMILY, 18, "bold"),
                                    anchor="w")
        self.title_label.pack(fill="x", pady=(6, 0))
        self.meta_label = tk.Label(info, text="", bg=CARD, fg="#c9c6d6", font=(FAMILY, 9), anchor="w")
        self.meta_label.pack(fill="x")
        self.sequence = tk.Label(info, text="No notes yet", bg=CARD, fg=MUTED, font=(FAMILY, 9), anchor="w",
                                 justify="left", wraplength=520)
        self.sequence.pack(fill="x", pady=(4, 2))
        self.wave = tk.Canvas(info, height=40, bg=CARD, highlightthickness=0)
        self.wave.pack(fill="x", pady=(2, 0))
        self.wave.bind("<Configure>", lambda _e: self._draw_wave())
        times = tk.Frame(info, bg=CARD)
        times.pack(fill="x")
        self.wave_pos = tk.Label(times, text="0:00", bg=CARD, fg=FAINT, font=(FAMILY, 7))
        self.wave_pos.pack(side="left")
        self.wave_len = tk.Label(times, text="0:00", bg=CARD, fg=FAINT, font=(FAMILY, 7))
        self.wave_len.pack(side="right")
        actions = tk.Frame(info, bg=CARD)
        actions.pack(fill="x", pady=(6, 0))
        self.feature_play = CircleButton(actions, "play", self._toggle_play, size=40, kind="gradient")
        self.feature_play.pack(side="left")
        stop = CircleButton(actions, "stop", self._stop, size=32)
        stop.pack(side="left", padx=(8, 14))
        buttons = [stop]
        for label, icon, cmd, variant in (("Regenerate", "↻", self._regenerate, "ghost"),
                                          ("Undo", "↶", self._undo, "ghost"),
                                          ("Export MIDI", "⤓", self._export_midi, "primary")):
            b = PillButton(actions, label, cmd, variant=variant, icon=icon, font=(FAMILY, 9, "bold" if
                           variant == "primary" else "normal"), height=34, radius=10, padx=14)
            b.pack(side="left", padx=(0, 8))
            buttons.append(b)
        self._busy_controls.extend([self.feature_play, *buttons[:-1]])

        # phrase history ("variations")
        bottom = tk.Frame(inner, bg=BG)
        bottom.pack(side="bottom", fill="x")
        hhead = tk.Frame(bottom, bg=BG)
        hhead.pack(fill="x", pady=(14, 6))
        tk.Label(hhead, text="Phrase history", bg=BG, fg=TEXT, font=(FAMILY, 11, "bold")).pack(side="left")
        tk.Label(hhead, text="  click a card to replay that human → AI exchange", bg=BG, fg=FAINT,
                 font=(FAMILY, 8)).pack(side="left", pady=(2, 0))
        self.history_row = tk.Frame(bottom, bg=BG, height=128)
        self.history_row.pack(fill="x")
        self.history_row.pack_propagate(False)
        for col in range(4):
            self.history_row.grid_columnconfigure(col, weight=1, uniform="history")
        self.history_row.grid_rowconfigure(0, weight=1)

        # quick actions ("suggested ideas")
        quick = tk.Frame(bottom, bg=BG)
        quick.pack(fill="x", pady=(12, 0))
        tk.Label(quick, text="Quick actions", bg=BG, fg=TEXT, font=(FAMILY, 10, "bold")).pack(side="left",
                                                                                            padx=(0, 12))
        for label, icon, cmd in (("Export interaction log", "≡", self._export_log),
                                 ("Open session logs", "▤", self._open_sessions_folder),
                                 ("Clear session", "✕", self._clear),
                                 ("Shortcuts & help", "?", self._show_help)):
            b = PillButton(quick, label, cmd, variant="chip", icon=icon, font=(FAMILY, 9), height=30,
                           radius=10, padx=12)
            b.pack(side="left", padx=(0, 8))
            if label == "Clear session":
                self._busy_controls.append(b)

        # piano roll
        roll = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        roll.pack(fill="both", expand=True, pady=(14, 0))
        rhead = tk.Frame(roll, bg=CARD)
        rhead.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(rhead, text="Piano roll", bg=CARD, fg=TEXT, font=(FAMILY, 11, "bold")).pack(side="left")
        tk.Label(rhead, text="  human cues and AI continuations on one timeline", bg=CARD, fg=FAINT,
                 font=(FAMILY, 8)).pack(side="left", pady=(2, 0))
        for colour, text in ((AI, "AI continuation"), (HUMAN, "Human input")):
            tk.Label(rhead, text=text, bg=CARD, fg=MUTED, font=(FAMILY, 8)).pack(side="right", padx=(2, 10))
            tk.Label(rhead, text="●", bg=CARD, fg=colour, font=(FAMILY, 9)).pack(side="right")
        area = tk.Frame(roll, bg=CARD)
        area.pack(fill="both", expand=True, padx=(10, 10))
        self.keys_canvas = tk.Canvas(area, width=44, bg=CARD, highlightthickness=0)
        self.keys_canvas.pack(side="left", fill="y")
        self.canvas = tk.Canvas(area, bg="#13131a", highlightthickness=0, height=120)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw_timeline())
        self.timeline_scroll = ttk.Scrollbar(roll, orient="horizontal", command=self.canvas.xview,
                                             style="Roll.Horizontal.TScrollbar")
        self.timeline_scroll.pack(fill="x", padx=(54, 10), pady=(4, 8))
        self.canvas.configure(xscrollcommand=self.timeline_scroll.set)

    def _build_player_bar(self, bar):
        left = tk.Frame(bar, bg=SIDEBAR, width=233)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text="●  Offline · local model", bg=SIDEBAR, fg=MUTED,
                 font=(FAMILY, 8)).pack(anchor="w", padx=18, pady=(22, 0))
        self.playback_label = tk.Label(left, text="", bg=SIDEBAR, fg=FAINT, font=(FAMILY, 8))
        self.playback_label.pack(anchor="w", padx=18)

        track = tk.Frame(bar, bg=SIDEBAR, width=300)
        track.pack(side="left", fill="y", padx=(16, 0))
        track.pack_propagate(False)
        self.thumb = tk.Canvas(track, width=52, height=52, bg=SIDEBAR, highlightthickness=0)
        self.thumb.pack(side="left", pady=17)
        names = tk.Frame(track, bg=SIDEBAR)
        names.pack(side="left", padx=12, fill="x", expand=True)
        self.track_title = tk.Label(names, text="Current composition", bg=SIDEBAR, fg=TEXT,
                                    font=(FAMILY, 10, "bold"), anchor="w")
        self.track_title.pack(anchor="w", pady=(24, 0), fill="x")
        tk.Label(names, textvariable=self.status, bg=SIDEBAR, fg=MUTED, font=(FAMILY, 8), anchor="w",
                 wraplength=210, justify="left").pack(anchor="w", fill="x")

        right = tk.Frame(bar, bg=SIDEBAR)
        right.pack(side="right", fill="y", padx=18)
        for label, cmd in (("MIDI", self._export_midi), ("LOG", self._export_log)):
            PillButton(right, label, cmd, icon="⤓", font=(FAMILY, 8), height=28, radius=8,
                       padx=10).pack(side="right", padx=(6, 0), pady=29)
        self.tempo_readout = tk.Label(right, text="", bg=SIDEBAR, fg="#c9c6d6", font=(FAMILY, 9))
        self.tempo_readout.pack(side="right", padx=(0, 10))

        center = tk.Frame(bar, bg=SIDEBAR)
        center.pack(side="left", fill="both", expand=True, padx=20)
        controls = tk.Frame(center, bg=SIDEBAR)
        controls.pack(pady=(10, 2))
        undo = CircleButton(controls, "↶", self._undo, size=30)
        stop = CircleButton(controls, "stop", self._stop, size=30)
        self.big_play = CircleButton(controls, "play", self._toggle_play, size=44, kind="white")
        replay = CircleButton(controls, "↻", self._play, size=30)
        clear = CircleButton(controls, "✕", self._clear, size=30)
        for widget in (undo, stop, self.big_play, replay, clear):
            widget.pack(side="left", padx=9)
        self._busy_controls.extend((undo, self.big_play, replay, clear))
        prog = tk.Frame(center, bg=SIDEBAR)
        prog.pack(fill="x")
        self.prog_pos = tk.Label(prog, text="0:00", bg=SIDEBAR, fg=FAINT, font=(FAMILY, 8), width=5)
        self.prog_pos.pack(side="left")
        self.prog_len = tk.Label(prog, text="0:00", bg=SIDEBAR, fg=FAINT, font=(FAMILY, 8), width=5)
        self.prog_len.pack(side="right")
        self.progress = tk.Canvas(prog, height=10, bg=SIDEBAR, highlightthickness=0)
        self.progress.pack(side="left", fill="x", expand=True, padx=6)
        self.progress.bind("<Configure>", lambda _e: self._draw_progress(self._progress_fraction()))

    # ------------------------------------------------------------------ behaviour (thin wrappers)
    def _choose_note(self, note: str) -> None:
        if note in PITCH and not self._generating:
            self._selected_note = note
            self._sync_controls()
        super()._choose_note(note)

    def _generate_selected(self) -> None:
        self._choose_note(self._selected_note)

    def _surprise(self) -> None:
        if not self._generating:
            self._choose_note(random.choice(NOTES))

    def _pick_model(self, value: str) -> None:
        if self._generating or value == self.model_choice.get():
            return
        self.model_choice.set(value)
        self._select_model()

    def _regenerate(self) -> None:
        """Undo the latest exchange and ask the AI again with the same human note."""
        if self._generating:
            return
        if not self.composition.interactions:
            self.status.set("Nothing to regenerate yet — choose a note first")
            return
        note = self.composition.interactions[-1].human_note
        self._undo()
        self._choose_note(note)

    def _toggle_play(self) -> None:
        player = self.player
        if isinstance(player, TrackedMidiPlayer) and player.position() is not None:
            self._pause()
        else:
            self._play()

    def _play_interaction(self, index: int) -> None:
        if self._generating or index >= len(self.composition.interactions):
            return
        self.player.play(self.composition.interactions[index].events)
        self.status.set(f"Replaying phrase {index + 1}")

    def _open_sessions_folder(self) -> None:
        folder = LOG_DIR / "sessions"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
            self.status.set(f"Session logs: {folder}")
        except (OSError, AttributeError) as exc:
            self.status.set(f"Session logs are saved in {folder} ({exc})")

    def _show_help(self) -> None:
        messagebox.showinfo(
            "How co-creation works",
            "1. Pick a note (click a pad or press C D E F G A B; H also plays B).\n"
            "2. The AI answers with a short continuation and plays it.\n"
            "3. Keep alternating: human → AI → human → AI.\n\n"
            "Generate repeats the selected note. Regenerate replaces the latest AI phrase.\n"
            "Model, continuation length, tempo and temperature apply to the next phrase.\n"
            "Every session is autosaved to logs/sessions as JSON.",
            parent=self.root)

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        if hasattr(self, "generate_button"):
            self.generate_button.set_text("AI is composing…" if busy else "Generate continuation")

    def _refresh_model_label(self, active_model=None, error=None) -> None:
        if not hasattr(self, "model_label"):
            return
        super()._refresh_model_label(active_model, error)
        if not hasattr(self, "engine_chip"):
            return
        gen = self.ml_generator
        loaded = gen.loaded if active_model is None else active_model
        if loaded:
            text = f"●  LSTM {gen.config.get('layers', 3)}×{gen.config.get('hidden_size', 512)}"
            text += f"  ·  {getattr(gen, 'device', 'cpu')}"
            colour = OK_GREEN
        else:
            text, colour = "●  Rule-based fallback", AMBER
        lora = getattr(self, "lora", None)
        if lora is not None and lora.enabled.get():
            text += "  ·  LoRA active" if lora.ready else "  ·  LoRA not ready"
        self.engine_chip.config(text=text, fg=colour)

    def _sync_controls(self) -> None:
        if not hasattr(self, "generate_button"):
            return
        for note, pad in self._pads.items():
            pad.set_selected(note == self._selected_note)
        for value, tile in self._model_tiles.items():
            tile.set_selected(value == self.model_choice.get())
        for value, chip in self._duration_chips.items():
            chip.set_selected(value == self.duration.get())
        try:
            bpm, temp = int(self.bpm.get()), float(self.temperature.get())
        except (tk.TclError, ValueError):
            return
        mood = "focused" if temp < 0.6 else "balanced" if temp <= 1.0 else "adventurous"
        self.bpm_value.config(text=f"{bpm} BPM")
        self.temp_value.config(text=f"{temp:.2f} · {mood}")
        self.tempo_readout.config(text=f"♩ {bpm} BPM")
        self.generate_button.set_text(self.generate_button._text, badge=f"♪ {self._selected_note}")
        length = self.duration.get().replace(" seconds", " s").replace(" second", " s")
        self.prompt_text.config(
            text=f"Human plays {self._selected_note} ({SOLFEGE[self._selected_note]}); the AI answers with a "
                 f"{length} continuation at {bpm} BPM, {mood} sampling.")

    # ------------------------------------------------------------------ rendering
    def _refresh(self) -> None:
        if not hasattr(self, "history_row"):
            return
        interactions = self.composition.interactions
        parts = [f"Human {i.human_note}  →  AI " + (" ".join(i.ai_notes) if i.ai_notes else "(no notes)")
                 for i in interactions]
        shown = parts[-3:]
        text = ("…   " if len(parts) > 3 else "") + "     |     ".join(shown)
        self.sequence.config(text=text if parts else
                             "No notes yet — choose a note on the left and the AI will continue your idea.")
        n = len(interactions)
        total = self.composition.end_time
        model = (interactions[-1].metadata.get("model", "") if interactions else "") or "—"
        self.title_label.config(text=f"Co-creation · {n} phrase{'s' if n != 1 else ''}" if n
                                else "Untitled co-creation")
        self.meta_label.config(text=f"⏱ {fmt_time(total)}     ♩ {self.bpm.get()} BPM     "
                                    f"♫ {len(self.composition.events)} notes     ◆ {model}")
        if interactions:
            try:
                stamp = datetime.fromisoformat(interactions[-1].timestamp).astimezone()
                self.updated_label.config(text=f"Last phrase {stamp:%H:%M:%S}")
            except ValueError:
                self.updated_label.config(text="")
        else:
            self.updated_label.config(text="Autosaves to logs/sessions")
        self.prompt_meta.config(text=f"{n} phrase{'s' if n != 1 else ''} · {len(self.composition.events)} notes")
        hue = NOTE_HUES.get(interactions[-1].human_note, PURPLE) if interactions else PURPLE
        self.cover.delete("all")
        draw_cover(self.cover, 0, 0, 170, hue)
        self.thumb.delete("all")
        draw_cover(self.thumb, 0, 0, 52, hue, glyph="")
        self.wave_len.config(text=fmt_time(total))
        self.prog_len.config(text=fmt_time(total))
        self._build_history()
        self._draw_timeline()

    def _build_history(self) -> None:
        for widget in self._history_cards:
            widget.destroy()
        self._history_cards.clear()
        interactions = self.composition.interactions
        if not interactions:
            empty = tk.Frame(self.history_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            empty.grid(row=0, column=0, columnspan=4, sticky="nsew")
            tk.Label(empty, text="Your human → AI exchanges will collect here as replayable cards.",
                     bg=CARD, fg=FAINT, font=(FAMILY, 9)).place(relx=0.5, rely=0.5, anchor="center")
            self._history_cards.append(empty)
            return
        start = max(0, len(interactions) - 4)
        for col, index in enumerate(range(len(interactions) - 1, start - 1, -1)):
            item = interactions[index]
            hue = NOTE_HUES.get(item.human_note, PURPLE)
            card = tk.Frame(self.history_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                            cursor="hand2")
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            art = tk.Canvas(card, height=72, bg=CARD, highlightthickness=0)
            art.pack(fill="x")
            pitches = [e.pitch for e in item.events if e.source == "ai"]
            art.bind("<Configure>", lambda e, c=art, h=hue, p=pitches, n=item.human_note:
                     (c.delete("all"), draw_phrase_art(c, e.width, e.height, h, p, n)))
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", padx=10, pady=(6, 0))
            tk.Label(row, text=f"Phrase {index + 1} · Human {item.human_note}", bg=CARD, fg=TEXT,
                     font=(FAMILY, 9, "bold"), anchor="w").pack(side="left")
            tk.Label(row, text="▶", bg=CARD, fg=VIOLET, font=(FAMILY, 9)).pack(side="right")
            notes = " → ".join(item.ai_notes) if item.ai_notes else "(no notes)"
            model = item.metadata.get("model", "")
            tk.Label(card, text=f"AI {notes}", bg=CARD, fg=MUTED, font=(FAMILY, 8), anchor="w").pack(
                fill="x", padx=10)
            tk.Label(card, text=f"{model} · {item.duration:g} s", bg=CARD, fg=FAINT, font=(FAMILY, 7),
                     anchor="w").pack(fill="x", padx=10, pady=(0, 6))
            for widget in (card, art, row, *row.winfo_children(), *card.winfo_children()):
                widget.bind("<Button-1>", lambda _e, i=index: self._play_interaction(i))
            self._history_cards.append(card)

    def _draw_timeline(self) -> None:
        if not hasattr(self, "keys_canvas"):
            return
        self._draw_roll()
        self._draw_wave()

    def _draw_roll(self) -> None:
        c, keys = self.canvas, self.keys_canvas
        c.delete("all")
        keys.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 160)
        ruler, lane = 20, 34
        grid_h = h - ruler - lane - 4
        events = self.composition.events
        pitches = [e.pitch for e in events]
        lo = min(min(pitches, default=60) - 2, 59)
        hi = max(max(pitches, default=72) + 2, 72)
        rows = hi - lo + 1
        row_h = grid_h / rows
        total = max(self.composition.end_time, 2.0)
        pps = max(90.0, (w - 24) / (total + 0.3))
        x0 = 10
        content_w = max(w, int(x0 + total * pps + 40))
        self._roll_geometry = (x0, pps, h)
        # rows
        for p in range(lo, hi + 1):
            y = ruler + (hi - p) * row_h
            black = p % 12 in BLACK_KEYS
            c.create_rectangle(0, y, content_w, y + row_h, outline="", fill="#111118" if black else "#16161e")
            if p % 12 == 0:
                c.create_line(0, y + row_h, content_w, y + row_h, fill="#2a2a38")
            keys.create_rectangle(0, y, 44, y + row_h, fill="#d9d6e3", outline="#9a97a8")
            if black:
                keys.create_rectangle(0, y + 1, 27, y + row_h - 1, fill="#1c1c26", outline="#0d0d12")
            if p % 12 == 0 and row_h >= 8:
                keys.create_text(4, y + row_h / 2, text=f"C{p // 12 - 1}", anchor="w", fill="#5b5870",
                                 font=(FAMILY, 7, "bold"))
        keys.create_rectangle(0, 0, 44, ruler, fill=CARD, outline="")
        keys.create_rectangle(0, h - lane - 4, 44, h, fill=CARD, outline="")
        keys.create_text(22, h - lane / 2 - 2, text="VEL", fill=FAINT, font=(FAMILY, 7, "bold"))
        # beat grid & ruler
        try:
            beat = 60.0 / max(1, int(self.bpm.get()))
        except (tk.TclError, ValueError):
            beat = 0.6
        c.create_rectangle(0, 0, content_w, ruler, fill=CARD, outline="")
        i = 0
        while x0 + i * beat * pps <= content_w:
            x = x0 + i * beat * pps
            bar = i % 4 == 0
            c.create_line(x, ruler, x, ruler + grid_h, fill="#2b2b3a" if bar else "#1d1d27")
            if bar:
                c.create_text(x + 3, ruler / 2, text=str(i // 4 + 1), anchor="w", fill=MUTED, font=(FAMILY, 7))
                c.create_line(x, ruler - 6, x, ruler, fill=FAINT)
            i += 1
        # velocity lane
        ly = h - lane
        c.create_rectangle(0, ly - 4, content_w, h, fill=CARD, outline="")
        c.create_line(0, ly - 4, content_w, ly - 4, fill=BORDER)
        if not events:
            c.create_text(w / 2, ruler + grid_h / 2, text="Pick a note to start — your co-created piano roll appears here",
                          fill="#6f6c82", font=(FAMILY, 10))
            c.configure(scrollregion=(0, 0, w, h))
            return
        # phrase separators
        for item in self.composition.interactions:
            human = next((e for e in item.events if e.source == "human"), None)
            if human:
                x = x0 + human.start * pps
                c.create_line(x, ruler, x, ruler + grid_h, fill="#3b2d5c", dash=(3, 3))
        for e in events:
            x = x0 + e.start * pps
            width = max(6.0, e.duration * pps - 2)
            y = ruler + (hi - e.pitch) * row_h
            colour = HUMAN if e.source == "human" else AI
            round_rect(c, x, y + 1, x + width, y + row_h - 1, min(4, row_h / 2), fill=colour,
                       outline=mix(colour, "#ffffff", 0.35))
            if width > 20 and row_h >= 11:
                c.create_text(x + 4, y + row_h / 2, text=e.label, anchor="w", fill="#0b0b11",
                              font=(FAMILY, 7, "bold"))
            vh = (lane - 10) * max(1, e.velocity) / 127
            c.create_line(x + 1, h - 4, x + 1, h - 4 - vh, fill=colour, width=3)
            c.create_oval(x - 1, h - 6 - vh, x + 3, h - 2 - vh, fill=colour, outline="")
        c.create_line(-10, 0, -10, h, fill="#ffffff", width=2, tags="playhead")
        c.configure(scrollregion=(0, 0, content_w, h))

    def _draw_wave(self, playhead: float | None = None) -> None:
        c = self.wave
        c.delete("all")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 40)
        events = self.composition.events
        total = max(self.composition.end_time, 0.001)
        bars = max(10, int(w // 6))
        for i in range(bars):
            t0, t1 = total * i / bars, total * (i + 1) / bars
            active = [e for e in events if e.start < t1 and e.start + e.duration > t0]
            if active:
                level = sum(0.35 + 0.65 * (e.pitch - 48) / 36 for e in active) / len(active)
                level = max(0.2, min(1.0, level)) * (0.72 + 0.28 * abs(math.sin(i * 1.7)))
            else:
                level = 0.08
            bh = max(3, level * (h - 4))
            x = i * w / bars + 1
            if playhead is not None and t0 <= playhead:
                colour = mix(PURPLE, PINK, i / bars)
            elif active and any(e.source == "human" for e in active):
                colour = "#3d5a73"
            else:
                colour = "#3a3a4a" if active else "#24242e"
            c.create_rectangle(x, h / 2 - bh / 2, x + max(2, w / bars - 2), h / 2 + bh / 2, fill=colour,
                               outline="")

    def _progress_fraction(self) -> float | None:
        player = self.player
        if not isinstance(player, TrackedMidiPlayer):
            return None
        pos = player.position()
        total = self.composition.end_time
        return None if pos is None or total <= 0 else min(1.0, pos / total)

    def _draw_progress(self, frac: float | None) -> None:
        c = self.progress
        c.delete("all")
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 10)
        cy = h / 2
        round_rect(c, 0, cy - 2, w, cy + 2, 2, fill="#272733", outline="")
        if frac is not None and frac > 0:
            gradient_round_rect(c, 0, cy - 2, max(4, w * frac), cy + 2, 2, PURPLE, mix(PURPLE, PINK, frac))
            x = w * frac
            c.create_oval(x - 5, cy - 5, x + 5, cy + 5, fill="#ffffff", outline="")

    def _tick(self) -> None:
        if self._closed:
            return
        player = self.player
        pos = player.position() if isinstance(player, TrackedMidiPlayer) else None
        playing = pos is not None and isinstance(player, TrackedMidiPlayer) and player.is_playing
        icon = "pause" if playing else "play"
        self.big_play.set_icon(icon)
        self.feature_play.set_icon(icon)
        if pos != self._last_playhead:
            self._last_playhead = pos
            total = self.composition.end_time
            self.prog_pos.config(text=fmt_time(pos or 0))
            self.wave_pos.config(text=fmt_time(pos or 0))
            self._draw_progress(None if pos is None or total <= 0 else min(1.0, pos / total))
            self._draw_wave(pos)
            geometry = getattr(self, "_roll_geometry", None)
            if geometry:
                x0, pps, h = geometry
                x = -10 if pos is None else x0 + pos * pps
                self.canvas.coords("playhead", x, 0, x, h)
                self.canvas.tag_raise("playhead")
        self.root.after(40, self._tick)


if __name__ == "__main__":
    ModernMusicApp().run()
