"""Offline Windows MIDI playback with a built-in tone fallback."""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

from composition import NoteEvent
from config import midi_to_frequency


class MidiPlayer:
    """Use the Windows MIDI mapper when available; otherwise play local tones."""
    def __init__(self, on_status: Callable[[str], None] | None = None) -> None:
        self.on_status = on_status or (lambda _status: None)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._midi = None
        self._handle = ctypes.c_void_p()
        self._open_windows_midi()
        try:
            import winsound
            self._beep = winsound.Beep
        except ImportError:
            self._beep = None

    def _open_windows_midi(self) -> None:
        if not hasattr(ctypes, "WinDLL"):
            return
        try:
            dll = ctypes.WinDLL("winmm", use_last_error=True)
            dll.midiOutGetNumDevs.restype = ctypes.c_uint
            dll.midiOutOpen.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint]
            dll.midiOutOpen.restype = ctypes.c_uint
            dll.midiOutShortMsg.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            dll.midiOutShortMsg.restype = ctypes.c_uint
            dll.midiOutReset.argtypes = [ctypes.c_void_p]
            dll.midiOutReset.restype = ctypes.c_uint
            dll.midiOutClose.argtypes = [ctypes.c_void_p]
            dll.midiOutClose.restype = ctypes.c_uint
            if dll.midiOutGetNumDevs() and dll.midiOutOpen(ctypes.byref(self._handle), 0, 0, 0, 0) == 0:
                self._midi = dll
        except (OSError, AttributeError, TypeError):
            self._midi = None

    @property
    def playback_mode(self) -> str:
        if self._midi:
            return "Windows MIDI"
        if self._beep:
            return "tone fallback"
        return "unavailable"

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    def play(self, events: list[NoteEvent]) -> None:
        previous = self._thread
        self.stop()
        if previous and previous.is_alive() and previous is not threading.current_thread():
            previous.join(timeout=0.15)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread = threading.Thread(target=self._run,
                                        args=(list(events),self._stop,self._pause),daemon=True)
        self._thread.start()

    def _send(self, pitch: int, velocity: int, note_on: bool) -> None:
        if not self._midi:
            return
        status = 0x90 if note_on else 0x80
        message = status | (max(0, min(127, pitch)) << 8) | (max(0, min(127, velocity)) << 16)
        self._midi.midiOutShortMsg(self._handle, message)

    def _run(self, events: list[NoteEvent], stop_event: threading.Event,
             pause_event: threading.Event) -> None:
        if self._midi:
            self.on_status("Playing AI continuation · Windows MIDI")
        elif self._beep:
            self.on_status("Playing AI continuation · tone fallback")
        else:
            self.on_status("Playback unavailable; composition remains saved")
            return
        if not events:
            self.on_status("Ready")
            return
        origin = min(e.start for e in events)
        schedule = []
        for event in events:
            on_time = max(0.0, event.start - origin)
            off_time = on_time + max(0.05, event.duration)
            schedule.append((on_time, 1, event))
            schedule.append((off_time, 0, event))
        schedule.sort(key=lambda row: (row[0], row[1]))
        clock = time.monotonic()
        try:
            for due, is_on, event in schedule:
                while not stop_event.is_set():
                    if pause_event.is_set():
                        time.sleep(0.03)
                        clock += 0.03
                        continue
                    remaining = due - (time.monotonic() - clock)
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.02))
                if stop_event.is_set():
                    break
                if self._midi:
                    self._send(event.pitch, event.velocity if is_on else 0, bool(is_on))
                elif is_on and self._beep:
                    try:
                        self._beep(midi_to_frequency(event.pitch), max(70, int(event.duration * 1000)))
                    except RuntimeError:
                        pass
        finally:
            if self._midi:
                self._midi.midiOutReset(self._handle)
            self.on_status("Ready")

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._midi:
            self._midi.midiOutReset(self._handle)

    def close(self) -> None:
        self.stop()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.2)
        if self._midi and self._handle:
            self._midi.midiOutClose(self._handle)
            self._handle = ctypes.c_void_p()
            self._midi = None
