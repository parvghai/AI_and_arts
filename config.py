"""Shared musical defaults and scale helpers."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
NOTES = ("C", "D", "E", "F", "G", "A", "B")
MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
DEFAULT_BPM = 100
DEFAULT_CONTINUATION = 1.0


def note_to_midi(name: str, octave: int = 4) -> int:
    """Convert a natural note name to MIDI pitch (C4 = 60)."""
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return 12 * (octave + 1) + semitones[name.upper()]


def midi_to_frequency(pitch: int) -> int:
    """Convert MIDI pitch to the nearest whole-number frequency in Hz."""
    return round(440 * (2 ** ((pitch - 69) / 12)))
