# Vinyl Studio implementation

## Repository integration — 24 November 2026

The default launcher opens Vinyl Studio. `--classic`, `--modern` and `--studio`
select the original, purple modern and extended editing interfaces. The modern
interface from upstream commit `3ea64c2` is integrated through `--modern` and
`UI/modern_main.py`. The earlier `UI/moder_main.py` entry point remains supported.

Generation settings are collapsible, and display preferences are saved locally.
The vinyl animation makes one revolution per 24 seconds of scheduler time.
Reduced motion disables vinyl rotation and ambient animation.

Session restoration rejects empty or malformed phrase histories before replacing
the composition. Older event records can omit channel and program fields and use
the existing NoteEvent defaults. A replaced playback worker cannot publish a stale
Ready status or reset the current playback device during cleanup.

Validation on 24 November 2026 used `python -B -m unittest discover -s tests -v`.
There were 64 tests, with 63 passing and one skipped because CUDA was unavailable.
The suite covers launcher selection, legacy modern launcher paths, restoration,
playback replacement and isolated GUI preferences. A separate Tkinter smoke check
exercised modern UI startup, generation display, phrase replay and undo with
mocked playback and model loading.

## Initial implementation — 9 November 2026

## Delivered

The default `python -B main.py` interface is a native Tkinter monochrome vinyl studio.
`--classic` opens the unchanged original GUI and `--studio` retains the extended reference/editor interface.
No GitHub changes, model retraining, architecture changes, or checkpoint writes were performed.

- Three-column layout, monochrome controls, turntable with grooves/reflections/tonearm.
- One visual mark per actual MIDI event; pitch determines radius, start determines angle,
  duration determines arc extent, velocity determines stroke width. Views never mutate events.
- Circular sequencer and performer timeline share the same composition. No piano roll in the vinyl UI.
- Existing note keyboard shortcuts, model selection, durations, tempo, temperature, LoRA adapter
  validation/selection, MIDI/log export, generation workers and autosave are reused.
- Phrase history remains chronological and horizontally scrollable; clicking a phrase replays its events.
- Regeneration replaces the last phrase and restores its events if generation fails.
- Session JSON restoration validates all events and shared history identities before changing state.
  Editing an imported session creates a new autosave, preserving its source file.
- Transport supports play/pause/stop and seeking (seek resumes playback with a trimmed copy of events).
- Position is published by the MIDI scheduling worker, not an animation timer. Rendering polls at 20 Hz
  and avoids redrawing when transport and tempo are unchanged. MIDI pause preserves scheduler position;
  stop resets the UI transport. Native Windows MIDI is the precise playback path. The existing blocking
  tone fallback remains available but has coarse position updates during each tone.

## Files

- `vinyl_gui.py`: interface, canvas rendering, transport controls, history, restore and themed LoRA dialog.
- `vinyl_view.py`: event mapping, seek adapter and validated session restoration.
- `midi_player.py`: expose scheduler position and playback state; audio generation unchanged.
- `main.py`: vinyl default and `--classic` switch.
- `tests/test_vinyl.py`: five data/scheduler/GUI tests.
- `tests/test_lora_gui.py`: launcher test targets the new default constructor.
- `README.md`: usage and behavior.

## Verification

Full suite: `python -B -m unittest discover -s tests -v` — 57 run, 56 passed,
1 skipped (`CUDA not installed`). Includes real local LSTM checkpoint and LoRA off/on/off tests
for both model sizes. This is not a fresh GPU validation.

After final grid/legend changes: six targeted tests passed (the five new tests and launcher test).
Tests cover mapping identity/timing, seek data preservation, restore/undo/invalid sessions,
actual scheduler pause/resume/end with a mocked MIDI device, GUI view switching, MIDI export,
phrase replay, LoRA settings, and regeneration failure rollback.

Live desktop verification: launched the interface and generated a real 3x512 LSTM continuation;
observed its six real notes in the turntable, circular sequencer, and phrase history, with active
transport time/tonearm and Windows MIDI status. Inspected normal and compact desktop layouts.
The compact pass found insufficient history space; the visualization's requested height was reduced
so it expands into available space while reserving room for history and transport.

No claim of improved musical quality or full human listening evaluation is made.

Final compact geometry and stop-state polish: the five vinyl tests passed again. Native MIDI audio was invoked in the live checks; no subjective audio quality claim is made.
