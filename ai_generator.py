"""Replaceable local symbolic continuation generator (C major baseline)."""
from composition import Composition, NoteEvent, human_note_duration

PITCH = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71}
SCALE = (60, 62, 64, 65, 67, 69, 71)


class RuleBasedGenerator:
    """A deterministic, context-aware baseline suitable for offline use."""
    def generate_continuation(self, composition: Composition, target_note: str,
                              duration: float, bpm: int) -> list[NoteEvent]:
        target = PITCH[target_note]
        beat = 60.0 / bpm
        count = max(2, round(duration / beat * 2))
        prior = [e.pitch for e in composition.events]
        previous = prior[-1] if prior else target
        # A short motif balances stepwise motion, a consonant skip, and target arrival.
        candidates = [p for p in SCALE if abs(p - previous) <= 4]
        if not candidates:
            candidates = list(SCALE)
        middle = [min(candidates, key=lambda p: abs(p - target)),
                  SCALE[(SCALE.index(target) + 2) % len(SCALE)],
                  SCALE[(SCALE.index(target) + 4) % len(SCALE)]]
        pitches = [middle[i % len(middle)] for i in range(max(0, count - 1))]
        pitches.append(target)
        # Avoid a robotic even grid by alternating eighth and quarter-note lengths.
        weights = [0.5 if i % 3 != 2 else 1.0 for i in range(len(pitches))]
        scale = duration / sum(weights)
        result: list[NoteEvent] = []
        # The incoming human cue is placed at the prior composition boundary;
        # begin this monophonic AI phrase after that cue has finished.
        cursor = composition.end_time + human_note_duration(bpm) + min(0.04, beat * 0.08)
        for i, (pitch, weight) in enumerate(zip(pitches, weights)):
            length = scale * weight
            result.append(NoteEvent(pitch, cursor, max(0.08, length * 0.82),
                                    78 if i % 2 else 86, "ai",
                                    next(n for n, p in PITCH.items() if p == pitch)))
            cursor += length
        return result
