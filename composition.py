"""Composition and interaction history, kept separate by performer."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class NoteEvent:
    pitch: int
    start: float
    duration: float
    velocity: int = 82
    source: str = "ai"
    label: str = ""


@dataclass
class Interaction:
    timestamp: str
    human_note: str
    ai_notes: list[str]
    duration: float
    bpm: int
    events: list[NoteEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Composition:
    def __init__(self) -> None:
        self.interactions: list[Interaction] = []
        self.events: list[NoteEvent] = []

    @property
    def end_time(self) -> float:
        return max((e.start + e.duration for e in self.events), default=0.0)

    def add_interaction(self, human_note: str, ai_events: list[NoteEvent],
                        duration: float, bpm: int, metadata: dict[str, Any] | None = None,
                        timestamp: str | None = None) -> Interaction:
        start = self.end_time
        human = NoteEvent(60, start, human_note_duration(bpm), 100, "human", human_note)
        # Pitch is assigned by the caller to avoid duplicating note mapping here.
        human_event = next((e for e in ai_events if e.source == "human"), human)
        if human_event is human:
            human_event.pitch = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71}[human_note]
            ai_events.insert(0, human_event)
        ai_events.sort(key=lambda e: (e.start, e.pitch, e.source))
        self.events.extend(ai_events)
        self.events.sort(key=lambda e: (e.start, 0 if e.source == "human" else 1, e.pitch))
        interaction = Interaction(timestamp or datetime.now(timezone.utc).isoformat(), human_note,
                                  [e.label for e in ai_events if e.source == "ai"],
                                  duration, bpm, list(ai_events), metadata or {})
        self.interactions.append(interaction)
        return interaction

    def undo(self) -> None:
        if not self.interactions:
            return
        removed = self.interactions.pop()
        count = len(removed.events)
        if count:
            del self.events[-count:]

    def clear(self) -> None:
        self.interactions.clear()
        self.events.clear()

    def to_dict(self) -> dict[str, Any]:
        return {"created_at": datetime.now(timezone.utc).isoformat(),
                "interactions": [{**asdict(i), "events": [asdict(e) for e in i.events]}
                                 for i in self.interactions],
                "events": [asdict(e) for e in self.events]}


def human_note_duration(bpm: int) -> float:
    """Keep the human cue duration consistent between placement and inference."""
    return min(0.22, 60.0 / max(1, int(bpm)) * 0.5)


def validate_timeline(events: list[NoteEvent], continuation_boundary: float | None = None,
                      monophonic_sources: tuple[str, ...] = ("ai",),
                      tolerance: float = 1e-7) -> dict[str, Any]:
    """Report ordering, duration, delta, overlap, duplicate, and boundary defects.

    Overlap checks are scoped by source so simultaneous human/AI layers or explicit
    chord tracks are not mistaken for accidental same-voice overlaps.
    """
    issues: dict[str, list[dict[str, Any]]] = {
        "out_of_order": [], "negative_durations": [], "negative_delta_times": [],
        "accidental_overlaps": [], "duplicates": [], "before_continuation_boundary": [],
    }
    seen: dict[tuple[Any, ...], int] = {}
    previous_start: float | None = None
    last_by_source: dict[str, tuple[int, NoteEvent]] = {}
    for index, event in enumerate(events):
        if previous_start is not None and event.start < previous_start - tolerance:
            issues["out_of_order"].append({"index": index, "start": event.start,
                                            "previous_start": previous_start})
        if previous_start is not None and event.start - previous_start < -tolerance:
            issues["negative_delta_times"].append({"index": index,
                                                     "delta_time": event.start - previous_start})
        previous_start = event.start
        if event.duration < -tolerance:
            issues["negative_durations"].append({"index": index, "duration": event.duration})
        if continuation_boundary is not None and event.start < continuation_boundary - tolerance:
            issues["before_continuation_boundary"].append({"index": index, "start": event.start,
                                                            "boundary": continuation_boundary})
        key = (event.source, int(event.pitch), round(event.start, 6), round(event.duration, 6),
               int(event.velocity))
        if key in seen:
            issues["duplicates"].append({"index": index, "duplicate_of": seen[key],
                                          "pitch": event.pitch, "start": event.start})
        else:
            seen[key] = index
        if event.source in monophonic_sources:
            prior = last_by_source.get(event.source)
            if prior and event.start < prior[1].start + prior[1].duration - tolerance:
                issues["accidental_overlaps"].append({"index": index, "overlaps_index": prior[0],
                    "start": event.start, "previous_end": prior[1].start + prior[1].duration,
                    "source": event.source})
            if prior is None or event.start >= prior[1].start:
                last_by_source[event.source] = (index, event)
    return {"valid": not any(issues.values()),
            "overlap_detected": bool(issues["accidental_overlaps"]),
            "counts": {key: len(value) for key, value in issues.items()},
            "issues": issues}


def sanitize_timeline(events: list[NoteEvent], continuation_boundary: float,
                      monophonic_sources: tuple[str, ...] = ("ai",)) -> tuple[list[NoteEvent], dict[str, Any]]:
    """Repair current generated events while preserving pitches and note lengths.

    The pre-sanitize report is returned and remains visible to callers for logging;
    same-source monophonic notes are shifted forward instead of being dropped.
    """
    before = validate_timeline(events, continuation_boundary, monophonic_sources)
    sanitized: list[NoteEvent] = []
    seen_original: set[tuple[Any, ...]] = set()
    seen_sanitized: set[tuple[Any, ...]] = set()
    last_end: dict[str, float] = {}
    for original in sorted(events, key=lambda e: (e.start, 0 if e.source == "human" else 1, e.pitch)):
        event = NoteEvent(int(original.pitch), float(original.start), float(original.duration),
                          int(original.velocity), original.source, original.label)
        original_key = (event.source, event.pitch, round(event.start, 6),
                        round(event.duration, 6), event.velocity)
        if original_key in seen_original:
            continue
        seen_original.add(original_key)
        if event.duration <= 0:
            event.duration = 0.08
        event.start = max(float(continuation_boundary), event.start)
        if event.source in monophonic_sources:
            event.start = max(event.start, last_end.get(event.source, float(continuation_boundary)))
        key = (event.source, event.pitch, round(event.start, 6), round(event.duration, 6), event.velocity)
        if key in seen_sanitized:
            continue
        seen_sanitized.add(key)
        sanitized.append(event)
        if event.source in monophonic_sources:
            last_end[event.source] = event.start + event.duration
    after = validate_timeline(sanitized, continuation_boundary, monophonic_sources)
    return sanitized, {"before": before, "after": after,
                       "events_before_sanitization": len(events),
                       "events_after_sanitization": len(sanitized),
                       "changed": before["valid"] is False or len(events) != len(sanitized)}
