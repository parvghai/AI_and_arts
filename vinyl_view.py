"""Pure presentation adapters: retain original MIDI events and their ordering."""
import json
import math
from dataclasses import replace
from composition import Composition, Interaction, NoteEvent, validate_timeline


def event_marks(events, total):
    """Each display mark references exactly one unmodified source event."""
    total = max(total, .001)
    low = min((e.pitch for e in events), default=60)
    high = max((e.pitch for e in events), default=72)
    return [(e, 360 * e.start / total, 360 * e.duration / total,
             (e.pitch-low)/max(1, high-low)) for e in events]


def seek_events(events, position):
    return [replace(e, start=max(position, e.start),
                    duration=e.start+e.duration-max(position, e.start))
            for e in events if e.start+e.duration > position]


def restore_composition(path):
    """Validate before replacing live state; preserve shared event identities for undo."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get('interactions'), list):
        raise ValueError('Session must contain a phrase history list')
    result = Composition()
    for item in data['interactions']:
        if not isinstance(item, dict) or not isinstance(item.get('events'), list) or not item['events']:
            raise ValueError('Each phrase must contain MIDI events')
        events = [NoteEvent(**row) for row in item['events']]
        if not validate_timeline(events, monophonic_sources=())['valid']:
            raise ValueError('Invalid phrase MIDI events')
        if not isinstance(item.get('ai_notes'), list) or not all(isinstance(note, str) for note in item['ai_notes']):
            raise ValueError('Invalid AI note labels')
        if item['human_note'] not in 'CDEFGAB' or len(item['human_note']) != 1:
            raise ValueError('Invalid human note')
        if not isinstance(item['bpm'], (int, float)) or not 1 <= item['bpm'] <= 400:
            raise ValueError('Invalid tempo')
        if not math.isfinite(item['duration']) or item['duration'] <= 0:
            raise ValueError('Invalid phrase duration')
        result.interactions.append(Interaction(item['timestamp'], item['human_note'],
            item['ai_notes'], item['duration'], item['bpm'], events, item.get('metadata', {})))
        result.events.extend(events)
    result.events.sort(key=lambda e:(e.start, 0 if e.source=='human' else 1, e.pitch))
    if not validate_timeline(result.events, monophonic_sources=())['valid']:
        raise ValueError('Invalid session MIDI events')
    saved_events = [NoteEvent(**row) for row in data['events']]
    if saved_events != result.events:
        raise ValueError('Session event list does not match phrase history')
    return result
