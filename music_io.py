"""Validated MIDI and versioned sessions. Times are absolute seconds, not beats."""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import copy
import json
import math
import os
import tempfile

from composition import Composition, NoteEvent, validate_timeline

MAX_EVENTS = 100_000


def validate_events(events):
    if len(events) > MAX_EVENTS:
        raise ValueError(f"At most {MAX_EVENTS:,} notes are supported")
    report = validate_timeline(sorted(events, key=lambda e: e.start), monophonic_sources=())
    if not report['valid']:
        raise ValueError(f"Invalid notes: {report['counts']}")


def midi_file(events, bpm=100):
    """Keep intentional polyphony; merge same-pitch/channel overlaps for valid note-offs."""
    import mido
    if not math.isfinite(bpm) or not 30 <= bpm <= 240:
        raise ValueError("Tempo must be between 30 and 240 BPM")
    validate_events(events)
    midi = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack(); midi.tracks.append(track)
    tempo = round(60_000_000 / bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    programs = {}
    intervals = {}
    for e in sorted(events, key=lambda e: e.start):
        if e.channel in programs and programs[e.channel] != e.program:
            raise ValueError("Use different MIDI channels for different instruments")
        programs[e.channel] = e.program
        group = intervals.setdefault((e.channel, e.pitch), [])
        if group and e.start < group[-1][1] - 1e-9:
            group[-1][1] = max(group[-1][1], e.start + e.duration)
        else:
            group.append([e.start, e.start + e.duration, e.velocity])
    for channel, program in sorted(programs.items()):
        if channel != 9:
            track.append(mido.Message('program_change', channel=channel, program=program, time=0))
    ticks = midi.ticks_per_beat * 1_000_000 / tempo
    schedule = []
    for (channel, pitch), spans in intervals.items():
        for start, end, velocity in spans:
            on = round(start*ticks); off = max(on+1, round(end*ticks))
            schedule.extend([(on, 1, mido.Message('note_on', note=pitch, velocity=velocity, channel=channel)),
                             (off, 0, mido.Message('note_off', note=pitch, velocity=0, channel=channel))])
    last = 0
    for tick, _, msg in sorted(schedule, key=lambda row: (row[0], row[1])):
        track.append(msg.copy(time=tick-last)); last = tick
    return midi


def read_midi(path):
    """Import tempo-aware MIDI without quantization or removal of polyphony."""
    import mido
    path = Path(path)
    if path.stat().st_size > 16*1024*1024:
        raise ValueError("MIDI files must be at most 16 MiB")
    mid = mido.MidiFile(path)
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise ValueError("Independent type-2 sequences and SMPTE timing are unsupported")
    schedule = []
    for ti, track in enumerate(mid.tracks):
        tick = 0
        for order, msg in enumerate(track):
            tick += msg.time
            schedule.append((tick, ti, order, msg))
    tempo = 500_000; last_tick = 0; seconds = 0.; active = {}; programs = {}; notes = []
    def close(key, end):
        for start, vel, program in active.pop(key, []):
            if end > start:
                notes.append(NoteEvent(key[2], start, end-start, vel, 'human', str(key[2]), key[1], program))
    for tick, ti, _, msg in sorted(schedule, key=lambda row: row[:3]):
        seconds += mido.tick2second(tick-last_tick, mid.ticks_per_beat, tempo); last_tick = tick
        if msg.type == 'set_tempo': tempo = msg.tempo
        elif msg.type == 'program_change': programs[(ti,msg.channel)] = msg.program
        elif msg.type == 'note_on' and msg.velocity:
            active.setdefault((ti,msg.channel,msg.note),[]).append((seconds,msg.velocity,programs.get((ti,msg.channel),0)))
        elif msg.type in ('note_on','note_off'):
            key=(ti,msg.channel,msg.note)
            if active.get(key):
                start,vel,program=active[key].pop(0)
                if seconds > start:notes.append(NoteEvent(msg.note,start,seconds-start,vel,'human',str(msg.note),msg.channel,program))
        elif msg.type == 'end_of_track':
            for key in list(active):
                if key[0] == ti:close(key,seconds)
        if len(notes)>MAX_EVENTS:raise ValueError('MIDI contains too many notes')
    for key in list(active):close(key,seconds)
    if not notes:raise ValueError('MIDI contains no usable notes')
    notes.sort(key=lambda e:(e.start,e.pitch));validate_events(notes)
    return notes


class CompositionDocument:
    """Bounded undo history, explicit saves, and no automatic full-history debug dumps."""
    def __init__(self):
        self.composition=Composition();self.settings={};self._history=[];self.dirty=False

    def set_events(self, events):
        events=sorted(copy.deepcopy(events),key=lambda e:(e.start,e.pitch))
        validate_events(events)
        self._history.append(copy.deepcopy(self.composition.events));self._history=self._history[-100:]
        self.composition.events=events;self.composition.interactions=[];self.dirty=True

    def undo(self):
        if self._history:
            self.composition.events=self._history.pop();self.composition.interactions=[];self.dirty=True

    def save(self,path):
        path=Path(path);validate_events(self.composition.events)
        payload={'schema_version':2,'events':[asdict(e) for e in self.composition.events], 'settings':self.settings}
        data=json.dumps(payload,indent=2,allow_nan=False)
        # Atomic save beside the destination; remove temporary files even on failure.
        fd,tmp=tempfile.mkstemp(prefix='.composition_',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as stream:stream.write(data)
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        self.dirty=False

    def load(self,path):
        path=Path(path)
        if path.stat().st_size>32*1024*1024:raise ValueError('Session exceeds 32 MiB')
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version',1) not in (1,2):raise ValueError('Unsupported session version')
        allowed={f.name for f in fields(NoteEvent)}
        events=[NoteEvent(**{k:v for k,v in row.items() if k in allowed}) for row in payload['events']]
        validate_events(events)
        settings=payload.get('settings',{})
        if not isinstance(settings,dict):raise ValueError('Invalid session settings')
        self.set_events(events);self.settings=settings;self._history=[];self.dirty=False
