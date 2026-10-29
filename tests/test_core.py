import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import mido
from composition import Composition,NoteEvent,validate_timeline
from ai_generator import RuleBasedGenerator
from music_io import CompositionDocument,read_midi,midi_file
from ml.preprocessing import parse_midi_bytes


class CoreTests(unittest.TestCase):
    def test_validator_rejects_bad_values(self):
        for kw in ({'start':float('nan')},{'duration':float('inf')},{'duration':0},{'pitch':128},
                   {'velocity':0},{'channel':16},{'program':-1}):
            values=dict(pitch=60,start=0,duration=.2);values.update(kw)
            self.assertFalse(validate_timeline([NoteEvent(**values)])['valid'])

    def test_polyphony_is_valid_when_requested(self):
        notes=[NoteEvent(p,0,.5) for p in [60,64,67]]
        self.assertTrue(validate_timeline(notes,monophonic_sources=())['valid'])
        self.assertFalse(validate_timeline(notes)['valid'])

    def test_fallback_grid(self):
        g=RuleBasedGenerator()
        for bpm in [60,100,160]:
            for seconds in [.5,1,2]:
                for name in 'CDEFGAB':
                    c=Composition();notes=g.generate_continuation(c,name,seconds,bpm)
                    self.assertTrue(validate_timeline(notes)['valid'])
                    c.add_interaction(name,notes,seconds,bpm);c.undo();self.assertEqual(c.events,[])

    def test_session_roundtrip_edit_undo(self):
        d=CompositionDocument();notes=[NoteEvent(60,0,.3),NoteEvent(64,0,.4,channel=1)]
        d.set_events(notes);d.settings={'generation':{'bpm':120}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.json';d.save(path)
            restored=CompositionDocument();restored.load(path)
            self.assertEqual(restored.composition.events,notes);self.assertEqual(restored.settings,d.settings)
            restored.set_events([NoteEvent(72,0,1)]);restored.undo()
            self.assertEqual(restored.composition.events,notes)
            self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_bad_session_does_not_mutate_document(self):
        d=CompositionDocument();d.set_events([NoteEvent(60,0,1)])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.json';p.write_text('{"schema_version":99,"events":[]}')
            with self.assertRaises(ValueError):d.load(p)
        self.assertEqual(len(d.composition.events),1)

    def test_midi_polyphony_and_duration_roundtrip(self):
        notes=[NoteEvent(p,0,.5) for p in [60,64,67]]+[NoteEvent(36,.5,.25,channel=9)]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'notes.mid';midi_file(notes,120).save(p);loaded=read_midi(p)
        self.assertEqual(len(loaded),4);self.assertAlmostEqual(max(e.start+e.duration for e in loaded),.75,places=3)
        self.assertEqual(loaded[-1].channel,9)

    def test_same_pitch_overlap_has_one_matching_off(self):
        mid=midi_file([NoteEvent(60,0,1),NoteEvent(60,.5,1)])
        self.assertEqual(sum(m.type=='note_on' for m in mid.tracks[0]),1)
        self.assertEqual(sum(m.type=='note_off' for m in mid.tracks[0]),1)
        self.assertAlmostEqual(mid.length,1.5)

    def test_note_off_precedes_retrigger(self):
        mid=midi_file([NoteEvent(60,0,.5),NoteEvent(60,.5,.5)])
        messages=[m.type for m in mid.tracks[0] if m.type.startswith('note_')]
        self.assertEqual(messages,['note_on','note_off','note_on','note_off'])

    def test_parser_orphan_closes_at_track_end(self):
        mid=mido.MidiFile(type=1,ticks_per_beat=480)
        mid.tracks.extend([mido.MidiTrack([mido.Message('note_on',note=60,velocity=80),mido.MetaMessage('end_of_track',time=480)]),
                           mido.MidiTrack([mido.Message('note_on',note=64,velocity=80),mido.Message('note_off',note=64,time=1920)])])
        b=io.BytesIO();mid.save(file=b)
        notes=parse_midi_bytes(b.getvalue());self.assertEqual(next(e.duration for e in notes if e.pitch==60),.5)

    def test_type2_rejected(self):
        mid=mido.MidiFile(type=2);mid.tracks.append(mido.MidiTrack([mido.Message('note_on',note=60)]))
        b=io.BytesIO();mid.save(file=b)
        with self.assertRaises(ValueError):parse_midi_bytes(b.getvalue())

if __name__=='__main__':unittest.main()
