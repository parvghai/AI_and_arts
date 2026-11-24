import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from composition import Composition, NoteEvent
from vinyl_view import event_marks, seek_events, restore_composition
from midi_player import MidiPlayer

class VinylDataTests(unittest.TestCase):
    def test_marks_preserve_identity_order_and_exact_timing(self):
        events=[NoteEvent(60,0,.2,100,'human','C'),NoteEvent(72,.2,.4,70,'ai','C')]
        marks=event_marks(events,.6)
        self.assertIs(marks[0][0],events[0]);self.assertIs(marks[1][0],events[1])
        self.assertAlmostEqual(marks[1][1],120);self.assertAlmostEqual(marks[1][2],240)
        self.assertEqual(marks[1][3],1)
        self.assertEqual(event_marks([],0),[])

    def test_seek_preserves_pitch_velocity_origin_and_trims_sounding_note(self):
        e=NoteEvent(65,1,2,91,'human','F',2,3)
        out=seek_events([e],2)
        self.assertEqual(out[0],NoteEvent(65,2,1,91,'human','F',2,3))
        self.assertEqual(e.duration,2);self.assertEqual(seek_events([e],3),[])

    def test_restore_roundtrip_and_undo(self):
        c=Composition();c.add_interaction('C',[NoteEvent(64,.22,.5,label='E')],1,100)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'session.json';p.write_text(json.dumps(c.to_dict()),encoding='utf-8')
            out=restore_composition(p)
            self.assertEqual(out.events,c.events)
            out.undo();self.assertEqual(out.events,[])
            data=c.to_dict();data['interactions'][0]['events'][0]['pitch']=999
            p.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):restore_composition(p)

    def test_audio_scheduler_position_pause_resume_and_end(self):
        with patch.object(MidiPlayer,'_open_windows_midi'):
            p=MidiPlayer()
        p._midi=MagicMock();p._handle=0
        p.play([NoteEvent(60,5,.35)])
        deadline=time.monotonic()+1
        while not p.is_playing and time.monotonic()<deadline:time.sleep(.005)
        time.sleep(.05);self.assertGreater(p.position,5)
        p.pause();time.sleep(.04);pos=p.position;time.sleep(.08)
        self.assertEqual(pos,p.position);self.assertFalse(p.is_playing)
        p.resume();p._thread.join(1)
        self.assertFalse(p.is_playing);self.assertAlmostEqual(p.position,5.35,places=2)
        p.close()

    def test_restore_rejects_malformed_history_and_accepts_legacy_event_defaults(self):
        c = Composition()
        c.add_interaction('C', [NoteEvent(64, .22, .5, label='E')], 1, 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'session.json'
            data = c.to_dict()
            for row in data['events'] + data['interactions'][0]['events']:
                row.pop('channel')
                row.pop('program')
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(restore_composition(path).events, c.events)
            for invalid in [[], {'interactions': [None]},
                            {'interactions': [{'events': []}], 'events': []}]:
                with self.subTest(invalid=invalid):
                    path.write_text(json.dumps(invalid), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        restore_composition(path)

    def test_replaced_playback_worker_does_not_reset_current_transport(self):
        with patch.object(MidiPlayer, '_open_windows_midi'):
            player = MidiPlayer()
        player._midi = None
        statuses = []
        player.on_status = statuses.append
        old_stop = player._stop

        def replace_during_tone(*_args):
            old_stop.set()
            player._stop = threading.Event()
            player._playing = True

        player._beep = replace_during_tone
        player._run([NoteEvent(60, 0, .1)], old_stop, threading.Event())
        self.assertTrue(player.is_playing)
        self.assertNotIn('Ready', statuses)
        player.close()

class VinylGUITests(unittest.TestCase):
    def test_controls_data_views_export_history_and_failure_rollback(self):
        from vinyl_gui import VinylStudio, BG
        from tkinter import Tk
        from types import SimpleNamespace
        root=Tk();root.withdraw()
        player=MagicMock();player.position=0.;player.is_playing=False;player.is_paused=False;player._playing=False
        with tempfile.TemporaryDirectory() as tmp,patch('gui.tk.Tk',return_value=root),patch('gui.MidiPlayer',return_value=player),patch('gui.LSTMGenerator') as loader,patch('gui.LOG_DIR',Path(tmp)),patch('vinyl_gui.LOG_DIR',Path(tmp)):
            loader.return_value.loaded=False;loader.return_value.config={}
            app=VinylStudio();root.update_idletasks()
            try:
                self.assertEqual(root.cget('bg'),BG);self.assertFalse(app.lora.enabled.get())
                app._preferences_path=Path(tmp)/'preferences.json'
                app.model_choice.set('LSTM 4x768 (experimental)')
                app.bpm.set(123);app.temperature.set(.9);app.duration.set('2 seconds')
                app.reduced_motion.set(True)
                app.generation_settings.toggle();root.update_idletasks()
                self.assertFalse(app.generation_settings.expanded)
                self.assertEqual(app.generation_settings.viewport.winfo_reqheight(),1)
                app.generation_settings.toggle();root.update_idletasks()
                self.assertEqual(app.bpm.get(),123);self.assertEqual(app.temperature.get(),.9)
                self.assertEqual(app.duration.get(),'2 seconds')
                self.assertEqual(app.model_choice.get(),'LSTM 4x768 (experimental)')
                self.assertTrue(app.generation_settings.viewport.winfo_reqheight()>100)
                from zen_widgets import load_preferences
                self.assertEqual(load_preferences(app._preferences_path),{'settings_expanded':True,'reduced_motion':True})
                app._paint_landscapes()
                self.assertTrue(app.bottom_landscape.find_withtag('landscape'))
                self.assertTrue(app.sidebar_landscape.find_withtag('landscape'))
                app._finish_generation('C',1,100,'2026-11-09T00:00:00+00:00',[NoteEvent(64,.22,.5,label='E')],{},False,None)
                self.assertEqual(len(app.composition.events),2)
                before=list(app.composition.events)
                app._switch_view('Timeline');app._switch_view('Circular')
                self.assertEqual(before,app.composition.events)
                app._replay_click(SimpleNamespace(x=10));player.play.assert_called_with(app.composition.interactions[0].events)
                with patch('gui.filedialog.asksaveasfilename',return_value=str(Path(tmp)/'music.mid')):app._export_midi()
                self.assertTrue((Path(tmp)/'music.mid').exists())
                with patch.object(app,'_choose_note'):
                    app._regenerate();self.assertEqual(app.composition.events,[])
                app._generation_failed('test failure');self.assertEqual(before,app.composition.events)
                app.lora.enabled.set(True);app.lora.show();self.assertIsNotNone(app.lora.window)
                app.lora.disable();self.assertFalse(app.lora.enabled.get())
                app._undo();self.assertEqual(app.composition.events,[])
            finally:
                app._closed=True
                for t in root.tk.call('after','info'):root.after_cancel(t)
                root.destroy()
