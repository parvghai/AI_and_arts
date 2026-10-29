"""Real Tk widgets/event loop with local fixture dialogs and silent player mocks."""
import builtins
import gc
import tempfile
import time
from pathlib import Path
import tkinter as tk
import unittest
from unittest.mock import patch,MagicMock
from studio_gui import MusicApp
from tests.test_reference import fixture


class GUITests(unittest.TestCase):
    def setUp(self):
        try:self.root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        self.root.withdraw()
        self.player_patch=patch('studio_gui.MidiPlayer',return_value=MagicMock());self.player_patch.start()
        self.app=MusicApp(self.root,autoload=False)
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)

    def tearDown(self):
        if not hasattr(self,'app'):return
        self.app._closed=True;self.app.preview.stop()
        for token in self.root.tk.call('after','info'):self.root.after_cancel(token)
        self.root.destroy();self.player_patch.stop();self.tmp.cleanup()
        del self.app
        gc.collect()

    def wait(self):
        end=time.monotonic()+15
        while self.app._busy and time.monotonic()<end:self.root.update();time.sleep(.01)
        self.root.update();self.assertFalse(self.app._busy,self.app.status.get())

    def test_generate_edit_accept_save_open_undo(self):
        a=self.app;a._generate();self.wait()
        self.assertEqual(len(a.candidates),3);self.assertEqual(a.composition.events,[])
        a.candidate_list.selection_set('1');a._edit_candidate();self.root.update()
        a.notes_tree.selection_set('1');a._note_selected();a.note_vars['pitch'].set('72');a._change_note()
        self.assertEqual(a.candidates[1].events[1].pitch,72)
        a._accept();self.assertGreater(len(a.composition.events),1)
        path=self.folder/'session.json'
        with patch('studio_gui.filedialog.asksaveasfilename',return_value=str(path)):a._save()
        expected=list(a.composition.events);a._clear();self.assertEqual(a.composition.events,[])
        a._undo();self.assertEqual(a.composition.events,expected)
        with patch('studio_gui.messagebox.askyesno',return_value=True),patch('studio_gui.filedialog.askopenfilename',return_value=str(path)):a._open()
        self.assertEqual(a.composition.events,expected)
        mid=self.folder/'session.mid'
        with patch('studio_gui.filedialog.asksaveasfilename',return_value=str(mid)):a._export_midi()
        self.assertTrue(mid.exists(),a.status.get())

    def test_classic_note_click_appends_plays_and_undoes(self):
        a=self.app
        self.assertEqual(a.root.cget('background'),'#11151d')
        self.assertEqual(a.seconds.get(),1)
        a._choose_note('C');self.wait()
        first=list(a.composition.events)
        self.assertTrue(first);self.assertEqual(first[0].source,'human')
        self.assertEqual(a.candidates,[])
        a.player.play.assert_called_once_with(first)
        a._choose_note('D');self.wait()
        self.assertGreater(len(a.composition.events),len(first))
        self.assertGreaterEqual(a.player.play.call_args.args[0][0].start,max(e.start+e.duration for e in first))
        a._undo();self.assertEqual(a.composition.events,first)

    def test_reference_import_generate_reject_and_failure_keeps_reference(self):
        a=self.app;wav=self.folder/'reference.wav';fixture(wav)
        with patch('studio_gui.filedialog.askopenfilename',return_value=str(wav)):a._import_reference()
        self.wait();self.assertIsNotNone(a.reference,a.status.get())
        a.follow_tempo.set(True);a.influence.set(1)
        self.assertEqual(a._settings().bpm,a.reference.tempo)
        a._generate();self.wait();self.assertTrue(a.candidates)
        a._reject();self.assertEqual(a.candidates,[]);self.assertEqual(a.composition.events,[])
        bad=self.folder/'bad.mp3';bad.write_bytes(b'broken')
        old=a.reference
        with patch('studio_gui.filedialog.askopenfilename',return_value=str(bad)):a._import_reference()
        self.wait();self.assertIs(a.reference,old)

    def test_no_torch_model_selection_falls_back(self):
        original=builtins.__import__
        def blocked(name,*args,**kwargs):
            if name=='ml.inference':raise ImportError('simulated optional dependency missing')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',blocked):self.app._load_model();self.wait()
        self.assertIsNone(self.app.ml_generator)
        self.assertIn('fallback',self.app.model_status.get())
        self.assertEqual(str(self.app.model_box.cget('state')),'readonly')

    def test_selected_range_regeneration_is_uncommitted_and_undoable(self):
        a=self.app;a._generate();self.wait();a._accept();previous=list(a.composition.events)
        a.edit_target.set('Composition');a._refresh_notes();a.notes_tree.selection_set(a.notes_tree.get_children())
        a._generate(replace=True);self.wait()
        self.assertEqual(a.composition.events,previous)
        self.assertIsNotNone(a.replace_range)
        a._accept();a._undo();self.assertEqual(a.composition.events,previous)

if __name__=='__main__':unittest.main()
