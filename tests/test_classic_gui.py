"""Original repository UI layout and classic interaction behavior."""
import ast
import gc
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch
from composition import NoteEvent
from gui import MusicApp


class ClassicTests(unittest.TestCase):
    def test_upstream_visual_methods_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        def methods(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MusicApp')
            # The sole intended addition to the original layout is the LoRA hook.
            for method in cls.body:
                if isinstance(method,ast.FunctionDef) and method.name=='_build':
                    method.body=[n for n in method.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_build_lora_controls')]
            return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
        upstream=methods(root/'reports/upstream_ui/gui.py.reference');actual=methods(root/'gui.py')
        for name in ('_build','_button','_draw_timeline','_refresh'):
            self.assertEqual(actual[name],upstream[name],name)

    def test_original_controls_append_play_undo_and_export(self):
        root=tk.Tk();root.withdraw()
        with tempfile.TemporaryDirectory() as tmp,patch('gui.tk.Tk',return_value=root),patch('gui.MidiPlayer') as player,patch('gui.LSTMGenerator') as loader,patch('gui.LOG_DIR',Path(tmp)):
            loader.return_value.loaded=False;loader.return_value.config={}
            app=MusicApp()
            self.assertEqual(root.cget('background'),'#11151d')
            self.assertEqual(app.duration.get(),'1 second')
            events=[NoteEvent(64,.3,.4,80,'ai','E')]
            app._finish_generation('C',1,100,'2026-10-29T00:00:00+00:00',events,{},False,None)
            self.assertEqual(len(app.composition.events),2)
            player.return_value.play.assert_called_once()
            with patch('gui.filedialog.asksaveasfilename',return_value=str(Path(tmp)/'music.mid')):
                app._export_midi()
            self.assertTrue((Path(tmp)/'music.mid').exists())
            app._undo();self.assertEqual(app.composition.events,[])
            app._closed=True
            for token in root.tk.call('after','info'):root.after_cancel(token)
            root.destroy()
            del app
        gc.collect()


if __name__ == '__main__':unittest.main()
