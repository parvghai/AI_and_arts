import tkinter as tk
import unittest
from unittest.mock import patch, MagicMock
from display_config import configure_windows_dpi, DesignCanvas


class DisplayTests(unittest.TestCase):
    def setUp(self):
        configure_windows_dpi()
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()

    def test_canvas_drawing_and_scrolled_hit_coordinates_agree(self):
        c = DesignCanvas(self.root)
        for scale in (1, 1.5, 2, 1):
            c.design_scale = scale
            item = c.create_rectangle(172, 0, 332, 84)
            self.assertEqual(c.coords(item), [172*scale, 0, 332*scale, 84*scale])
            c.configure(scrollregion=(0, 0, 1720, 84))
            self.assertEqual(float(c.cget('scrollregion').split()[2]), 1720*scale)
            # Physical pointer positions are converted back into phrase indices.
            self.assertEqual(int(c.canvasx(200*scale)//172), 1)

    def test_studio_scales_as_a_unit_without_rounding_drift(self):
        from vinyl_gui import VinylStudio
        player = MagicMock(position=0, is_playing=False, is_paused=False, _playing=False)
        with patch('gui.tk.Tk', return_value=self.root), patch('gui.MidiPlayer', return_value=player), patch('gui.LSTMGenerator') as loader:
            loader.return_value.config = {}
            app = VinylStudio()
        try:
            sidebar = app.sidebar_landscape.master
            for scale in (1, 1.5, 2, 1.5, 1):
                with patch.object(self.root, 'winfo_width', return_value=round(1440*scale)), patch.object(self.root, 'winfo_height', return_value=round(920*scale)):
                    app.layout.fit()
                self.assertEqual(int(sidebar.cget('width')), round(170*scale))
                self.assertEqual(int(app.history.cget('height')), round(84*scale))
                self.assertEqual(app.visual.design_scale, scale)
                self.assertEqual(app.history.design_scale, scale)
                self.assertEqual(app.bpm.get(), 100)
        finally:
            app._closed = True
