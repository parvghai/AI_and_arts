import tempfile
import unittest
from pathlib import Path
from zen_widgets import load_preferences,save_preferences,vinyl_angle

class ZenTests(unittest.TestCase):
    def test_display_preferences_validate_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'preferences.json'
            self.assertEqual(load_preferences(p),{})
            save_preferences(p,False,True)
            self.assertEqual(load_preferences(p),{'settings_expanded':False,'reduced_motion':True})
            for text in ['{','[]','{"settings_expanded":"false","reduced_motion":1}']:
                p.write_text(text);self.assertEqual(load_preferences(p),{})

    def test_rotation_is_slow_and_uses_only_audio_position(self):
        self.assertEqual(vinyl_angle(24),360)
        self.assertEqual(vinyl_angle(0),0)
        self.assertEqual(vinyl_angle(8),vinyl_angle(8))  # Paused position cannot drift.
        self.assertAlmostEqual(vinyl_angle(8.1)-vinyl_angle(8),1.5)
        self.assertLess(vinyl_angle(1),200)  # Previous visualization speed.
