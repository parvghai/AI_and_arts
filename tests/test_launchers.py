"""Regression coverage for interface routing and the modern launcher filename."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import main


class LauncherTests(unittest.TestCase):
    def test_interface_selection(self):
        for flag, target in [([], 'vinyl_gui.VinylStudio'),
                             (['--classic'], 'gui.MusicApp'),
                             (['--modern'], 'UI.modern_gui.ModernMusicApp'),
                             (['--studio'], 'studio_gui.MusicApp')]:
            with self.subTest(flag=flag), patch('sys.argv', ['main.py', *flag]), patch(target) as app:
                main.main()
                app.assert_called_once_with(**({} if flag == ['--studio'] else {'adapter_hint': None}))
                app.return_value.run.assert_called_once_with()

    def test_conflicting_interfaces_are_rejected(self):
        with patch('sys.argv', ['main.py', '--classic', '--modern']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main.main()
        self.assertEqual(error.exception.code, 2)

    def test_modern_launchers_work_outside_project_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ['modern_main.py', 'moder_main.py']:
                with self.subTest(filename=filename):
                    result = subprocess.run(
                        [sys.executable, '-B', str(root / 'UI' / filename), '--help'],
                        cwd=tmp, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--adapter', result.stdout)
