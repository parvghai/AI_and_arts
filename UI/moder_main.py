"""Launch the modern studio UI. Behaviour matches main.py; LoRA always starts OFF."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, help='Suggest an adapter in the GUI; LoRA still starts OFF')
    args = parser.parse_args()
    os.environ.pop('MUSIC_LORA_ADAPTER', None)
    from UI.modern_gui import ModernMusicApp
    ModernMusicApp(adapter_hint=args.adapter).run()


if __name__ == "__main__":
    main()
