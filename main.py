"""Launch the original studio. LoRA always starts OFF and is enabled in the GUI."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, help='Suggest an adapter in the GUI; LoRA still starts OFF')
    parser.add_argument('--studio', action='store_true', help='Open optional reference/candidate/editing tools')
    args = parser.parse_args()
    os.environ.pop('MUSIC_LORA_ADAPTER', None)
    if args.studio:
        if args.adapter:
            parser.error('--adapter is a hint for the default GUI; omit --studio to select LoRA there')
        from studio_gui import MusicApp
        MusicApp().run()
    else:
        from gui import MusicApp
        MusicApp(adapter_hint=args.adapter).run()


if __name__ == "__main__":
    main()
