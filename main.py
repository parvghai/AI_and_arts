"""Launch the monochrome vinyl studio. LoRA always starts OFF and is enabled in the GUI."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, help='Suggest an adapter in the GUI; LoRA still starts OFF')
    interface = parser.add_mutually_exclusive_group()
    interface.add_argument('--studio', action='store_true', help='Open optional reference/candidate/editing tools')
    interface.add_argument('--classic', action='store_true', help='Open the original interface')
    interface.add_argument('--modern', action='store_true', help='Open the purple modern studio interface')
    args = parser.parse_args()
    os.environ.pop('MUSIC_LORA_ADAPTER', None)
    if args.studio:
        if args.adapter:
            parser.error('--adapter is a hint for the default GUI; omit --studio to select LoRA there')
        from studio_gui import MusicApp
        MusicApp().run()
    else:
        if args.classic:
            from gui import MusicApp
        elif args.modern:
            from UI.modern_gui import ModernMusicApp as MusicApp
        else:
            from vinyl_gui import VinylStudio as MusicApp
        MusicApp(adapter_hint=args.adapter).run()


if __name__ == "__main__":
    main()
