"""Run Vinyl Studio. The first launch installs dependencies in a local environment."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, help='Suggest an adapter in the GUI; LoRA still starts OFF')
    args = parser.parse_args()
    from startup import ensure_runtime
    if not ensure_runtime():
        return
    os.environ.pop('MUSIC_LORA_ADAPTER', None)
    from vinyl_gui import VinylStudio
    app = VinylStudio(adapter_hint=args.adapter)
    if not app.ml_generator.loaded:
        from tkinter import messagebox
        messagebox.showerror('Model could not load', app.ml_generator.load_error, parent=app.root)
        app._on_close()
        raise SystemExit(1)
    app.run()


if __name__ == "__main__":
    main()
