"""Launch the modern studio. LoRA starts OFF and accepts an optional adapter hint."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UI.moder_main import main


if __name__ == '__main__':
    main()
