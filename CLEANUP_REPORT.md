# Workspace cleanup — 29 October 2026

Removed 194 previously inventoried disposable files (699,494.0 bytes): 46 Python bytecode caches and 148 verified empty auto-session JSON files. Session hashes were rechecked before deletion; populated sessions were preserved. The machine-readable removal list is `reports/workspace_cleanup_20261029.json`.

All eight original model checkpoint SHA-256 hashes match the pre-upgrade inventory. No dataset, populated historical session, original checkpoint, evaluation history, chart or user composition was removed.

## Layout

- `gui.py`: original repository studio layout; default entry point remains `main.py`.
- `studio_gui.py`: preserved optional reference/candidate/editor interface (`--studio`).
- `ml/lora.py`, `ml/train_lora.py`: adapter implementation and trainer.
- `docs/LORA.md`: training, launch and evaluation instructions.
- `runs/lora/`: local-only adapter runs, ignored by Git. Each run contains adapter files, report and data manifest.
- `reports/upstream_ui/`: pinned upstream UI source/revision for reproducible layout verification.
- `logs/sessions/`: one atomic autosave per app launch, replacing repeated snapshots of the same session. Explicit log export remains available.
- `reports/`: small validation/cleanup evidence; historical reports retained.

The source backup before LoRA is outside the project: `../co_creative_music_before_lora_20261029T170803.zip`. The previous upgrade backup is also preserved outside the project.

Two initial diagnostic runs (`runs/lora/smoke_20261029` and `runs/lora/smoke_4x768_20261029`) encountered a CUDA process-exit failure after saving outputs. Keep them as diagnostics, not released adapters. Subsequent `verified_baseline_20261029` and `verified_large_20261029` runs completed with exit code 0 after the native recurrent training fix. Automatic approval review blocked a final attempt to relocate the diagnostic runs and remove empty cache directories; these remain in their original locations. No further deletion was attempted.
