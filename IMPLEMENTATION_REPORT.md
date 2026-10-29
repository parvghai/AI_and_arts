# LoRA and original UI restoration — 29 October 2026

> Historical initial LoRA implementation. The current optional GUI behavior and latest tests are documented in [Optional LoRA GUI report](reports/lora_gui_report.md). LoRA now always starts OFF.

## Delivered

The default single-screen layout is restored from `parvghai/AI_and_arts`, commit `79b7255daa09821fceca6d6b0f404a8ad77a6f5c`. The layout builder, button rendering, sequence rendering and timeline drawing are checked against the pinned source. Original note buttons, keyboard shortcuts, model selector, continuation duration, tempo, temperature, Play/Pause/Stop/Undo/Clear, MIDI export and interaction-log export remain. The existing status label adds LoRA active/inactive only when an adapter is requested. No LoRA controls were added. Extended tools are preserved separately in `studio_gui.py`, launched with `--studio`.

`ml/lora.py` implements zero-initialized low-rank updates to the recurrent matrices, tempo projection and prediction heads. Base embeddings, biases and weights remain frozen. Saved adapters contain only low-rank tensors and metadata, with an exact base-checkpoint hash. Loading validates keys, shapes and finite values before merging into a copy for ordinary inference. Loading twice does not accumulate updates; selecting a different model leaves its base unadapted.

`ml/train_lora.py` uses the corrected masked loss, the base split seed, reserved test files, content-hashed split manifests, fresh run directories and validation-selected adapter files. `ml/evaluate.py --adapter` provides held-out evaluation with base/adapter provenance. `main.py --adapter` validates an adapter before opening the unchanged layout. No new package installation was required.

## Validation

Three-step CUDA smoke runs completed with exit code 0 for both model sizes; reports and parameter counts are in `reports/lora_validation_20261029.json`. Baseline: 140,496 trainable parameters out of 5,934,233 base parameters; large: 271,568 out of 17,951,385. All base weights stayed unchanged. Real checkpoints plus adapters were reloaded and generated valid half-second continuations.

A Windows cuDNN parametrized-RNN shutdown failure was found during the first smoke runs. Parametrized CUDA recurrent training now explicitly uses native PyTorch; merged inference retains the fused path. A subprocess regression checks process exit, rather than only observing a saved file.

Tests cover zero-adapter equivalence; gradients and decreasing fixture loss; frozen base weights; adapter-only saves; load/merge equivalence; repeated loading and switching bases; wrong-base/corrupt rejection without partial mutation; evaluator integration; CUDA training process exit; original UI source compatibility; and existing MIDI/reference/editing behavior. See `reports/final_test_results.txt` for the final suite outcome.

## Limits

Implementation and short smoke training are complete. No full style fine-tuning or perceptual quality improvement is claimed. Tiny smoke validation changes are not meaningful quality evidence; adapters remain off by default. The annotation-aware experiment is not LoRA-enabled. This recurrent implementation materializes dense updates and does not promise faster training or reduced activation memory. Resume/optimizer checkpoints are not implemented. Historical base models still reflect their original training; corrected losses apply to new runs.

See `docs/LORA.md` for runnable commands and `CLEANUP_REPORT.md` for preservation/deletion details. No repository push, retraining of original weights, or dataset download was performed.
