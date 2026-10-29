# Optional LoRA GUI — implementation report

## Behavior

LoRA starts OFF on every launch. The previous environment-variable activation path is removed. `--adapter` supplies a file hint only; it does not load or activate an adapter. With LoRA OFF, the original base checkpoint and generation path are used and no adapter files are read.

The only addition to the original main layout is an **Enable LoRA** checkbox in the existing export row. When enabled, a **LoRA settings** button and a separate dark settings window appear. Original colors, note buttons, model/duration/tempo/temperature controls, playback, undo/clear, export controls and timeline remain in place. Source-comparison tests confirm the original layout/rendering methods apart from the single LoRA insertion hook.

The settings window provides a compatible-adapter dropdown, Apply, Browse, Refresh and Training/evaluation help. It shows selected model, adapter path, rank, alpha, epoch and smoke-test status. The dedicated adapter directory is the existing `runs/lora/`; Browse supports compatible adapters elsewhere. Training continues to be a separate explicit CLI action, with commands available in the help window; no training starts from enabling LoRA.

## How to use

1. Start `python -B main.py` and choose the existing baseline or experimental LSTM.
2. Check **Enable LoRA**.
3. Select a compatible adapter (or Browse), inspect its configuration, and click **Apply adapter**.
4. Generate using the original note-button workflow. The model label reads **LoRA active**.
5. Uncheck **Enable LoRA**, or close the settings window, to reload the pristine base checkpoint.

A model change clears the active selection and filters adapters for the newly selected checkpoint. Until a valid adapter is applied, generation is blocked with a clear message. Missing, corrupt, non-finite and incompatible adapters are reported; they do not trigger silent mode switching. An unsuccessful Browse explicitly states when a previous active selection is retained. An adapted inference failure reports an error rather than silently using symbolic fallback. Controls are disabled during generation, and typing in the settings window does not trigger note shortcuts.

Evaluation remains the existing CLI: omit `--adapter` for original-model evaluation; supply the selected adapter path for adapted evaluation. GUI help shows the matching command. Neither model architecture nor dataset processing was changed. Existing adapter format, checkpoint format and saved-project formats are retained. Incompatible explicit adapter/base pairings are now rejected rather than silently evaluated as a plain base model. The optional extended `--studio` UI remains separate; interactive LoRA controls are in the default original UI.

## Files changed or added

| File | Change |
| --- | --- |
| `gui.py` | Single layout hook, selection/reload wiring, generation readiness/error guards and busy-control handling. |
| `lora_controls.py` (new) | Opt-in checkbox, dark settings window, adapter discovery/selection, validation feedback and optional training/evaluation help. |
| `main.py` | Startup remains OFF; CLI adapter argument becomes a hint, not activation. |
| `ml/inference.py` | Removes environment activation, adds explicit adapter reset/reload, rejects incompatible adapter/base combinations. |
| `tests/test_lora_gui.py` (new) | Real Tk GUI transitions/errors, launch defaults, real-model inference/evaluation OFF/ON checks. |
| `tests/test_classic_gui.py` | Preserves upstream layout comparison while allowing the single intended LoRA hook. |
| `tests/test_lora.py` | Updates compatibility regression for explicit rejection and base restoration. |
| `README.md`, `docs/LORA.md` | Current enable/disable, training, evaluation and startup instructions. |
| `IMPLEMENTATION_REPORT.md` | Historical report linked to this current GUI update. |
| `reports/lora_gui_report.md` | This report. |
| `reports/lora_gui_tests.txt` | Complete final test output. |
| `reports/lora_gui_gpu_tests.json` | CUDA smoke-test results for both existing models. |
| `reports/lora_gui_checkpoint_checks.json` | Hash-preservation evidence for all eight original checkpoints. |

Existing `ml/lora.py`, `ml/train_lora.py`, `ml/evaluate.py`, model architecture and preprocessing were reused without modification in this task. No workspace cleanup, checkpoint replacement or original-model retraining was performed.

## Tests and limits

- Full existing and new suite: **52 passed**, 12.723 seconds, exit code 0 (`python -B -m unittest discover -s tests -v`). No tests skipped on this machine.
- GUI tests use real Tk widgets and small checkpoint/adapter fixtures; playback and dialogs are mocked where needed. They check OFF startup even with environment/hint, both model selections, active/inactive transitions, exact base-weight restoration, invalid/missing files, model switching, guarded generation and ordinary append/play behavior.
- Real 3x512 and 4x768 checkpoints were tested with their existing trained smoke adapters on CUDA. OFF → ON → OFF restored identical seeded continuation results. Evaluation of both real models explicitly exercised adapter-free and adapted paths using small temporary datasets.
- Existing adapter save/load, zero-initialization, frozen-weight, merge-equivalence, malformed-tensor and CUDA subprocess-exit tests pass.
- Separate existing-trainer CUDA smoke checks: **3 adapter optimizer steps per model**, both exited 0 with frozen base weights unchanged. Outputs were temporary test fixtures and removed automatically; no production training run or adapter activation occurred.
- All **8 original checkpoint SHA-256 hashes** match the protected inventory.

Implementation success does not demonstrate better musical quality. The available local adapters are short smoke-test artifacts and are labelled accordingly; curated fine-tuning, held-out comparison and listening evaluation are still required for a quality claim.
