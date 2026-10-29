# LoRA for the MIDI LSTM

LoRA learns `W + (alpha / rank) * B @ A`, leaving the original model weights frozen. A uses Kaiming initialization; B starts at zero, so enabling an untrained adapter initially preserves predictions. This is a local PyTorch implementation of the [LoRA method](https://arxiv.org/abs/2106.09685); no PEFT/Transformers dependency or downloaded model is required.

## Scope

Supported bases: the existing MIDI-only 3x512 and 4x768 MusicLSTM checkpoints. All LSTM input/recurrent matrices, the tempo projection and four output heads receive adapters. Biases and embeddings stay frozen. The separate annotation-aware model is not supported. This does not add MP3 understanding or automatically teach a style.

At rank 8 the baseline has 140,496 trainable parameters versus 5,934,233 original parameters (2.37%); the large model has 271,568 versus 17,951,385 (1.51%). Optimizer state and adapter storage are smaller than full fine-tuning. This implementation materializes dense weight updates; faster training or lower total activation memory is not guaranteed.

The parametrized recurrent training path uses native PyTorch CUDA operations. cuDNN recurrent training with these recomputed weights caused a Windows process shutdown failure during validation, so it is explicitly bypassed for parametrized CUDA LSTMs. Merged inference retains the normal fused LSTM path. A subprocess regression verifies clean CUDA shutdown.

## Train

Use the installed PyTorch and project requirements. Commands run from the project root:

```powershell
# Three optimizer steps; confirms mechanics, not musical improvement:
python -B -m ml.train_lora --smoke --output runs/lora/smoke_new

# A deliberate fine-tuning run (not launched automatically):
python -B -m ml.train_lora --base models/best_model_3x512.pt `
  --data data/processed --output runs/lora/my_style `
  --rank 8 --alpha 16 --epochs 5 --batch-size 8 --learning-rate 0.0001

# Larger MIDI-only base:
python -B -m ml.train_lora --base models/best_model_4x768.pt `
  --data data/processed --output runs/lora/large_style --batch-size 2
```

`--data` takes preprocessed JSONL, not MP3/WAV or raw MIDI. For a style dataset, use the existing MIDI preprocessor into a new folder; curate suitable examples and keep related versions within one composition group. The trainer preserves the base checkpoint's split seed, reserves the test partition, and records filename/content hashes in `split_manifest.json`. With an entirely new dataset this creates a new split; it cannot prove absence of overlap with an unknown base pretraining corpus. Losses and validation metrics exclude padding.

Outputs are adapter-only `adapter_epoch_NNN.pt` files whenever validation loss improves, plus `report.json` and `split_manifest.json`. `report.json` identifies `best_adapter` and compares validation loss against the zero-adapter baseline using the same backend. A run can fail to improve the base. Output folders must be new; existing runs are never overwritten. Resume/optimizer checkpointing is not implemented.

## Enable or disable LoRA in the GUI

Start normally with `python -B main.py`. **LoRA always starts OFF**, even if the old `MUSIC_LORA_ADAPTER` environment variable is set. The original base checkpoint and note-click workflow are used; adapter files are not inspected or loaded while OFF.

1. Select the baseline 3x512 or experimental 4x768 in the existing model selector.
2. Check **Enable LoRA**, beside the existing export buttons. This opens a separate dark settings window; the original main-window controls/layout stay in place.
3. Select a compatible adapter from `runs/lora/` or click **Browse…**. Rank, alpha, epoch, path and smoke-test status are shown.
4. Click **Apply adapter**. The model label reports **LoRA active** and new generations use that adapter.
5. Uncheck **Enable LoRA**, or close the settings window, to reload the original checkpoint with no adapter modifications.

Until a valid adapter is applied, generation is paused with an explicit message. Changing models or choosing a different listed adapter clears the active selection; select/apply a compatible adapter for the selected model. A failed Browse leaves an existing active selection unchanged and says so explicitly. Missing, corrupt and incompatible files never cause a silent fallback. If adapted inference itself fails, the GUI reports the error instead of silently generating a symbolic substitute.

Adapter compatibility is verified against the exact base-file SHA-256, target names, shapes and finite tensor values before merging. Original checkpoint files are never overwritten. Turning LoRA OFF reloads pristine weights rather than subtracting deltas. Neither the enabled state nor the adapter selection is restored from saved compositions.

The optional launch argument is now only a file hint:

```powershell
python -B main.py --adapter runs/lora/my_style/adapter_epoch_001.pt
```

It does **not** enable or load the adapter on startup. Check Enable LoRA, select it and Apply. The separate `--studio` interface remains available without LoRA controls; use the default GUI for interactive LoRA selection. Combining `--studio` with `--adapter` reports that distinction.

Training remains an explicit CLI operation using the unchanged trainer above. **Training / evaluation help** in the settings window shows commands for the selected base/adapter. The GUI never launches training automatically. Existing checkpoints, adapter files and saved projects remain compatible; invalid adapter/base pairings are now rejected rather than silently using the base model.

## Evaluate

```powershell
python -B -m ml.evaluate --data data/processed `
  --checkpoint models/best_model_3x512.pt --output runs/lora/eval_base
python -B -m ml.evaluate --data data/processed `
  --checkpoint models/best_model_3x512.pt `
  --adapter runs/lora/my_style/adapter_epoch_001.pt --output runs/lora/eval_adapter
python -B -m unittest discover -s tests -v
```

Evaluation is a separate CLI operation: omit `--adapter` for OFF, or supply the same selected adapter path for ON. The GUI help displays the matching command. Evaluation records both checkpoint and adapter hashes. Use the same data/split for comparison, and do not tune on test results. Teacher-forced pitch accuracy is not a listening assessment. Compare generated continuations using identical prompts and seeds, listen blind, and assess rhythm, repetition, phrasing and control adherence.

Only three-step smoke runs were executed during implementation. Their adapters are under `runs/lora/verified_baseline_20261029` and `runs/lora/verified_large_20261029`, disabled by default. No perceptual improvement is claimed.
