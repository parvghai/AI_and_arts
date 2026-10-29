# AAM annotation-aware LSTM experiment

## Status

The preprocessing pipeline, encoded data, CUDA smoke test, full training, and held-out evaluation have completed. Smoke training verified tensor shapes, CUDA forward/backward passes, decreasing training loss, and checkpoint save/reload. The full model trained for 18 epochs on the NVIDIA GeForce RTX 4060 Laptop GPU and early-stopped; its best validation checkpoint is epoch 12 (validation loss 0.6854). The PowerShell launcher displayed a final error because the Python process returned nonzero after writing the checkpoint and completed evaluation report; both artifacts were inspected and verified.

## Data and source splits

- Inputs: AAM MIDI files plus the corresponding AAM `beatinfo`, `onsets`, and `segments` annotations. Audio-mix ZIPs are not opened or used.
- 22,574 MIDI files validated; 0 invalid; 7,552,855 note events.
- Composition/source-level assignments reuse the MIDI-only baseline's `split_files` assignments with seed 42. There are 3,000 source groups: train 2,400, validation 300, test 300; source-ID split leakage is 0.
- At sequence length 256: train 18,114 files / 6,061,647 events / 34,039 windows; validation 2,212 / 766,783 / 4,235; test 2,248 / 724,425 / 4,130. Test remains held out.
- Chord, key, and instrument mappings are fitted only on train sources. Unknown/missing annotation values map to ID 0.
- Full encoded arrays occupy 311,537,754 bytes (about 297 MiB) on disk and are read through memory maps.

## Alignment and timing decisions

The MIDI event parser stores the first event relative to time zero, while AAM onset annotations use composition time. For each MIDI member the pipeline compares the ordered MIDI note pitches and onsets to the relevant AAM onset attack stream. A constant source offset is recovered as the median annotation-onset minus relative-MIDI-onset, and the maximum residual must be at most 0.06 seconds. Onset annotation pitches are used only for this alignment check and are never model input or prediction targets.

21,867 MIDI members aligned; 707 did not. All MIDI events remain in the dataset. For unaligned members, annotation-derived context is unknown/default rather than guessed. This prevents uncertain beat/chord/key/tempo values from being attached to notes. The alignment counts are recorded per split and per member in `dataset_manifest.json`.

Each event retains its absolute onset after the validated source offset, duration, and delta time. Beat phase is interpolated between AAM beat timestamps. Tempo and key context come from the latest segment at or before the event; chord context comes from the latest beat annotation at or before the event. Timing inputs include log-normalized inter-onset gap in beats, beat phase sine/cosine, within-beat fraction, normalized tempo, normalized composition progress, and normalized duration. The timing prediction target is the normalized inter-onset gap. Malformed MIDI/annotation rows are rejected; valid simultaneous notes and overlapping durations are preserved. The dataset recorded 16,530,269 overlap pairs across splits; exact per-split counts are in `aam_annotated_dataset_stats.json` (overlap is diagnostic, not a filtering rule).

## Event representation

For each event the categorical vector is `[pitch, duration, velocity, instrument, chord, key]`. Pitch has 128 classes, quantized duration 16, quantized velocity 8, instrument 33 IDs including unknown, chord 27 including unknown, and key 25 including unknown. Categorical values use separate learned embeddings of dimension 384; the seven normalized continuous timing/context values pass through a learned linear projection from 7 to 384. These representations are summed into the 384-dimensional LSTM input. No annotation value is converted to an arbitrary ordinal numeric feature.

The training window is 256 events. Batch tensor shapes are categorical input `[B,256,6]`, numeric input `[B,256,7]`, categorical targets `[B,256,3]` (pitch/duration/velocity), and timing target `[B,256]`. The model also conditions each window on the final pitch of its target window, following the existing project training interface.

## Model and resource estimate

The isolated experimental model is a four-layer LSTM, hidden size 768, embedding size 384, dropout 0.30. With the observed feature vocabulary sizes it has **17,980,185 trainable parameters** (about 17.97M). Float32 parameters, gradients, and two Adam moments require about 274.4 MiB before activations and CUDA workspace. Encoded data is memory-mapped, so it is not all copied into RAM/GPU memory. The smoke test used batch size 2, and observed peak CUDA allocation was 534,447,104 bytes. Full-training peak GPU memory has not been measured; plan within the RTX 4060's 8 GiB and monitor the first full-run batch before committing to a long run.

## Smoke test command

From the project root in PowerShell, after confirming the CUDA Python environment works, run:

```powershell
python -m ml.aam_annotated.train --experiment .\experiments\aam_annotated --mode smoke --seed 42 --smoke-steps 12 --smoke-batch-size 2 --require-cuda
```

This bounded smoke mode checked tensor shapes and finite timing values, verified CUDA, ran forward and backward passes on a fixed real-data batch, confirmed the training-batch loss decreased, and saved/reloaded a checkpoint. The final validation loss increased during the tiny overfit check, so the smoke test confirms implementation and checkpoint mechanics, not generalization. Full training ran separately and selected its checkpoint using validation data; the held-out test split was evaluated once after selection. Results and checkpoints are under `experiments/aam_annotated/`.

## Reproducibility and preservation

The new code is isolated in `ml/aam_annotated/`; generated data, mappings, manifest, and preprocessing configuration are under `experiments/aam_annotated/`. The external summary is `reports/aam_annotated_dataset_stats.json`. Existing MIDI-only preprocessing, GUI, `models/`, baseline outputs, and both 3x512 and 4x768 checkpoints were not changed by this experiment.

## Full training result

The best epoch 12 checkpoint achieved held-out test loss 0.6830, pitch loss 0.5437, pitch accuracy 0.7947, duration MAE 0.0363 seconds, velocity MAE 0.0, and timing MAE 0.1107 seconds on 1,057,280 target events from the 300 held-out source groups. Training ran 18 of the 30 maximum epochs before six validation epochs without improvement. The checkpoint is at `experiments/aam_annotated/models/aam_annotated_best.pt`; complete epoch history and evaluation are in `experiments/aam_annotated/reports/training_summary.json` and `evaluation_report.json`. These symbolic metrics do not establish musical quality; listen to generated examples.

The zero velocity error is not an expressive-velocity result: all 7,552,855 events across train, validation, and test encode to velocity token 3. The model therefore had no velocity variation to learn. The timing MAE is 0.1107 seconds; the existing MIDI-only 4×768 report recorded 0.066 seconds. The annotation-aware run improves some symbolic metrics but has not demonstrated better timing or more musical improvisation.

To repeat full training, run `train_aam_annotated.ps1 -Python (Get-Command python).Source`. To generate a continuation after training, use the command documented in the project README.
