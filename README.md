# AI Music Co-Creator

An offline Windows desktop research prototype for iterative human–AI symbolic music composition. The user chooses a note; a conditional LSTM proposes a short MIDI continuation; the user listens, edits, or chooses the next note. The application is designed for co-creative iteration rather than autonomous full-song composition.

## At a glance

- Tkinter GUI with C–B note buttons and C, D, E, F, G, A, B keyboard shortcuts (H also triggers B natural).
- MIDI playback/export, tempo and continuation controls, temperature sampling, undo/clear, interaction log export, and a human/AI timeline.
- Stable GUI model: **LSTM 3x512 (baseline)**. **LSTM 4x768 (experimental)** remains available in the GUI selector.
- Separate AAM annotation-aware research pipeline (4 layers × 768 hidden) with source-level splits; it does not replace the GUI model and does not use audio.
- MIDI-only preprocessing and training remain available as the reproducible baseline.

## Important data and model availability

This repository intentionally does **not** include AAM archives, derived event arrays, dataset manifests containing per-source details, model weights/checkpoints, local training logs, or session outputs. Those files remain in the local project folder and are ignored by Git. Obtain the AAM files independently from the [official Zenodo dataset record](https://doi.org/10.5281/zenodo.5794629), and review the dataset terms before downloading, redistributing, or using the data. The Zenodo v1.1.0 record currently has a blank license field, so this project does not make a license or redistribution claim for the dataset. Do not upload the archives or checkpoints to this repository.

The reported CUDA training environment was Python 3.14, PyTorch 2.11.0+cu128, and an NVIDIA RTX 4060 Laptop GPU. Other PyTorch/CUDA combinations may work but were not used for the reported run. Install the PyTorch build appropriate to your operating system and hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Repository contents

```text
main.py, gui.py, composition.py, ai_generator.py, midi_player.py  Desktop app
ml/                                                               MIDI-only model and AAM experiment code
train_full.ps1                                                    MIDI-only training/evaluation launcher
train_aam_annotated.ps1                                           AAM annotation-aware launcher
reports/                                                         Compact data and evaluation summaries
experiments/aam_annotated/                                        Small configs/mappings; large derived data ignored
examples/                                                        Small generated MIDI example and metadata
models/, data/, logs/, test_outputs/                              Local-only artifacts; ignored by Git
```

## Install and launch (Windows / PowerShell)

Use Python 3.11 or newer with Tcl/Tk. From a fresh clone, create an environment, install PyTorch for the target CPU/GPU, then install this project's dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the appropriate PyTorch build from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks a project training script, allow it for the current shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The GUI can start without PyTorch and uses its rule-based fallback. To use the GUI's learned model selector, place compatible checkpoints at `models/best_model_3x512.pt` and/or `models/best_model_4x768.pt`. Checkpoints are not distributed here; the command below trains the 3x512 baseline from the local MIDI data. The annotation-aware model is not wired into the GUI.

## Dataset setup

The MIDI-only pipeline discovers `.mid` and `.midi` files recursively, including MIDI files inside ZIP archives; it does not use the AAM audio mixes. Put downloaded AAM MIDI ZIPs or extracted MIDI files under `data/raw/`:

```powershell
python -m ml.preprocessing --data ".\data\raw" --output ".\data\processed"
```

The local audited AAM collection had 22,574 valid MIDI files, 7,552,855 events, and no invalid MIDI members. The source/composition-grouped 80/10/10 split yielded 18,114 train files, 2,212 validation files, and 2,248 test files. These are local audit statistics; the raw dataset and full source manifest are not included.

## MIDI-only baseline model

The GUI baseline architecture is 3 LSTM layers, hidden size 512, embedding size 256, dropout 0.25, and context length 128. The experiment selector also offers a 4×768 MIDI-only model (4 layers, hidden size 768, embedding size 384, dropout 0.30, context 256). Both use event heads for pitch, duration, velocity, and onset timing. The baseline checkpoint is not included.

After downloading the AAM MIDI archives and preprocessing, train the baseline to the GUI's expected path:

```powershell
python -m ml.train --data ".\data\processed" --output "." `
  --epochs 30 --batch-size 16 --sequence-length 128 `
  --hidden-size 512 --embedding-size 256 --layers 3 --dropout 0.25 `
  --learning-rate 0.0005 --weight-decay 0.01 --gradient-accumulation 1 `
  --model-name best_model_3x512
```

The resulting checkpoint is `models/best_model_3x512.pt`. The held-out test split is reserved by the trainer; evaluate that checkpoint with:

```powershell
python -m ml.evaluate --data ".\data\processed" --models ".\models" `
  --checkpoint ".\models\best_model_3x512.pt" --output ".\test_outputs\baseline_3x512"
```

To reproduce the experimental MIDI-only 4×768 run instead, use `train_full.ps1` (it writes checkpoints under `models/`, logs under `logs/`, and test outputs under `test_outputs/large_lstm/`). It does not replace the GUI's selected 3×512 default. The reported runs were:

| Model | Validation loss | Test loss | Pitch accuracy | Duration MAE | Timing MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×512 MIDI-only baseline | 0.9138 | 0.9281 | 0.7253 | 0.049 s | 0.079 s |
| 4×768 MIDI-only experimental | 0.7340 | 0.7293 | 0.7830 | 0.0384 s | 0.066 s |

These symbolic metrics do not establish better perceived musical quality. Both reported runs had velocity MAE 0.0; source velocity was effectively constant in this dataset.

The GUI's monophonic AI line maps generated local phrase times onto the existing absolute composition timeline, constrains each new onset to follow the previous generated note's end, and records timing diagnostics. A same-seed regression command for both checkpoints at 0.5, 1, and 2 seconds is:

```powershell
python -m ml.regression_timing --models ".\models" --output ".\logs\timing_regression.json" --seed 41027
```

## AAM annotation-aware experiment

The experimental pipeline is under `ml/aam_annotated/`. It joins MIDI with AAM beat, tempo, chord, key, and instrument annotations at the composition level. Categorical features use learned embeddings; normalized beat/onset/tempo/duration features use a numeric projection. Onset annotations are used to validate alignment; unaligned files receive unknown annotation context. Legitimate polyphony is retained. Audio mixes are not read or used.

The experiment reuses the 80/10/10 source-level split (2,400/300/300 compositions) and has a bounded CUDA smoke test, preprocessing stats, feature mappings, and training configuration. The completed full training selected epoch 12. The compact held-out test summary is [`reports/aam_annotated_evaluation.json`](reports/aam_annotated_evaluation.json); the methodology and alignment details are in [`reports/aam_annotated_training_plan.md`](reports/aam_annotated_training_plan.md) and [`reports/aam_annotated_dataset_stats.json`](reports/aam_annotated_dataset_stats.json). The checkpoint and encoded arrays are excluded, so rerunning requires the local AAM MIDI and annotation archives plus preprocessing.

From PowerShell, preprocess and run the small smoke check first:

```powershell
python -m ml.preprocessing --data ".\data\raw" --output ".\data\processed"
python -m ml.aam_annotated.preprocess
python -m ml.aam_annotated.train --experiment ".\experiments\aam_annotated" `
  --mode smoke --seed 42 --smoke-steps 12 --smoke-batch-size 2 --require-cuda
```

After reviewing the smoke outputs, start the full training run explicitly:

```powershell
.\train_aam_annotated.ps1 -Python (Get-Command python).Source
```

The launcher requires CUDA and trains/evaluates the isolated 4×768 model. The previous completed run reported test loss 0.6830, pitch accuracy 0.7947, duration MAE 0.0363 s, and timing MAE 0.1107 s across 1,057,280 held-out targets from 300 source groups. All source events shared one velocity token, so velocity MAE 0.0 is not evidence of learned dynamics. Its timing MAE was higher than the MIDI-only 4×768 experiment; it is research material, not a proven overall improvement.

After preprocessing and training, generate an example continuation with the annotation-aware checkpoint:

```powershell
python -m ml.aam_annotated.generate --source-id 1 --member "0001_Drums.mid" `
  --target-pitch 60 --seconds 8 `
  --output ".\experiments\aam_annotated\generated\continuation.mid"
```

A small generated MIDI example is available at [`examples/aam_annotated_continuation.mid`](examples/aam_annotated_continuation.mid). Its metadata is beside it; it is illustrative and not a quality benchmark.

## Research notes and limitations

AAM consists of algorithmically composed artificial music and is intended as a symbolic/audio benchmark; results on it should not be assumed to generalize to human-composed music. See the original paper and the dataset record below. No claim is made that this prototype is production-ready or that its symbolic scores establish musical quality.

The historical dataset audit is a snapshot from before the annotation-aware experiment; the newer pipeline and its results are documented separately. Audit details and counts are in [`reports/aam_dataset_audit.md`](reports/aam_dataset_audit.md) and [`reports/aam_dataset_stats.json`](reports/aam_dataset_stats.json). The 20 MB per-source manifest is omitted from publication because it is unnecessary for reproducing the code and exposes detailed dataset inventory.

## Acknowledgements and citations

This project used the AAM MIDI and annotation files for its reported experiments. The dataset files are not redistributed here. The Zenodo record is v1.1.0 and lists a blank license field; consult the record and its linked materials for the current terms before reuse.

- Ostermann, F., & Vatolkin, I. (2022). *AAM: Artificial Audio Multitracks Dataset* (v1.1.0). Zenodo. [https://doi.org/10.5281/zenodo.5794629](https://doi.org/10.5281/zenodo.5794629)
- Ostermann, F., Vatolkin, I., & Ebeling, M. (2023). AAM: a dataset of Artificial Audio Multitracks for diverse music information retrieval tasks. *EURASIP Journal on Audio, Speech, and Music Processing*, 2023, 13. [https://doi.org/10.1186/s13636-023-00278-7](https://doi.org/10.1186/s13636-023-00278-7)

## License and reuse

No repository license file is included because the project code's ownership and licensing have not been verified. Without a `LICENSE` file, GitHub users do not receive an explicit open-source license to reuse this code. Confirm the code authorship and intended license, then add the chosen license before describing the repository as open source. The AAM dataset's terms are separate from this project's code license.
