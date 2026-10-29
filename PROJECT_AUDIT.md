# AI Music Co-Creator — Project Audit

Audit date: 28 October 2026 (Europe/Berlin). Scope: the current local project, its source, documentation, configurations, and available local artifacts. This is an engineering and research assessment, not a listening study or an independent replication of training.

Only this report was created. No existing source, tests, configuration, datasets, checkpoints, or outputs were intentionally changed. No packages were installed, no network requests were made, and no training was run. Diagnostics used `python -B` to avoid bytecode writes. File-producing diagnostic paths were inspected first; selected exports and the existing regression runner had their output writes redirected to memory.

Evidence labels used throughout:

- **Verified:** reproduced in a bounded diagnostic during this audit.
- **Confirmed by source:** directly established by the implementation, without exercising the complete user workflow.
- **Recorded:** an existing report, configuration, or checkpoint contains this result; training/full evaluation was not rerun.
- **Risk:** a plausible failure or limitation whose incidence/impact still needs measurement.
- **Missing:** no implementation or supporting evidence was found.

Source references use paths relative to this report and the original source line numbers. No source lines were changed during the audit.

## 1. Executive summary

**Verdict: a functioning local research prototype with useful interaction mechanics, but not yet a reliable benchmark or a release-ready music tool.** The baseline and experimental GUI checkpoints load, the existing six-case timing regression passes, the deterministic fallback passes 63 combinations, and GUI MIDI export works in a memory-only round trip. The annotation-aware checkpoint also loads, produces finite outputs, and generates a bounded continuation.

The largest problem is evaluation validity. Both datasets repeat the last event to pad short windows, and both training and evaluation count these repetitions as genuine targets. On the available test inventory, **15.98% of the 128-step baseline targets and 31.70% of the 256-step targets are artificial padding**. Padding repeats the same pitch supplied as the endpoint condition, creating an especially easy shortcut. Scores do not isolate real-note performance, and comparisons across context lengths have different padding burdens. This is not evidence of composition-ID train/test leakage: the current source groups are disjoint, which was independently verified.

Other material findings are that the live MIDI-only model receives varying BPM despite training always receiving 100 BPM; phrase duration limits constrain onsets but not final note ends; the no-PyTorch model selector crashes; and some parser behavior is unsafe for general multitrack MIDI. The annotation experiment is substantial implementation work, but it changes multiple factors simultaneously, uses a different timing target, retains the padding problem, and has deployment-time conditioning mismatches. It has **not demonstrated better perceived musical quality or better co-creation**.

The strongest next step is to repair and regression-test padding masks, then reevaluate the existing checkpoints on real targets before training anything new. Preserve the current 3×512 checkpoint as the reference model while doing this.

### Subjective scorecard

These scores are reviewer judgments on a 1–10 scale, not measurements, calibrated against a dependable and independently reproducible research prototype.

| Dimension | Score | Evidence and uncertainty |
|---|---:|---|
| Architecture | 6 | Composition, playback, inference, and experimental pipeline are separated. GUI orchestration, repeated exporters/encoders, and shared mutable state limit maintainability. |
| AI/ML implementation | 5 | Working conditional LSTMs, validation selection, grouped splits, and memory-mapped annotation data. Unmasked padding and inconsistent conditioning undermine interpretation. |
| Musical quality | 4, provisional | Some generated phrases satisfy structural constraints; repeated motifs and limited control are observable. No listening study was performed, so this is a low-confidence assessment of demonstrated capability, not an acoustic quality rating. |
| Usability | 5 | Immediate note choices, short turns, playback, undo, and visible authorship. No note editing, session reopening, or candidate comparison; live GUI/audio ergonomics remain untested here. |
| Testing | 3 | A useful executable timing regression exists and passes, but no conventional test files or CI were found, and that regression misses duration overruns. |
| Reproducibility | 4 | Useful scripts/configurations and substantial local artifacts; no pinned environment, no usable commit identity in this checkout, artifact provenance gaps, and documented command/config differences. |
| Research potential | 7 | Strong platform for studying human control and autonomy duration; contribution and novelty remain hypotheses pending controlled studies. |
| Overall maturity | 4 | Suitable for supervised local demonstrations and exploratory work after known limitations are explained; not ready for strong comparative claims or general distribution. |

## 2. Project overview and architecture

### Purpose and intended users

The project supports iterative symbolic composition: a person selects a natural note, hears a short machine continuation, and decides what to do next. Its likely users are computational-creativity researchers, students, and musicians exploring a simple interaction prototype. It is not a full sequencer, audio generator, or complete-song composer.

The selected note has two roles: it is played as a human cue and supplied to the model as a desired future endpoint pitch. The incoming cue is not appended to the model context before generation; prior composition events form the context, while the new note enters through the target embedding (`gui.py:203–224`, `ml/inference.py:156–173`, `composition.py:37–53`). This distinction matters when describing the user's control.

### Module responsibilities and execution paths

| Component | Responsibility and evidence |
|---|---|
| `main.py:1–7` | Constructs `MusicApp` and starts Tk's event loop. |
| `gui.py:33–171` | Builds the Windows-oriented Tk interface; reads note, duration, BPM, and temperature; starts a daemon generation worker. |
| `gui.py:193–295` | Chooses learned inference or fallback, measures latency, sanitizes timing, marshals completion to Tk, appends history, saves a log, and starts playback. |
| `composition.py:7–77` | Stores events and interactions; computes absolute composition end; appends a human cue; supports whole-interaction undo/clear and JSON serialization. |
| `ai_generator.py:8–40` | Deterministic C-major continuation using the previous pitch, target, a repeating motif, and alternating note-length weights. |
| `ml/inference.py:11–104` | Stateful autoregression with temperature, top-k/top-p sampling, repetition penalties, leap limits, timing clamps, and a 256-event ceiling. |
| `ml/inference.py:107–203` | Loads a selected checkpoint, checks its architecture/conditioning, converts composition events to model input, and maps generated local times to the absolute timeline. |
| `midi_player.py:13–143` | Windows `winmm` note scheduling; `winsound.Beep` fallback; unavailable-audio mode elsewhere. |
| `ml/preprocessing.py:75–266` | Custom MIDI parser; recursive MIDI/ZIP discovery; 50 ms quantization; JSONL output. |
| `ml/dataset.py`, `ml/model.py`, `ml/train.py`, `ml/evaluate.py` | Source grouping, fixed windows, baseline LSTM, training/validation, and held-out teacher-forced evaluation. |
| `ml/aam_annotated/` | Isolated annotation alignment, train-only vocabulary fitting, binary feature storage, model, training, evaluation, and CLI generation. Not connected to the GUI. |
| `ml/regression_timing.py`, `ml/compare_models.py` | Timing regression and small matched-seed generation comparisons. Both normally write artifacts. |
| `ml/audit_aam_dataset.py` | Historical archive/data inventory generator. Despite its “read-only” description, its entry point writes reports; it was inspected, not run. |

Normal interaction flow:

```text
Note button/key + settings
  -> GUI snapshots prior history and disables generation controls
  -> worker invokes selected LSTM or deterministic fallback
  -> local generated onsets become absolute times after the human cue
  -> timing validation/sanitization
  -> Tk callback appends interaction and redraws authorship timeline
  -> local JSON snapshot + Windows MIDI/tone playback
  -> next note, undo, clear, replay, or export
```

The human cue starts at the previous composition end and lasts at most 0.22 seconds. The AI begins at or after the cue end. Generation retains LSTM hidden state within a phrase after one context prefill; a new interaction rebuilds context from the latest events. Source training data can contain polyphony, but GUI generation deliberately imposes a monophonic AI line.

### Data and model design

The baseline represents each event with pitch, duration, velocity, and onset-to-onset gap. Pitch has 128 classes, duration 16, velocity 8, and timing input 16. Separate embeddings are summed with target-pitch and tempo embeddings/projections. A unidirectional LSTM predicts three classification heads plus a scalar timing head (`ml/model.py:8–46`). The timing target is a bucket index divided by 7.5, rather than a continuous second or beat duration.

The available baseline has 3 layers, hidden size 512, embedding size 256, 128-event context, and **5,934,233 parameters**. The larger MIDI-only model has 4 layers, hidden size 768, embedding size 384, 256-event context, and **17,951,385 parameters**. The annotation model has **17,980,185 parameters**. Approximate float32 parameter storage is 22.6, 68.5, and 68.6 MiB respectively; this excludes activations, runtime workspace, optimizer state, and Python overhead. The GUI loads entire training checkpoints, including optimizer state where present, before retaining the model.

Training uses AdamW, gradient clipping, validation-based checkpoint selection, a plateau scheduler, and early stopping. The MIDI-only trainer supports gradient accumulation and CUDA autocast; the annotation trainer has a separate bounded overfit/smoke mode and full training mode. No training was rerun for this audit.

The annotation features are six categorical fields (pitch, duration, velocity, instrument, chord, key) and seven numeric fields (beat-normalized gap, phase sine/cosine, within-beat position, tempo, composition progress, duration). Instrument/chord/key mappings are fitted only on training sources; unknown IDs are zero. MIDI/onset alignment uses pitch-order agreement and a constant timing offset with a 60 ms residual tolerance (`ml/aam_annotated/preprocess.py:110–196`). Source polyphony is retained in preprocessing. Generation holds the final chord/key/instrument context constant and enforces monophony.

### Local assets versus a source-only checkout

The local folder contains much more than the distributable source described in the README:

- 22,574 processed JSONL files, plus metadata; about 568 MiB in `data/processed`.
- Three MIDI ZIPs, three annotation ZIPs, three large audio-mix ZIPs, and `info.zip`; about 41.2 GiB under `data/raw`. Audio is not needed for model inference or symbolic training.
- Both named GUI checkpoints, legacy/latest checkpoints, and annotation checkpoints.
- About 517 MiB under the annotation experiment, including arrays and artifacts; 1,808 files/about 226.5 MiB under `logs` at inventory time.

These assets are excluded by `.gitignore`. Their local presence is not evidence that another developer receives them with the source. This checkout has a `.git` directory but `git ls-files` returned zero tracked files and no verified `HEAD` was available. Consequently, this report identifies a filesystem snapshot, not an auditable release commit. Git ownership checks were handled with a command-scoped `safe.directory` option; no global Git configuration was changed.

## 3. Feature implementation and verification status

| Feature | Status | What the evidence does and does not establish |
|---|---|---|
| Natural-note buttons and C–B/H keyboard shortcuts | Source confirmed | `gui.py:83–91,173–179`; no live keyboard/UI session exercised. Only one octave of natural-note input is exposed. |
| Background generation and fallback on inference errors | Source confirmed; components verified | Worker exists; real inference and fallback work separately. Shutdown/race behavior not stress-tested. |
| Exact named GUI checkpoints | Verified | Both named checkpoints passed loading with restricted deserialization in the audit and matched the expected configurations. |
| Learned continuation | Verified, bounded | Existing regression exercised both models at 0.5/1/2 seconds. This establishes output validity for those fixtures, not musical quality. |
| Fallback continuation | Verified | 63 note/BPM/duration combinations were deterministic, reached the requested final note, and satisfied current timeline validation. |
| Human/AI timeline | Source confirmed | Events and authorship labels are displayed. No screenshots, clipping measurements, or accessibility testing performed. |
| Undo and clear | Partly verified | Whole-interaction undo verified on normal sequential fallback compositions; clear is a straightforward state reset. There is no per-note edit interface. |
| Playback, pause, resume, stop | Source inspected only | No audio device was opened and no sound was played. Device/timbre/pause behavior remains unverified. |
| MIDI export | Verified in memory | GUI exporter wrote a four-note composition to bytes; Mido reopened it and recovered duration 1.17 s, matching composition end. No file/dialog workflow exercised. |
| Interaction logs | Source confirmed | Rich metadata and local snapshots exist. Save failures, session recovery, and user-facing error persistence were not exercised. |
| Session reopening and note editing | Missing | There is export plus undo/clear, but no composition import/reopen, drag/edit, redo, or alternative-candidate workflow. “Edit” in the README is broader than the implemented controls. |
| Source-grouped split | Verified | 2,400/300/300 composition IDs; zero pairwise ID intersections. Annotation index assignments matched the current baseline assignments. |
| Train-only annotation vocabulary | Source confirmed | `_annotation_vocab` visits training IDs. A full independent rebuild was not performed. |
| Annotation checkpoint forward pass | Verified | Epoch-12 checkpoint loaded; one real test window produced finite heads with expected tensor shapes. |
| Annotation generation | Verified, bounded | Source 1 / Drums, 0.5 s request, at most four events; one note generated and MIDI reopened in memory. |
| Full corpus validity and completed training | Recorded | Existing reports say 22,574 valid MIDI files, 7,552,855 events, and completed runs. All archives were not reparsed and training was not independently reproduced. |
| Fully disconnected operation | Supported by source, not end-to-end certified | No application network calls found; local inference/export worked without network operations in the audit. Network-disabled clean-machine installation and dependency behavior were not tested. |

## 4. Strengths and weaknesses

### Five major strengths

1. **A clear, low-friction co-creative loop.** Human decisions repeatedly constrain short AI turns; users can replay, undo a turn, and choose the next cue. The timeline distinguishes authorship. These are meaningful agency mechanisms, even without granular editing (`gui.py:150–171,280–295,322–395`).
2. **Useful separation of responsibilities.** Composition is independent of Tk and Torch; generation and playback have adapters; the annotation experiment is isolated. This made noninteractive testing possible without rewriting the application (`composition.py`, `ai_generator.py`, `midi_player.py`, `ml/aam_annotated/`).
3. **Explicit timing conventions and instrumentation.** Onset-gap semantics, absolute continuation boundaries, off-before-on MIDI ordering, validation, and recorded timing diagnostics directly address a common source of symbolic-music errors. Six controlled regression cases passed (`ml/inference.py:156–203`, `composition.py:80–165`, `gui.py:433–443`).
4. **Real safeguards against obvious experimental leakage.** Current AAM composition groups are disjoint across splits, annotation vocabularies are fitted on train sources, and validation selects checkpoints. This is much stronger than random overlapping-window splits (`ml/dataset.py:18–34`, `ml/aam_annotated/preprocess.py:80–130`). It does not remove padding or oracle-conditioning concerns.
5. **Practical local experimentation with candid limitations.** Both checkpoint paths work locally; the annotation pipeline uses memory maps, preserves source overlaps, and records alignment coverage. The README explicitly cautions against inferring perceptual quality from symbolic metrics and against treating constant-velocity accuracy as expressive skill. Local execution avoids a required service account or per-request API fee.

### Other positive decisions

- Recurrent inference prefills context once and then consumes only each new event; this is efficient for the short interactive use case (`ml/inference.py:52–103`).
- Sampling includes configurable temperature and filtering, finite-logit repair, repetition controls, a pitch-leap clamp, and bounded event counts. Their artistic effects need evaluation, but basic output protection is deliberate.
- Archive MIDI discovery avoids extraction and refuses MIDI ZIP members over 64 MiB (`ml/preprocessing.py:195–218`). Annotation preprocessing does not offer the same member-size safeguard.
- Annotated full-training checkpoints use a temporary file followed by replacement (`ml/aam_annotated/train.py:214–226`), reducing exposure to partially written checkpoints.
- Historical versus current experiments are clearly separated in documentation, and missing public weights/data are disclosed.

### Five largest weaknesses

1. **Reported evaluation includes large amounts of synthetic padding**, with a direct endpoint shortcut and an unequal burden across model contexts.
2. **Training conditions and live controls disagree:** constant training BPM, polyphonic training versus monophonic generation, long event-count endpoints versus short second-based turns, and annotation features that evolve differently in generation.
3. **User control is narrow and duration semantics are unreliable:** seven cue pitches, no note editor, no candidate audition/acceptance, no saved-session reopening, and final notes can exceed the selected duration.
4. **Reliability testing is too sparse:** several reproduced edge failures escaped the existing regression, while live playback, clean installation, and GUI shutdown behavior lack demonstrated coverage.
5. **Research evidence does not establish the claimed benefits users would care about:** no controlled annotation ablations, listening study, co-creation usability study, or evidence of generalization beyond algorithmically composed AAM material.

Maintainability is also weakened by compact multi-statement lines, repeated pitch maps, several separate MIDI writers, duplicated encoding/decoding rules, implicit configuration contracts, and repeated logging of historical state. These warrant incremental consolidation rather than a wholesale architectural replacement.

## 5. Confirmed defects and technical risks

### C1 — Unmasked padding contaminates training and evaluation (P0)

**Verified and source confirmed.** `ml/dataset.py:80–97` and `ml/aam_annotated/dataset.py:43–59` repeat the last event to fill a window. Neither returns a validity mask. Losses and metrics consume every position (`ml/train.py:120–123`, `ml/evaluate.py:44–56`, `ml/aam_annotated/train.py:19–28`, `ml/aam_annotated/evaluate.py:69–89`).

An in-memory three-event fixture with pitches 60, 62, 64 and sequence length 8 produced targets `[62,64,64,64,64,64,64,64]`, endpoint condition 64, and eight scored positions despite only two real transitions.

Counts computed from the local per-file annotation index and the exact windowing rule:

| Split / context | Real target positions | Artificial target positions | Total scored positions | Artificial share |
|---|---:|---:|---:|---:|
| Train / 128 | 6,043,452 | 1,109,060 | 7,152,512 | 15.51% |
| Train / 256 | 6,043,503 | 2,670,481 | 8,713,984 | 30.65% |
| Validation / 128 | 764,566 | 135,274 | 899,840 | 15.03% |
| Validation / 256 | 764,571 | 319,589 | 1,084,160 | 29.48% |
| Test / 128 | 722,172 | 137,348 | 859,520 | 15.98% |
| Test / 256 | 722,173 | 335,107 | 1,057,280 | 31.70% |

The raw baseline test JSONL row count was independently checked: 724,425. Real target counts differ from rows because the first event is context and some short tail windows are discarded. The annotation index supplies the per-file counts used above; this audit did not rebuild the complete encoded corpus.

**Impact:** the existing scores mix real transitions with synthetic repeated notes. Inflation is strongly plausible, but its magnitude and whether model ranking changes require masked evaluation; the table alone does not quantify accuracy inflation. Larger models also changed context length, so the padding burden is a comparison confound. This issue does not establish source-ID leakage.

### C2 — BPM conditioning is untrained as a variable control (P1)

**Confirmed by source and a dataset fixture.** Every baseline training window returns `tempo=100.0` (`ml/dataset.py:96`). The GUI exposes 60–160 BPM and inference projects the selected value into the model (`gui.py:113–119`, `ml/inference.py:49–55`). The training data can contain real tempo changes, but their BPM metadata is discarded and event durations are encoded in seconds.

The learned system therefore has no training evidence for the advertised conditioning response at other BPMs. Changing BPM does not rescale existing composition events; export adjusts ticks and writes a matching tempo, preserving their duration in seconds. This is not a demonstrated “global speed” control. The rule-based generator does use BPM for phrase construction. The cleanest short-term fix is to explain/fix the control's contract, then evaluate a beat-based representation before promising learned tempo control.

### C3 — Requested duration does not bound the phrase's final note end (P1)

**Verified.** Both learned generators stop based on a next onset, without clipping the duration of the final note (`ml/inference.py:80–96`, `ml/aam_annotated/generate.py:149–165`). The GUI calls this an autonomy duration in metadata.

With the existing regression fixture, target A, seed 41027, and 100 BPM:

| Model | Requested seconds | Actual AI span after boundary | Notes | Timeline validator |
|---|---:|---:|---:|---|
| 3×512 | 0.5 | 0.600 | 3 | Pass |
| 3×512 | 1.0 | 1.050 | 6 | Pass |
| 3×512 | 2.0 | 2.100 | 13 | Pass |
| 4×768 | 0.5 | 0.425 | 1 | Pass |
| 4×768 | 1.0 | 0.975 | 3 | Pass |
| 4×768 | 2.0 | 2.075 | 7 | Pass |

The bounded annotation diagnostic also produced a 0.509 s span for a 0.5 s request. Its included eight-second example lasts about 8.273 s. The regression verifies order/overlap/boundary, not the duration ceiling. Define whether the control means onset horizon or total audible phrase duration; current labels imply the latter more strongly than the code guarantees.

### C4 — Model selection crashes without the optional ML dependency (P1)

**Verified with an import-failure simulation.** The fallback `LSTMGenerator` in `gui.py:16–24` has no `load_error` attribute. `_select_model` reads `self.ml_generator.load_error` directly at line 188. When `ml.inference` import is unavailable, selecting a model raises `AttributeError`. Initial fallback use can still work; this is specifically a model-selection failure, not proof that the whole no-Torch GUI cannot start.

### C5 — General-MIDI parser limitations can change note durations (P1)

**Verified synthetic cases; corpus incidence unknown.** The parser comment says unmatched notes close at their validated track end, but implementation closes all remaining notes at the final merged timestamp (`ml/preprocessing.py:158–182`). A note without a note-off whose track ends at 0.5 s was extended to 2.0 s when another track ended at 2.0 s.

Pairing is by channel/pitch across tracks, with FIFO matching. A two-track, same-channel/pitch example produced durations 0.5/0.75 s rather than track-associated 1.0/0.25 s. Such shared-channel overlapping notes require an explicit policy; this is not proof that all cross-track pairing is invalid under MIDI channel semantics. Preserve track/channel provenance and detect ambiguity instead of silently claiming full fidelity.

The parser also accepts type-2 MIDI and merges its independent tracks onto one timeline. A synthetic type-2 case confirmed that behavior. Existing AAM reports list formats 0 and 1, so this is a general-input defect rather than a demonstrated cause of the current benchmark results. Sustain pedals, instrument programs, and channel identity do not survive the event representation, by design; that limits portability to arbitrary MIDI performances.

### C6 — Annotation features differ between training and deployment (P1/P2)

**Confirmed by source.** In preprocessing, progress is absolute onset divided by the complete annotation timeline; during continuation it resets to progress through the requested new phrase (`preprocess.py:218–245`, `generate.py:183`). Beat phase is trained using an inferred meter, while generation advances phase using a hard-coded four beats per bar (`preprocess.py:304–310`, `generate.py:169–172`). Instrument/chord/key IDs are copied from the last context event and never updated for a new harmonic plan (`generate.py:173–176`).

The claim that all input context is causal (`dataset.py:53–54`) needs qualification: normalized progress requires the full composition endpoint, and beat interpolation can use the next annotated beat. These may be legitimate preplanned score features, but availability in an open-ended interaction has not been established. This is a future-context assumption, not demonstrated cross-split leakage.

For unaligned files, chord/key IDs are zeroed, but tempo still comes from median annotation beat intervals (`preprocess.py:215–228`). A synthetic unaligned example confirmed a 120 BPM annotation-derived value. Documentation claiming all uncertain tempo context is unknown/default is too broad; the policy should explicitly describe this fallback.

### C7 — Timeline validation accepts non-finite and zero-length notes (P1)

**Verified.** `validate_timeline([NoteEvent(60, nan, nan)])` and a zero-duration note both return valid (`composition.py:80–126`). There are no finite-value or pitch/velocity-range checks. The normal generators clamp several values, so this is a validation-contract defect, not a demonstrated NaN emitted by the tested checkpoints. Future imported/custom events could pass validation and fail serialization, scheduling, drawing, or MIDI conversion.

### C8 — Checkpoints use unrestricted deserialization (P1; external-file trust boundary)

**Confirmed by source.** All production checkpoint loaders explicitly use `torch.load(..., weights_only=False)`, including `ml/inference.py:131`, `ml/evaluate.py:22`, and the annotation evaluator/generator. Loading an untrusted pickle-compatible checkpoint can execute code before architecture checks run. No malicious checkpoint or compromise was found. The audit successfully loaded the three principal checkpoints with `weights_only=True`, demonstrating that unrestricted loading is unnecessary for these artifacts. The audit's safe loader override was process-local; source behavior was not changed.

### C9 — Reproducibility and artifact-management defects (P1)

- **Confirmed mismatch:** README baseline training uses batch size 16, while the local baseline checkpoint and `models/model_config_3x512.json` record 32. The command can train a model but does not exactly specify the recorded run.
- **Confirmed risk in code:** baseline training overwrites common `logs/training_history.csv`, `logs/dataset_summary.json`, and `models/model_config.json` for different runs (`ml/train.py:134–154`). No dataset fingerprint or immutable split manifest is embedded in checkpoints; evaluation recomputes the split from whatever files are currently present.
- **Confirmed stale-artifact acceptance path:** `train_full.ps1:30–35` tolerates a failed training process if checkpoints and nonempty history merely exist, without checking that they belong to this run. The annotation launcher improves this with modification-time/status checks, but a nonzero exit still deserves investigation rather than being treated as full process health.
- **Missing:** full resume support. The baseline saves optimizer/scheduler state but has no resume CLI and does not save all RNG/scaler state; annotation checkpoints omit optimizer state (`ml/aam_annotated/train.py:214–226`).
- **Confirmed drift risk:** the annotation evaluator loads feature mappings from the current experiment folder rather than binding their semantic identity to the checkpoint. Equal-size reordered mappings can silently change meaning (`ml/aam_annotated/evaluate.py:46–51`).
- **Confirmed reporting defect for other corpora:** annotation evaluation hard-codes 300 test groups (`ml/aam_annotated/evaluate.py:102`); it is correct for this inventory but not derived from the evaluated data.

### Risks requiring targeted verification

| Risk | Evidence and practical limit |
|---|---|
| Pause sustains active MIDI notes | `_pause` stops scheduler progress but sends no note-offs/reset (`midi_player.py:101–114,125–129`). Audible behavior was not tested. Beep calls are blocking and cannot stop mid-call. |
| Tk shutdown and synchronous checkpoint load | Model loading runs on the Tk thread (`gui.py:181–191`); worker callbacks call `root.after`. Short GPU tests do not establish responsive startup/switching or safe shutdown on every platform. |
| Fallback load failures outside ImportError | The optional-import guard handles `ImportError`, not all possible native Torch/DLL initialization failures. No such failure occurred here. |
| GUI configuration state drift | `_set_busy(False)` restores only the duration selector to `readonly`; the model selector becomes `normal` (`gui.py:312–320`). An editable invalid label can no longer reliably represent the loaded model. Source-confirmed UI defect; not visually exercised. |
| Log growth and privacy | Every interaction stores `previous_composition`, and each action writes a new full snapshot (`gui.py:165–166,251–266,398–405`). With fixed notes per turn, one snapshot grows quadratically and cumulative snapshots can grow cubically in turn count. This is structural analysis, not a measured long-session benchmark. |
| Generic-corpus split leakage | AAM ID grouping works, but filename heuristics and path-based hashes do not prove separation of renamed copies, alternate renders, transpositions, or related pieces outside this corpus (`ml/dataset.py:23–25`, `ml/preprocessing.py:237–240`). |
| Dataset refresh leaves stale records | Preprocessing writes new JSONL files but does not reconcile files from older inventories. Re-running into an existing directory can retain removed/renamed-source records. Not reproduced by writing during this audit. |
| Unsupported dependency versions | Lower bounds are broad and no clean minimum-version test exists. The annotation trainer assumes APIs for which the baseline has a compatibility fallback. Installed versions worked; older versions were not tested. |

## 6. AI/ML and research-quality assessment

### What the recorded metrics actually say

The following are existing artifact values, corroborated against checkpoint metadata where applicable, not freshly rerun full-test metrics:

| Model | Best epoch | Validation loss | Test loss | Pitch accuracy | Duration MAE | Timing MAE |
|---|---:|---:|---:|---:|---:|---:|
| MIDI-only 3×512 | 13 | 0.9138 | 0.9281 | 72.53% | 0.0487 s | 0.0785 s |
| MIDI-only 4×768 | 16 | 0.7340 | 0.7293 | 78.30% | 0.0384 s | 0.0660 s |
| Annotated 4×768 | 12 | 0.6854 | 0.6830 | 79.47% | 0.0363 s | 0.1107 s |

Evidence: `test_outputs/baseline_after_audit/evaluation_report.json`, `test_outputs/large_lstm/evaluation_report.json`, `reports/aam_annotated_evaluation.json`, local checkpoint metadata. These test-output directories are local ignored artifacts. The legacy `test_outputs/evaluation_report.json` names `best_model.pt` but records the older 3×512 architecture; generic artifact paths should not be treated as immutable provenance.

Interpretation limits:

1. **All scores include padding.** The 128-step and 256-step totals are different populations of real and synthetic positions.
2. **Evaluation is teacher forced.** Correct prior notes and annotation context are supplied at every step; this does not measure error accumulation in sampled continuations.
3. **The future final pitch is supplied as a condition.** This is defensible for a known-endpoint task. It is not an unconditional forecasting benchmark, and the final target is explicitly revealed. Padded endpoints amplify the shortcut. Evaluate user-specified endpoints separately and report the endpoint task honestly.
4. **Total losses are not directly comparable between baseline and annotation models.** Timing targets change from scaled bucket indices to log-normalized beat intervals. Different numerical losses can change the total without a corresponding musical improvement.
5. **Duration MAE compares decoded token representatives**, not original continuous note durations. Baseline timing MAE also uses decoded timing buckets. Quantization error is excluded from those reported targets (`ml/evaluate.py:36–56`).
6. **Annotation timing conversion uses the input event's tempo to decode the next event's gap** (`ml/aam_annotated/evaluate.py:87–88`), although preprocessing normalizes each event's gap using its own tempo. At tempo transitions this can misstate seconds; incidence and effect were not measured.
7. **All reported source events have velocity token 3.** Zero velocity error mostly reflects a constant-class task. It provides no evidence of expressive dynamics.
8. **No confidence intervals or replicated seeds are reported.** Closely related track members and windows are not independent observations. Use composition-level aggregation/resampling rather than treating a million targets as a million independent samples.

The annotation model's recorded pitch increase over the same-sized MIDI model is about 1.17 percentage points, with a small duration-MAE reduction and a larger reported timing MAE. These observations do not isolate the value of annotations. Architecture, timing features/targets, optimization, and generation behavior differ; controlled ablations are necessary.

### Musical coherence and diversity

The model operates on a flattened note stream rather than explicit voices, bars, or sections. The historical corpus audit records extensive source overlap and mixed instrument/percussion content. A pitch number for a drum is treated as a pitched note in the MIDI-only model, and exported examples generally use one channel with no instrument identity. The included annotation example is seeded from Drums, yet its MIDI note-ons are on channel 0 with no program change; instrument conditioning does not imply faithful instrument rendering.

Short contexts and short turns can support motifs, but no explicit phrase-form, harmonic plan, or long-term structural objective is evaluated. In one verified two-second baseline fixture, the output was `[60,64,69,60,64,69,60,64,69,60,64,69,69]`. This establishes repetition in that example, not corpus-wide collapse. The fallback is intentionally deterministic, limited to C4–B4 in C major, and ignores the temperature control.

Top-k/top-p and repetition penalties provide exploration controls but have not been ablated. Hard octave-leap limits and forced monophony can suppress useful ideas as well as invalid outputs. The target cue is guidance, not an endpoint guarantee: the verified 0.5-second 4×768 run produced only C for an A target. Historical comparison reports also show absent target pitches for some short turns.

### Co-creativity, novelty, and appropriate claims

The system is genuinely interactive in the limited sense that human choices recur throughout composition, affect the generated material, and can undo entire AI turns. Authorship is visible, and adjustable turn duration controls how often the user intervenes. It lacks explicit candidate acceptance, independent regeneration, phrase locking, direct note editing, or a representation of user intent beyond a pitch and scalar settings. It is therefore a constrained turn-taking collaborator, not yet a rich shared editor.

The strongest prospective research contribution is **how AI turn duration and musical conditioning affect human agency, exploration, and useful musical outcomes**. An LSTM architecture or the mere addition of chord/key embeddings is not sufficient novelty by itself. No external literature search was performed because network access was not approved; this audit cannot establish novelty relative to the complete literature.

Demonstrated: working local components, source-group separation, recorded training artifacts, and bounded valid generation. Hypotheses: annotations improve human-perceived coherence; different autonomy durations improve agency; filtering improves usefulness rather than merely suppressing variation. Those hypotheses require experiments.

Recommended evidence design, after evaluation repair:

- **Cheap baselines:** deterministic fallback, repeat-last-note, endpoint-copy, empirical transition/Markov baseline, and the frozen 3×512 model. These detect padding/target shortcuts and whether a larger model adds practical value.
- **Matched ablations:** same architecture, source IDs, real targets, timing representation, optimization budget, and seeds; add instrument, chord/key, beat phase, and progress separately. Include unknown/shuffled annotation controls and remove future-derived progress when testing open-ended operation.
- **Autoregressive evaluation:** common seed contexts and target notes; multiple sampling seeds; target arrival/presence, rhythmic alignment, repetition, pitch-range validity, phrase-duration violations, silence/overlap, and latency. Report pre- and post-sanitization results to expose how much heuristics repair the model.
- **Human study:** counterbalanced within-participant comparison of baseline/fallback/annotated conditions and 0.5/1/2-second autonomy. Blind model labels where possible; measure perceived control, usefulness, coherence, surprise, frustration, time to a satisfactory phrase, rejection/undo rate, and retained AI material. A small pilot should refine the protocol before a powered study.
- **Novelty/generalization:** detect near-duplicates against training phrases, hold out related composition families where identifiable, and evaluate a small legally usable human-composed MIDI corpus. Diversity alone is not musical quality, and familiarity alone is not evidence of memorization.

## 7. Ranked upgrade roadmap

Effort figures below are **estimates of developer hours**, not measured durations or commitments. They assume familiarity with this codebase and exclude dataset acquisition, human-subject recruitment/approval, long training time, and GPU wall time. Items should be implemented only in a subsequent authorized change task.

### P0 — Critical research-correctness fix

**R1. Mask padding and establish trustworthy real-target evaluation.**

- Problem: C1 undermines losses, model selection, and reported comparisons.
- Solution/benefit: return valid lengths/masks from both datasets; use masked losses and metrics; separate real transitions, endpoints, and padding; publish corrected checkpoint evaluations before retraining.
- Complexity / effort: **Medium, estimated 8–16 h**, plus later evaluation runtime.
- Requirements: CPU for fixtures; existing checkpoints and processed data; GPU helpful for full reevaluation. No new dataset needed.
- Risks/trade-offs: corrected scores can fall and rankings can change; existing weights learned padded repetitions and will not become unbiased merely by changing evaluation.
- Files: both `dataset.py`, `train.py`, `evaluate.py` families; reports and the timing/test tooling.
- Acceptance: adding any amount of padding to a fixture changes neither real-token loss nor metrics; valid target counts match the window inventory; all three checkpoints have real-only reports with versioned protocol IDs.

### P1 — High-value reliability and usability improvements

**R2. Fix optional-dependency fallback and selector state.**

- Problem: C4 and model selector state drift break a core advertised fallback path.
- Solution/benefit: give fallback and learned adapters the same status interface; preserve selector `readonly` state; show selected versus active generator accurately.
- Complexity / effort: **Low, estimated 1–3 h**.
- Requirements: CPU and simulated import failures; no weights/data required.
- Risks/trade-offs: broad exception handling must not conceal real dependency failures.
- Files: `gui.py:14–24,181–191,312–320`, small adapter tests.
- Acceptance: no-Torch launch/generation/model selection and missing-checkpoint selection all produce clear fallback status without exceptions; model selection remains read-only after generation.

**R3. Define and enforce phrase-duration and event-validity contracts.**

- Problem: C3/C7 make user controls and diagnostics disagree with output.
- Solution/benefit: choose total audible duration as the contract, then shorten/drop a final note that exceeds the budget; reject non-finite/range-invalid values; extend the existing regression with these invariants.
- Complexity / effort: **Low–Medium, estimated 4–8 h**.
- Requirements: CPU fixtures; local checkpoints for integration cases.
- Risks/trade-offs: clipping can harm musical articulation; alternatively explicitly label an onset horizon if preserving tails is intentional.
- Files: `composition.py`, `ml/inference.py`, `ml/aam_annotated/generate.py`, `ml/regression_timing.py`, GUI labels.
- Acceptance: 0.5/1/2-second phrases stay within the chosen contract for seeded tests; NaN/Inf, zero/negative durations, out-of-range pitches, and nonmonotonic times are rejected or explicitly repaired with diagnostics.

**R4. Add a focused, side-effect-free regression suite.**

- Problem: missing automated coverage allowed small, reproducible failures to persist.
- Solution/benefit: use standard-library `unittest` or a declared test dependency; cover padding masks, fallback, parser fixtures, export round trips, selector status, shutdown callbacks, and timing budgets. Run CPU tests automatically; keep GPU/checkpoint cases optional and labeled.
- Complexity / effort: **Medium, estimated 6–12 h**.
- Requirements: CPU for default suite; optional local GPU/weights. No full corpus required.
- Risks/trade-offs: mocks can miss actual audio/Tk behavior; include a short manual Windows smoke checklist alongside tests.
- Files: new test files/CI configuration in a future task, existing regression runner, small separations in GUI export as needed.
- Acceptance: one documented command runs without data downloads, audio playback, or output files and catches C1/C3/C4/C5/C7 fixtures.

**R5. Make checkpoint loading safe and lightweight.**

- Problem: C8 and loading training-only optimizer tensors into the GUI increase trust and memory exposure.
- Solution/benefit: use restricted loading, validate schema and conditioning version, load on CPU first, and distribute inference-only state/config/mapping fingerprints.
- Complexity / effort: **Low–Medium, estimated 3–6 h**.
- Requirements: current checkpoint files; CPU sufficient. No retraining needed for packaging changes.
- Risks/trade-offs: incompatible legacy checkpoints need explicit trusted conversion rather than an automatic unsafe fallback.
- Files: `ml/inference.py`, both evaluators, annotation generator and checkpoint writers.
- Acceptance: current principal checkpoints load through the restricted path; malformed schemas fail clearly; unsafe custom-object fixtures are rejected; GUI checkpoint contains no optimizer tensors.

**R6. Make experiments and installation reproducible.**

- Problem: C9, broad dependency bounds, and incomplete public artifacts prevent exact replication.
- Solution/benefit: use run-specific directories; persist source/data/split/feature hashes, resolved CLI arguments, environment versions, and seeds; correct README batch size; fail launchers on unverified nonzero exits; provide separate minimal-runtime and training environment instructions.
- Complexity / effort: **Medium, estimated 8–16 h**.
- Requirements: clean Windows test environment for validation; existing source/data for fingerprint checks. CUDA remains optional for the app, recommended for training.
- Risks/trade-offs: GPU builds vary by hardware; avoid presenting one lockfile as universal. Fingerprinting must avoid unnecessary repeated hashing of the 44 GB unused audio collection.
- Files: `requirements.txt`, README, launchers, both training/evaluation pipelines, configs; future environment/manifest files.
- Acceptance: a clean Windows install runs fallback and a known checkpoint using documented steps; a report identifies its exact run/split/mappings; changed data or mappings trigger an explicit mismatch; failed training cannot pass using stale artifacts.

**R7. Repair MIDI parsing with explicit ambiguity policies.**

- Problem: C5 corrupts some general MIDI inputs and loses provenance needed to diagnose overlaps.
- Solution/benefit: track end times and channel/track provenance; specify note-pairing policy; reject unsupported type-2/SMPTE cases or handle them deliberately. Consider using the already-required Mido parser for event decoding while retaining project quantization.
- Complexity / effort: **Medium, estimated 8–16 h**.
- Requirements: small synthetic MIDI fixtures and a bounded AAM comparison sample; no new large dataset.
- Risks/trade-offs: fixing parsing can change derived data and invalidate old checkpoint comparisons. Preserve the current preprocessing version for historical reproduction.
- Files: `ml/preprocessing.py`, `ml/audit_aam_dataset.py`, annotation alignment, future parser tests.
- Acceptance: orphan-note duration respects its track policy; ambiguous overlaps are reported; type-2 input is explicitly supported/rejected; tempo-change and running-status fixtures round-trip within documented quantization tolerance.

**R8. Clarify tempo behavior and remove misleading controls.**

- Problem: C2 exposes a variable conditioning control without variable-condition training evidence.
- Solution/benefit: immediately explain whether tempo affects future generation or existing playback; for the learned baseline either constrain conditioning to its trained value or separate a deterministic playback-speed control. Evaluate beat-based training before promising learned tempo response.
- Complexity / effort: **Low–Medium, estimated 3–6 h** for the honest UI contract; beat-based retraining belongs to R11.
- Requirements: CPU and seeded checkpoint checks; no new data for the UI correction.
- Risks/trade-offs: constraining a control reduces apparent capability but makes behavior predictable; rescaling existing music needs consistent log/export semantics.
- Files: `gui.py`, `ml/inference.py`, `midi_player.py`, export paths, README.
- Acceptance: automated tests show exactly which timestamps change under a tempo adjustment, and the UI explains the same behavior; model conditioning stays within the declared supported contract.

**R9. Add session recovery and bounded local logging.**

- Problem: exported compositions cannot be reopened and full-history snapshots grow rapidly.
- Solution/benefit: a versioned session format and loader; stable session ID; append only new actions or atomically update one snapshot; retain provenance without copying all prior events into every interaction; surface save errors persistently.
- Complexity / effort: **Medium, estimated 8–16 h**.
- Requirements: CPU; synthetic session histories.
- Risks/trade-offs: migration must preserve old logs; retention policies must be explicit and must not silently delete user compositions.
- Files: `composition.py`, `gui.py:251–266,398–417`, future session persistence module.
- Acceptance: save/reopen preserves events, authorship, settings, and undo semantics; 1,000 fixed-size interactions have linear or near-linear persisted size; failed saves remain visible to the user.

### P2 — Research extensions justified against the current baseline

**R10. Run controlled annotation ablations.**

- Problem: the current experiment cannot attribute improvements to annotations and deploys them inconsistently.
- Solution/benefit: fix progress/meter/unknown-context semantics; use matched architecture, timing target, optimizer budget, masks, source groups, and seeds; separately add instrument, chord/key, and beat features. Retain the 3×512 reference.
- Complexity / effort: **High, estimated 24–48 h**, excluding GPU runs.
- Requirements: current MIDI and annotations; local GPU for replicated training; no audio dataset needed.
- Risks/trade-offs: annotations may offer no measurable benefit; extra conditioning can overfit dataset structure. Do not integrate into the GUI solely because a scalar test loss is lower.
- Files: `ml/aam_annotated/`, baseline experiment configs, evaluation/reporting.
- Acceptance: matched real-only metrics with composition-level uncertainty estimates; feature-availability policy matches generation; a preregistered primary outcome improves or a null/negative result is clearly reported.

**R11. Align rhythm and voice representation with the intended output.**

- Problem: seconds/bucket timing, constant tempo condition, and mixed polyphonic/percussion streams do not match a monophonic melodic assistant.
- Solution/benefit: first compare a filtered melodic/voice-preserving baseline and beat-normalized onset/duration representation against the current model. Preserve meter/instrument where useful. A larger/new architecture is not a prerequisite.
- Complexity / effort: **High, estimated 24–48 h**, excluding training.
- Requirements: existing MIDI/annotation archives and GPU; a legally usable external MIDI test set only for generalization claims.
- Risks/trade-offs: filtering shrinks data and can remove legitimate cross-voice context; quantization choices trade expressive timing against learnability.
- Files: preprocessing, event schemas, dataset/model adapters, MIDI writers, inference and evaluation.
- Acceptance: matched-seed continuation tests demonstrate fewer beat/voice violations without worse user preference or unacceptable latency; tempo response is measured over a declared supported range.

**R12. Evaluate co-creation and improve user agency.**

- Problem: objective metrics omit control/usefulness and the current interface commits a single candidate immediately.
- Solution/benefit: add two or three auditionable continuations, accept/regenerate, phrase locking, and eventually simple pitch/duration edits; instrument retained/rejected material; pilot a counterbalanced autonomy-duration study.
- Complexity / effort: **High, estimated 32–64 h** for interaction implementation and pilot tooling, excluding recruitment/study execution.
- Requirements: target users, suitable study procedures, local CPU/GPU; no new training dataset for the first pilot.
- Risks/trade-offs: alternatives add decision load and generation latency; retaining logs requires participant awareness and a defined retention plan.
- Files: GUI, composition action history, generator API, local logging, future study protocol.
- Acceptance: participants can audition/reject without changing committed music; pilot records agency/usefulness/coherence and retention measures; conclusions are limited to the tested participant sample and conditions.

### P3 — Optional polish

**R13. Improve feedback, accessibility, and playback ergonomics.**

- Problem: compact controls, numeric labels for non-natural pitches, synchronous loading, and uncertain pause behavior reduce confidence in the interface.
- Solution/benefit: responsive layout, readable pitch names, focus-aware shortcuts, disabled unavailable controls, visible active model/loading state, and a tested pause policy that silences or deliberately sustains notes.
- Complexity / effort: **Medium, estimated 6–12 h**.
- Requirements: Windows display/audio hardware; manual testing at minimum window size and common display scaling levels.
- Risks/trade-offs: changing shortcuts or pause semantics can surprise existing users; user-facing behavior should be documented.
- Files: `gui.py`, `midi_player.py`, README.
- Acceptance: controls remain accessible at the supported minimum size/scaling; shortcuts do not interfere with text input; repeated play/pause/stop does not leave stuck notes or stale status.

**R14. Consolidate duplicated utilities and distribution documentation.**

- Problem: separate pitch maps, MIDI exporters, and decoders can drift; ownership/license intent is not resolved.
- Solution/benefit: centralize pure conversion/export helpers behind current behavior tests; document supported input formats and asset requirements; have the owner decide code licensing and verify data/example redistribution terms before publication.
- Complexity / effort: **Low–Medium, estimated 4–8 h** for engineering/documentation; external rights review time is unknown.
- Requirements: CPU and owner-provided licensing decisions; no model training.
- Risks/trade-offs: avoid broad refactoring that obscures behavior changes; no license should be invented or assigned without authority.
- Files: `config.py`, generator maps, exporter/decoder modules, README, and a future license file if authorized.
- Acceptance: one tested seconds-to-MIDI writer serves the application and examples; source and dataset terms are separately documented; no claim of open-source status is made without an explicit license.

## 8. Quick wins achievable in one day

For one developer, a realistic first day is R2 (1–3 h), the duration/finite-value core of R3 (4–5 h), and documenting the evaluation caveat plus batch-size mismatch (about 1 h). If research credibility is the immediate goal, instead spend the day on the first masked dataset/evaluator implementation and fixtures under R1; do not claim the entire roadmap fits in a day.

Other small independent wins: use restricted checkpoint loading for the existing compatible files; preserve the model selector's read-only state; label tempo semantics; stop describing undo/clear as full editing; distinguish selected model from the actual fallback generator. All require implementation authorization beyond this audit.

## 9. Suggested work for the next week

Suggested 35–40-hour focus, adjusted after the first tests:

1. Mask real targets and correct metric aggregation; test short windows and endpoint exposure (8–16 h).
2. Fix fallback selection, event validity, and phrase-duration contract (5–8 h).
3. Add CPU regression fixtures and restricted checkpoint loading (7–10 h).
4. Correct documentation/run identity and produce masked reports for existing checkpoints (remaining time, with evaluation runtime tracked separately).

Exit criteria: no padding in headline metrics, all known small regressions covered, honest tempo/duration controls, and reports tied to an identifiable dataset/configuration. Defer broad UI redesign, new architectures, and full annotation retraining until these results are available.

## 10. Longer-term research opportunities

The most defensible sequence is: reliable evaluation, matched feature ablations, better rhythm/voice representation if evidence warrants it, then user studies. Study shorter versus longer AI turns as a trade-off between agency and interruption frequency. Compare endpoint pitch conditioning with a richer but still understandable intent such as a target chord or contour. Evaluate candidate audition and phrase locking before assuming more autonomous generation helps.

Measure novelty with training-set similarity controls and perceived usefulness with users. Investigate whether a smaller model or simple transition baseline performs as well after removing synthetic-target shortcuts. Transformer replacement, audio generation, or a distributed architecture is not justified by the current evidence and is not recommended as an initial upgrade.

## 11. Testing, reproducibility, security, privacy, and portability

### Diagnostics actually run

| Diagnostic | Result | Scope / limitations |
|---|---|---|
| Python AST parsing | 23 source files passed | Syntax only; no bytecode generated. |
| Module imports | 21 explicitly imported modules passed; package initializers loaded normally | Current installed environment only. No application window constructed. |
| Runtime inventory | Python 3.14.7, Torch 2.11.0+cu128, NumPy 2.5.3, Mido 1.3.3, Matplotlib 3.11.2 | Package versions from installed metadata; no dependency changes. Pytest was not installed. |
| CUDA availability | RTX 4060 Laptop GPU available | No training or full hardware benchmark. |
| Existing timing regression | Six runs passed | Invoked its existing `main` with original default fixture/seed; report writes intercepted in memory and checkpoint loads restricted. |
| Principal checkpoint loads | Both GUI checkpoints and annotated epoch-12 checkpoint passed | `weights_only=True`; this intentionally avoids exercising the production unsafe loader behavior. |
| Annotation forward | One real 256-step test window; all heads finite | Not a full evaluation or quality test. |
| Annotation generation | One bounded 0.5 s / max-four-event case generated one note | Export and JSON writes intercepted; MIDI reopened from bytes. |
| Fallback grid | 63 combinations passed | Seven notes × three BPM values × three durations; determinism, endpoint, timeline, append/undo checked. |
| GUI MIDI export | Four events; duration preserved at 1.17 s | Fake dialog/status, real MIDI serialization in memory; no audible playback. |
| Split inspection | Baseline and annotation source assignments match; no ID overlaps | Current inventory only, not semantic near-duplicate detection. |
| Padding analysis | Counts in C1 reproduced from per-file index | No full training/evaluation rerun. Baseline test row total independently counted. |
| Parser fixtures | Orphan-track-end and format-2 defects reproduced; cross-track ambiguity demonstrated | Does not quantify frequency in AAM. |
| Optional-dependency fixture | Model-select `AttributeError` reproduced | Simulated ML import failure, not a second clean Python installation. |
| Validator fixtures | NaN and zero-duration events accepted | Defects of validation; normal model fixture outputs were finite. |
| Source/network inspection | No application HTTP/socket/service-client path found | Dependencies and OS traffic not packet-captured. |

No `test*.py` files or conventional test suite were found. The annotation “smoke test” performs gradient updates and writes checkpoints/configs, so it was **not run** under the audit-only restrictions. Neither trainer, either full evaluator, preprocessing entry points, nor the historical data-audit entry point was run because they write artifacts and/or do substantial work. The timing regression was the existing automated check run with its output redirected to memory.

The six generation calls took approximately 0.634/0.015/0.026 seconds for baseline 0.5/1/2-second requests, and 0.007/0.008/0.016 seconds for the larger model. These are single observations on this GPU, with warmup/order effects; the first call is cold. They are not stable benchmarks, p95 estimates, or evidence that the larger model is always faster. CPU inference, startup responsiveness, full-training peak memory, and long-session behavior were not benchmarked.

### Reproducing these diagnostics without altering the project

Use `python -B` and ephemeral code passed on stdin. The performed checks imported modules, parsed source using `ast.parse`, called pure functions with in-memory fixtures, read existing JSON/index/checkpoint files, and used `io.BytesIO` for MIDI serialization. For the existing regression, `unittest.mock.patch` intercepted `Path.mkdir` and `Path.write_text`; `torch.load` was wrapped to force `weights_only=True`. For annotated generation, `MidiFile.save` was also redirected to bytes. No diagnostic source files were created.

A repeat audit should not simply execute the documented CLI commands and assume they are read-only: `ml.regression_timing`, `ml.evaluate`, `ml.compare_models`, `ml.aam_annotated.generate`, and `ml.audit_aam_dataset` all write output. Reuse the memory interception approach or run them only with permission to create their artifacts.

### Offline operation and privacy

Application source uses local Python/Tk, Torch weights, MIDI events, Windows APIs, and local files. No code was found that sends notes, session logs, or model inputs to external services. Local generation and export were exercised without making network requests. This supports an offline execution design once dependencies/assets are installed; it does not certify every dependency under a network-disabled clean-machine test.

Sessions contain timestamped choices, past compositions, model/device details, settings, and latency. Logs are automatically saved under the project directory and persist after undo/clear as earlier snapshots. “Local” is not the same as “ephemeral” or “encrypted”; users should be told what is saved. No cloud synchronization is implemented by this project, although the surrounding filesystem could independently be synchronized by the user's software.

Checkpoint trust is the concrete security concern in C8. ZIP extraction traversal is avoided by the baseline reader because it does not extract members. General untrusted file size/resource exhaustion remains a hardening concern, especially in annotation preprocessing; this is not evidence of a remotely reachable service or an observed exploit.

### Licensing and distribution

No project `LICENSE` file was found. The README explicitly says code ownership/licensing is unverified and describes the AAM record's license field as blank. Those are repository statements, not a current legal or external-record verification; no network access or legal-source review was performed. Dataset terms, code licensing, checkpoint redistribution, and the included generated example should be resolved separately by the owner before public distribution or stronger reuse claims.

### What is needed on another machine

| Goal | Required assets/environment | Current evidence |
|---|---|---|
| Run the rule-based GUI | Python with Tcl/Tk, app source; Windows audio support for sound | Components work locally; README installs Torch even though fallback can avoid it. No clean install tested. |
| Export MIDI | Above plus Mido | Verified through in-memory GUI export. |
| Run learned GUI models | Compatible Torch/NumPy, the named compatible checkpoint, source; CPU or supported CUDA setup | Both checkpoints work on local CUDA. No training data required for inference. |
| Retrain MIDI-only model | MIDI archives/files or processed JSONL, Torch and training dependencies, storage and substantial compute | Scripts/source inspected; reported runs exist. No exact independent replay performed. |
| Retrain annotation model | Corresponding MIDI and annotation archives, baseline split inventory, generated mappings/arrays, training environment | Local assets exist; launcher requires CUDA. Audio archives are unnecessary. |
| Generate with annotation CLI | Checkpoint, mappings, binary arrays/indexes, and processed source JSONL | Unlike the GUI adapter, it is tied to experiment/source data. Bounded local case verified. |

Windows is the only implemented audible playback platform. Tk and CPU inference may run elsewhere, but Linux/macOS have no equivalent audio backend here; the player reports unavailable if neither Windows MIDI nor winsound is present. Cross-platform operation was not tested. A requirements file with broad minimum versions is not a reproducible environment specification; clean Windows reproduction remains unverified despite the working local setup.

## 12. Final verdict on maturity and readiness

The project is more than a mockup: trained local models exist, inference runs quickly in small GPU cases, timing placement is substantially defended, fallback behavior is useful, and the annotation pipeline is implemented through generation. It is a credible basis for a research prototype and a supervised local demo.

It is not ready for benchmark-quality comparative claims until padded targets are excluded and conditioning/evaluation contracts are made explicit. It is not ready as a general music application until fallback-selection reliability, phrase-duration behavior, session recovery, playback semantics, installation validation, and licensing/distribution questions are addressed. No evidence currently supports an assertion that the annotation experiment improves human-perceived music or creative agency.

The most economical route forward is to preserve the existing model family, repair the measurement and small reliability defects, and then decide from corrected evidence whether more modeling work is worthwhile.

## 13. Recommended next three actions

1. **Repair masks and reevaluate existing checkpoints on real targets.** This is the single most important next step; freeze current reports as historical and label their padding limitation.
2. **Fix the small reproduced reliability failures and make them regression tests:** no-Torch model selection, phrase end budgets, finite event validation, and orphan-track note closure. Use restricted checkpoint loading at the same time.
3. **Run a matched, small-scope comparison and user pilot:** frozen baseline versus fallback, then controlled annotation ablations, with common contexts, seeds, timing semantics, and explicit control/usefulness outcomes. Decide on retraining or richer UI only after that evidence is available.

No upgrades in this roadmap were implemented as part of this audit.


## Implementation addendum — 29 October 2026

This document records the original audit. Subsequent work corrected masked evaluation/training, added reference-guided generation and editing in the optional extended studio, restored the linked repository UI as the default, and implemented opt-in LoRA for both MIDI LSTMs. See `IMPLEMENTATION_REPORT.md`, `docs/LORA.md`, `CLEANUP_REPORT.md`, and `reports/lora_validation_20261029.json` for current behavior and validation. Original checkpoints were preserved; only bounded adapter smoke training was performed. No perceptual improvement is established.
