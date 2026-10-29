# AAM dataset audit (historical snapshot)

> Historical inventory and static-code-review snapshot from 2026-10-07, before annotation-aware preprocessing and training were implemented. Sections describing future work or saying no retraining was run refer only to that audit date. The current annotation-aware implementation and completed experiment are documented in `aam_annotated_training_plan.md` and `aam_annotated_dataset_stats.json`.

Generated: 2026-10-07; historical audit snapshot, before the annotation-aware training experiment

## Scope and safety

The audit enumerated the local archives under `data/raw/`. It did not extract archives, decode audio frames, train a model, overwrite checkpoints, or change model/GUI/inference code. MIDI archives were parsed in memory one member at a time. Annotation records and ZIP inventories were read. For each FLAC member, the audit read only its 42-byte FLAC marker and STREAMINFO header. Audio files are references/metadata only and are not LSTM inputs.

## A. Archives found

Found 3 MIDI ZIPs (22,574 members), 3 annotation ZIPs (9,000 ARFF members), 3 audio-mix ZIPs (3,000 FLAC members), and 1 `info.zip`. The three audio ZIPs occupy 44,219,516,556 bytes on disk; individual member expanded size sums to 44,219,028,774 bytes. The info archive contains 35 files: {'.conf': 3, '.txt': 1, '.nkm': 31}.

| Archive | Component | Files/members | Valid | Invalid | Source IDs | Bytes |
|---|---|---:|---:|---:|---:|---:|
| `0001-1000-midis.zip` | midi | 6,275 | 6,275 | 0 | 1,000 | 2,108,953 |
| `1001-2000-midis.zip` | midi | 7,920 | 7,920 | 0 | 1,000 | 3,544,696 |
| `2001-3000-midis.zip` | midi | 8,379 | 8,379 | 0 | 1,000 | 3,835,784 |
| `0001-1000-annotations-v1.1.0.zip` | annotations | 3,000 | 3,000 | 0 | 1,000 | 8,451,769 |
| `1001-2000-annotations-v1.1.0.zip` | annotations | 3,000 | 3,000 | 0 | 1,000 | 8,601,012 |
| `2001-3000-annotations-v1.1.0.zip` | annotations | 3,000 | 3,000 | 0 | 1,000 | 8,745,087 |
| `0001-1000-audio-mixes.zip` | audio_reference_only | 1,000 | 1,000 | 0 | 1,000 | 14,710,551,938 |
| `1001-2000-audio-mixes.zip` | audio_reference_only | 1,000 | 1,000 | 0 | 1,000 | 14,730,539,954 |
| `2001-3000-audio-mixes.zip` | audio_reference_only | 1,000 | 1,000 | 0 | 1,000 | 14,778,424,664 |
| `info.zip` | metadata_and_sampler_patches | 35 | 35 | 0 | 0 | 11,464,390 |


Archive sizes, MIDI/annotation/info SHA-256 values, member counts, and source IDs are in `aam_dataset_manifest.json`. Audio ZIP hashes were skipped because the files total over 44 GB; the manifest says so explicitly.

## B. MIDI validation and use

The MIDI archives contain 22,574 MIDI members: 22,574 parsed as valid, 0 invalid, 0 empty, and 0 containing fewer than three note events. Parsed MIDI note events: 7,552,855. Exact duplicate MIDI byte-content groups: 0.

Raw MIDI duration (from SMF end-of-track tick/tempo map): {'count': 22574, 'min': 62.83315200000001, 'median': 148.2454305, 'mean': 146.4008463780234, 'max': 180.98350399999998}. Tracks per MIDI: {'count': 22574, 'min': 1, 'median': 1.0, 'mean': 3.518251085319394, 'max': 16}; format counts: {'1': 4981, '0': 17593}; explicit tempo-event BPM distribution: {'count': 471859, 'min': 60.0, 'median': 125.0, 'mean': 123.41841432099419, 'max': 180.00018}; time signatures found: none encoded in the MIDI files. Pitch range among parsed note events: [35, 90]. Program-change numbers (General MIDI zero-based): {57: 1169, 17: 1253, 66: 663, 24: 4063, 32: 1231, 4: 1444, 110: 1763, 104: 1442, 41: 620, 43: 1287, 77: 621, 42: 1568, 33: 1218, 56: 671, 40: 575, 1: 1520, 0: 1505, 29: 1524, 75: 647, 30: 609, 73: 662, 65: 599, 76: 582, 71: 604, 27: 1516}. Processed event count per MIDI: {'count': 22574, 'min': 4, 'median': 141.0, 'mean': 334.5820412864357, 'max': 3575}. The current preprocessed corpus has 22,574 JSONL files, 7,552,855 valid JSONL events, 0 malformed JSONL rows, and 0 raw MIDI members without a corresponding processed JSONL. The preprocessing discovery scans recursively for `.mid`/`.midi`, parses ZIP members without extraction, and ignores ARFF/FLAC as model inputs.

**Are all raw MIDI files used?** The report compares each raw MIDI member’s expected source-hash output name to `data/processed`. See `unmapped_raw_midi_files`; any valid-but-unmapped members (including files too short for windows) are explicitly countable in the manifest. Existing preprocessing does not turn the AAM audio or ARFF annotations into LSTM sequences.

## C. Annotation inventory and MIDI mapping

There are 3,000 distinct composition IDs represented in the annotation archives. Parsed files/rows by annotation type:

| Type | Files | Records | Malformed rows | Fields |
|---|---:|---:|---:|---|
| `beatinfo` | 3,000 | 874,308 | 0 | 4 |
| `onsets` | 3,000 | 1,712,284 | 0 | 32 |
| `segments` | 3,000 | 28,456 | 0 | 6 |


Annotation field names/types, representative values and onset-cell counts are recorded in `aam_dataset_stats.json`. `beatinfo` contains Start time in seconds, Bar count, Quarter count, and Chord name. `onsets` contains Onset time in seconds plus 31 `Onset events of <instrument>` STRING columns; the payloads contain signed/numeric tokens in bracketed lists (not explicit duration or velocity columns). `segments` contains Start time in seconds, Mark, Tempo, Key, Instruments, and Generator. Annotation↔MIDI missing IDs: 0 annotation-only, 0 MIDI-only. All 3,000 IDs join exactly by numeric AAM composition ID, verified across full ID sets; a composition can map to multiple MIDI files (3–14 here), so this is intentionally not a one-annotation-file-to-one-MIDI-filename join. No malformed data rows were found.

## D. Audio inventory and mapping

The audio archives contain 3,000 named FLAC files with 0 invalid headers. STREAMINFO-derived duration statistics: {'count': 3000, 'min': 122.72083900226757, 'median': 159.13945578231292, 'mean': 156.5474334542706, 'max': 183.70460317460316}; sample rates: {44100: 3000}; channel counts: {1: 3000}. Audio↔MIDI missing IDs: 0 audio-only, 0 MIDI-only. The numeric source ID in `NNNN_mix.flac` is mapped to the MIDI/annotation composition ID; this does not mean audio is used in model training.

## E. Split leakage and token/window counts

The existing deterministic `ml.dataset.split_files` split (seed 42) groups processed AAM files by numeric composition ID. `EventWindowDataset` builds next-event windows of `sequence_length` events, with stride equal to that length and last-event padding for the short final window. The source-ID grouping audit found 0 composition IDs appearing across multiple splits. Exact event counts and sequence windows:

| Split | Files | Groups | Events | Windows (L=128) | Windows (L=256) |
|---|---:|---:|---:|---:|---:|
| train | 18,114 | 2,400 | 6,061,647 | 55,879 | 34,039 |
| validation | 2,212 | 300 | 766,783 | 7,030 | 4,235 |
| test | 2,248 | 300 | 724,425 | 6,715 | 4,130 |


Approximate target-token exposures per full pass are train windows × sequence length: {'128': 7152512, '256': 8713984}. This is exposure count, not unique event count. The 3×512 baseline uses the existing 128-event context by default; 256 is the larger experiment’s recorded context. No split files or training artifacts were changed.

## F. Composition-level overlap in MIDI

The parser’s flattened event streams contain overlaps in 21,929 MIDI files: 4,748,127 notes start while at least one earlier note remains active; 16,530,269 overlapping pairs, including 1,036,175 same-pitch pairs. Chord/polyphonic overlaps are expected in source material; same-pitch pairs merit review but are not automatically errors. The parser merges tracks and does not preserve track identity, so this audit cannot label every overlap as intentional or accidental. Counts are after 50 ms quantization; exact before/after overlap attribution was not established here.

### Timing-path code inspection

- **Source/parser:** MIDI ticks become seconds using tempo meta-events (default 500,000 μs/beat); note-on/off pairs are collected by channel and pitch, then sorted by onset/pitch. Onsets and durations are rounded to 50 ms; `delta_time` is the nonnegative gap between successive quantized onsets. This flattens tracks and omits instrument identity from JSONL.
- **Encoding/decoding:** Events carry pitch, duration, velocity, and onset-gap timing. Training buckets duration/timing to 16 categories, velocity to 8, pitch to 128. Decoding is representative reconstruction, not exact MIDI timing; long gaps clip to the largest bucket.
- **Inference:** `Composition.end_time` is max(`start + duration`). `LSTMGenerator.generate_continuation` captures `composition_end`, sets `continuation_boundary = composition_end + human_note_duration(bpm)`, creates local events by cumulative generated `delta_time`, sanitizes them, then adds the boundary to each local start. It logs composition end, local/absolute starts and ends, overlap, and before/after sanitization counts. `autoregressive_generate` raises each AI onset gap to at least the prior AI note duration.
- **Append/export:** GUI sanitizes against the same boundary; `Composition.add_interaction` places the human cue at the previous composition end and appends already-absolute AI events. MIDI export converts absolute seconds using selected BPM and ticks-per-beat; note-offs sort before note-ons at equal ticks. Static inspection found no local-time reset/offset defect in this path.
- **Diagnosis:** Source MIDI already has extensive overlap. Preprocessing approximately retains it in flattened, quantized data. Current inference boundary checks and monophonic AI sanitization should prevent same-AI-line overlap and pre-boundary events. This is static code review, not a live GPU regression: the 0.5/1/2 second controlled cases were not run here.
- **Defect categories:** MIDI delta ticks are unsigned. Parser clamps negative gaps to zero and note durations to at least 50 ms, so preprocessed JSONL has no negative deltas/durations. Exact duplicate MIDI contents are absent, though same-pitch overlaps exist. Note pairing is by channel/pitch across tracks, without track identity.

## G. `info.zip`

`info.zip` contains {'.conf': 3, '.txt': 1, '.nkm': 31}, including 3 configuration files, 35 nonblank instrument-catalog lines, and 31 sampler patch files. Its purpose is configuration/catalog/patch metadata; it is not an additional labeled MIDI/audio corpus. Active config lines are included in the manifest.

## H. Event representation and potential annotation utility

Each processed JSONL row stores `pitch` (integer 0–127), `duration` (seconds quantized to 50 ms), `velocity` (MIDI velocity), and `delta_time` (seconds from the preceding note onset, quantized to 50 ms). The model embeds pitch/duration/velocity/timing with vocabulary sizes 128/16/8/16, predicts four heads, and also receives target/future-endpoint pitch and tempo. Dataset windows default to length 128 for the baseline; the larger experiment used 256. Training supplies a constant 100 BPM condition. No instrument, track, beat/bar, chord, key, or segment feature is currently fed to the LSTM. The AAM annotations could support future beat-aware timing, chord/key harmony conditioning, instrument-aware modeling, and section/continuation context, after careful mapping and controlled ablation.

## I. Preprocessing/model impact

The new ARFF and FLAC archives are cataloged and joined by source ID only. The current LSTM preprocessing uses MIDI events alone. Adding annotation-derived conditioning would require a separately designed feature pipeline and a retraining plan; this audit intentionally makes no such change. All MIDI-member and processed JSONL counts match exactly; window counts depend on sequence length and split. The audit does not extract archives or rewrite the corpus.

## J. Training recommendation (audit-only)

Keep the stable LSTM 3×512 checkpoint as the baseline and keep the 4×768 checkpoint experimental, as requested in prior project context. Use the existing 80/10/10 composition-grouped split and report validation and held-out test metrics. Do not use audio mixes as inputs for the current symbolic LSTM. Before a future annotation-conditioned experiment, define a leakage-safe composition-level join, document annotation coverage/missingness, and compare against the current MIDI-only baseline. No retraining was run for this audit.

## K. Validation issues

There are 0 archive/MIDI/FLAC naming or parse issues and 0 processed JSONL parse issues. Up to 100 representative issues are listed in each JSON report. No files were deleted or rewritten.

## Requested A–M answers

**A. Are all existing AAM MIDI files being used?** Yes. All 22,574 raw MIDI members have processed JSONL files; zero are unmapped, invalid, or too short.

**B. How many MIDI files are available?** 22,574 across the three MIDI ZIPs; per-archive counts and hashes are listed above and in the manifest.

**C. How many are valid?** 22,574 parse successfully; 0 are invalid and 0 are empty.

**D. How many training sequences?** For baseline length 128: train 55,879, validation 7,030, test 6,715, total 69,624. For length 256: total 42,404.

**E. Are annotations correctly mapped to MIDI?** Yes by numeric AAM source/composition ID: all 3,000 IDs occur in both and missing-ID counts are zero. Each ID maps to multiple MIDI files and three ARFF files, so matching is composition-level.

**F. What annotation information is available?** Beat timestamps/bar/quarter/chord; onset timestamps and 31 instrument-specific onset token columns; segment start, mark, tempo, key, instrument list, and generator. No explicit duration or velocity columns were found.

**G. Are audio mixes correctly mapped?** Yes: all 3,000 named FLAC IDs map to the same source IDs as MIDI and annotations. They are 44.1 kHz mono per FLAC metadata and remain excluded from LSTM data.

**H. Are overlaps present in source MIDI?** Yes. The parsed/quantized streams show overlaps in 21,929 MIDI files, including 1,036,175 overlapping same-pitch note pairs. Polyphony may be legitimate; track identity is lost in the current representation.

**I. Are overlaps introduced by preprocessing?** Preprocessing does not remove overlaps and preserves the same event count per source; it rounds timing to 50 ms, which can alter edge cases. This audit did not compare all raw unquantized intervals against quantized intervals, so it cannot attribute every overlap to source versus quantization.

**J. Are overlaps introduced by AI continuation/inference?** Static inspection found the generated local timestamps shifted by the computed continuation boundary and same-AI-line note gaps clamped to prior duration, plus timeline validation/sanitization. No shared-code offset defect is apparent. Controlled runtime GPU cases at 0.5/1/2 seconds were not run in this environment.

**K. Does the representation preserve timing correctly?** It preserves approximate onset gaps and durations. 50 ms rounding and categorical buckets are lossy; track/instrument identity, tempo map, and beat/chord/key context are discarded.

**L. Is there source-level train/test leakage?** No. Composition-ID groups are disjoint across train/validation/test (2,400/300/300 IDs); zero source IDs cross splits.

**M. What should change before retraining?** First capture the requested same-composition/same-seed 0.5/1/2-second timeline diagnostics on the user's installed GPU checkpoint. Retain the 3×512 MIDI-only model as baseline and 4×768 as experimental. If a future model change is approved, evaluate annotation-conditioned beat/chord/key/instrument features with grouped splits and controlled ablations. Do not add audio to the current LSTM.

See `aam_dataset_manifest.json` for per-source mappings and `aam_dataset_stats.json` for machine-readable totals.
