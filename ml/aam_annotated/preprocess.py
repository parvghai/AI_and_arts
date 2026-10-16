"""Build source-grouped, annotation-aware AAM MIDI features without using audio."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.dataset import split_files
from ml.preprocessing import (DURATION_EDGES, VELOCITY_EDGES, parse_midi_bytes)

ROOT = Path(__file__).resolve().parents[2]
ARFF_ATTR = re.compile(r"^\s*@attribute\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.+?)\s*$", re.I)
MIDI_NAME = re.compile(r"^(\d{4})_(.+)\.midi?$", re.I)
ARFF_NAME = re.compile(r"^(\d{4})_(beatinfo|onsets|segments)\.arff$", re.I)
NUMERIC_TOKEN = re.compile(r"\+(-?\d+)")
CATEGORICAL_FIELDS = ("pitch", "duration", "velocity", "instrument", "chord", "key")
NUMERIC_FIELDS = ("delta_beats_log_norm", "beat_phase_sin", "beat_phase_cos",
                  "within_beat", "tempo_norm", "onset_progress", "duration_norm")
SPLIT_NAMES = ("train", "validation", "test")


def _attribute(line: str):
    match = ARFF_ATTR.match(line)
    if not match:
        return None
    name = next(value for value in match.groups()[:3] if value is not None).strip()
    return name, match.group(4).strip()


def parse_arff(payload: bytes) -> tuple[list[str], list[list[str]], int]:
    """Parse the actual AAM ARFF dialect, including beatinfo without @DATA."""
    attributes: list[str] = []
    rows: list[list[str]] = []
    malformed = 0
    text = payload.decode("utf-8-sig", "replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or stripped.lower() == "@data" or stripped.startswith("@relation"):
            continue
        attr = _attribute(line)
        if attr:
            attributes.append(attr[0])
            continue
        if stripped.startswith("@") or not attributes:
            continue
        try:
            record = next(csv.reader([line], delimiter=",", quotechar="'", skipinitialspace=True))
        except csv.Error:
            malformed += 1
            continue
        if len(record) != len(attributes):
            malformed += 1
            continue
        rows.append([value.strip().strip("'\"") for value in record])
    return attributes, rows, malformed


def _open_archives(raw: Path, pattern: str, id_range_re: str):
    result = {}
    for path in sorted(raw.glob(pattern)):
        match = re.search(id_range_re, path.name)
        if not match:
            raise ValueError(f"Cannot infer AAM ID range from {path.name}")
        result[int(match.group(1))] = path
    if not result:
        raise FileNotFoundError(f"No archives matching {pattern} in {raw}")
    return result


def _source_splits(processed: Path, seed: int) -> dict[int, str]:
    """Reuse the baseline's exact grouped split instead of inventing a new split."""
    files = sorted(processed.glob("*.jsonl"))
    train, validation, test = split_files(files, seed)
    assignment: dict[int, str] = {}
    for split_name, paths in zip(SPLIT_NAMES, (train, validation, test)):
        for path in paths:
            match = re.match(r"^(\d{3,})[_-]", path.stem)
            if not match:
                continue
            source_id = int(match.group(1))
            previous = assignment.setdefault(source_id, split_name)
            if previous != split_name:
                raise RuntimeError(f"Source ID {source_id} crosses baseline splits")
    if not assignment:
        raise RuntimeError(f"No baseline AAM source-ID split found in {processed}")
    return assignment


def _read_annotations(archive: zipfile.ZipFile, source_id: int) -> dict:
    result = {}
    for kind in ("beatinfo", "onsets", "segments"):
        member = f"{source_id:04d}_{kind}.arff"
        fields, rows, malformed = parse_arff(archive.read(member))
        if malformed:
            raise ValueError(f"Malformed ARFF rows in {member}: {malformed}")
        result[kind] = (fields, rows)
    return result


def _annotation_vocab(ann_archives: dict[int, Path], train_ids: set[int],
                      midi_labels: dict[int, set[str]]) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = {"instrument": set(), "chord": set(), "key": set()}
    for sid in sorted(train_ids):
        values["instrument"].update(midi_labels.get(sid, set()))
    for start, path in sorted(ann_archives.items()):
        stop = start + 999
        relevant = [sid for sid in train_ids if start <= sid <= stop]
        if not relevant:
            continue
        with zipfile.ZipFile(path) as archive:
            for sid in relevant:
                annotations = _read_annotations(archive, sid)
                bfields, beats = annotations["beatinfo"]
                chord_col = bfields.index("Chord name")
                values["chord"].update(row[chord_col] for row in beats if row[chord_col])
                sfields, segments = annotations["segments"]
                key_col = sfields.index("Key")
                values["key"].update(row[key_col] for row in segments if row[key_col])
    return {name: {value: index + 1 for index, value in enumerate(sorted(items))}
            for name, items in values.items()}


def _beat_context(beats: list[tuple[float, int, int, str]], times: list[float], t: float,
                  beats_per_bar: int):
    i = max(0, bisect.bisect_right(times, t) - 1)
    current = beats[i]
    next_time = beats[i + 1][0] if i + 1 < len(beats) else None
    if next_time is not None and next_time > current[0]:
        fraction = min(1.0, max(0.0, (t - current[0]) / (next_time - current[0])))
    elif i > 0 and current[0] > beats[i - 1][0]:
        interval = current[0] - beats[i - 1][0]
        fraction = min(1.0, max(0.0, (t - current[0]) / interval))
    else:
        fraction = 0.0
    phase = ((current[2] - 1) + fraction) / max(1, beats_per_bar)
    phase %= 1.0
    return current[3], math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase), fraction


def _segment_context(segments: list[tuple[float, float, str]], times: list[float], t: float,
                     fallback_tempo: float):
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        return fallback_tempo, ""
    row = segments[i]
    return (row[1] if row[1] > 0 else fallback_tempo), row[2]


def _align_events(events, instrument: str, onset_fields: list[str], onset_rows: list[list[str]],
                  tolerance: float):
    if instrument == "MIXED_OR_UNSPECIFIED":
        columns = [i for i, field in enumerate(onset_fields) if field.startswith("Onset events of ")]
    else:
        field = f"Onset events of {instrument}"
        columns = [onset_fields.index(field)] if field in onset_fields else []
    attacks = []
    for row in onset_rows:
        timestamp = float(row[0])
        for column in columns:
            attacks.extend((timestamp, int(pitch)) for pitch in NUMERIC_TOKEN.findall(row[column]))
    attacks.sort(key=lambda pair: (pair[0], pair[1]))
    local_times, cursor = [], 0.0
    for event in events:
        cursor += float(event.delta_time)
        local_times.append(cursor)
    midi_pairs = sorted(zip(local_times, (int(event.pitch) for event in events)), key=lambda pair: (pair[0], pair[1]))
    result = {"status": "unaligned", "annotation_attacks": len(attacks),
              "midi_notes": len(midi_pairs), "offset_seconds": None,
              "max_residual_seconds": None}
    if len(attacks) != len(midi_pairs) or not attacks:
        result["reason"] = "attack_count_mismatch_or_no_attacks"
        return result, local_times
    if [pitch for _, pitch in attacks] != [pitch for _, pitch in midi_pairs]:
        result["reason"] = "pitch_sequence_mismatch"
        return result, local_times
    offsets = [at - mt for (at, _), (mt, _) in zip(attacks, midi_pairs)]
    offset = statistics.median(offsets)
    if abs(offset) <= .001:
        offset = 0.0
    residual = max(abs(value - offset) for value in offsets)
    result["offset_seconds"] = round(offset, 6)
    result["max_residual_seconds"] = round(residual, 6)
    result["status"] = "aligned" if residual <= tolerance else "unaligned"
    if residual > tolerance:
        result["reason"] = "event_time_residual_exceeds_tolerance"
    return result, local_times


def _make_features(events, local_times, alignment, annotations, instrument_id,
                   vocab, composition_end: float, beats_per_bar: int):
    bfields, b_rows = annotations["beatinfo"]
    sfields, s_rows = annotations["segments"]
    beat_time = bfields.index("Start time in seconds")
    bar_col, quarter_col, chord_col = bfields.index("Bar count"), bfields.index("Quarter count"), bfields.index("Chord name")
    beats = [(float(row[beat_time]), int(float(row[bar_col])), int(float(row[quarter_col])), row[chord_col])
             for row in b_rows]
    beats.sort()
    beat_times = [row[0] for row in beats]
    seg_time, tempo_col, key_col = (sfields.index("Start time in seconds"),
                                    sfields.index("Tempo"), sfields.index("Key"))
    segments = [(float(row[seg_time]), float(row[tempo_col] or 0), row[key_col])
                for row in s_rows if row[key_col] or (row[tempo_col] and float(row[tempo_col]) > 0)]
    segments.sort()
    segment_times = [row[0] for row in segments]
    beat_intervals = [b[0] - a[0] for a, b in zip(beats, beats[1:]) if b[0] > a[0]]
    fallback_tempo = 60.0 / statistics.median(beat_intervals) if beat_intervals else 120.0
    fallback_tempo = min(240.0, max(30.0, fallback_tempo))
    end = max(composition_end, beats[-1][0] if beats else 0.0, segments[-1][0] if segments else 0.0, .01)
    cats, nums, cursor, previous_abs = [], [], 0.0, None
    unknown = not (alignment["status"] == "aligned")
    for event, local_t in zip(events, local_times):
        absolute_t = local_t + (alignment["offset_seconds"] or 0.0)
        delta_seconds = max(0.0, float(event.delta_time))
        tempo, key = _segment_context(segments, segment_times, absolute_t, fallback_tempo)
        if unknown:
            tempo, key = fallback_tempo, ""
        if unknown or not beats:
            chord, phase_sin, phase_cos, within_beat = "", 0.0, 1.0, 0.0
        else:
            chord, phase_sin, phase_cos, within_beat = _beat_context(beats, beat_times, absolute_t, beats_per_bar)
        chord_id = vocab["chord"].get(chord, 0) if not unknown else 0
        key_id = vocab["key"].get(key, 0) if not unknown else 0
        duration_token = min(len(DURATION_EDGES), next(
            (i for i, edge in enumerate(DURATION_EDGES) if event.duration <= edge), len(DURATION_EDGES)))
        velocity_token = min(len(VELOCITY_EDGES), next(
            (i for i, edge in enumerate(VELOCITY_EDGES) if event.velocity <= edge), len(VELOCITY_EDGES)))
        event_codes = (min(127, max(0, int(event.pitch))), duration_token,
                       min(7, max(0, velocity_token)))
        delta_beats = delta_seconds * tempo / 60.0
        delta_norm = min(1.0, math.log1p(min(16.0, delta_beats)) / math.log1p(16.0))
        cats.append((*event_codes, instrument_id, chord_id, key_id))
        nums.append((delta_norm, phase_sin, phase_cos, within_beat,
                     min(1.0, max(0.0, tempo / 240.0)),
                     min(1.0, max(0.0, absolute_t / end)) if not unknown else 0.0,
                     min(1.0, max(0.0, float(event.duration) / 8.0))))
        previous_abs = absolute_t
    return np.asarray(cats, dtype=np.uint16), np.asarray(nums, dtype=np.float32), end


def build_dataset(raw: Path, processed: Path, output: Path, stats_path: Path,
                  seed: int = 42, sequence_length: int = 256,
                  alignment_tolerance: float = .06) -> dict:
    midi_archives = _open_archives(raw, "*-midis.zip", r"^(\d{4})-")
    ann_archives = _open_archives(raw, "*-annotations*.zip", r"^(\d{4})-")
    assignment = _source_splits(processed, seed)
    ids_from_baseline = set(assignment)
    midi_paths: dict[int, list[tuple[str, str]]] = defaultdict(list)
    midi_labels: dict[int, set[str]] = defaultdict(set)
    archive_owner = {}
    for start, path in sorted(midi_archives.items()):
        archive_owner[path] = start
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                match = MIDI_NAME.match(Path(info.filename).name)
                if not match:
                    continue
                sid, label = int(match.group(1)), match.group(2)
                if sid not in ids_from_baseline:
                    continue
                instrument = "MIXED_OR_UNSPECIFIED" if label.lower() == "demo" else label
                midi_paths[sid].append((path.name, info.filename))
                midi_labels[sid].add(instrument)
    if ids_from_baseline - set(midi_paths):
        raise RuntimeError(f"No raw MIDI found for source IDs: {sorted(ids_from_baseline-set(midi_paths))[:10]}")
    train_ids = {sid for sid, split in assignment.items() if split == "train"}
    vocab = _annotation_vocab(ann_archives, train_ids, midi_labels)
    vocab_sizes = {key: len(value) + 1 for key, value in vocab.items()}
    output.mkdir(parents=True, exist_ok=True)
    raw_data = output / "data"
    raw_data.mkdir(parents=True, exist_ok=True)
    index_records = {split: [] for split in SPLIT_NAMES}
    event_counts = Counter()
    token_counts = {split: {feature: Counter() for feature in CATEGORICAL_FIELDS}
                    for split in SPLIT_NAMES}
    valid_files, invalid_files, alignment_counts = Counter(), [], Counter()
    overlap_notes, duplicate_notes, negative_durations, negative_gaps = Counter(), Counter(), 0, 0
    total_by_split = Counter()
    raw_streams = {}
    for split in SPLIT_NAMES:
        raw_streams[split] = (open(raw_data / f"{split}_categorical.bin", "wb"),
                              open(raw_data / f"{split}_numeric.bin", "wb"))
    manifest_sources = []
    midi_zip_cache = {p: zipfile.ZipFile(p) for p in midi_archives.values()}
    ann_zip_cache = {p: zipfile.ZipFile(p) for p in ann_archives.values()}
    cursors = Counter()
    try:
        for sid in sorted(midi_paths):
            ann_start = max(start for start in ann_archives if start <= sid)
            annotations = _read_annotations(ann_zip_cache[ann_archives[ann_start]], sid)
            onset_fields, onset_rows = annotations["onsets"]
            bfields, b_rows = annotations["beatinfo"]
            # Infer the time signature from observed AAM quarter-count spans, not MIDI defaults.
            quarter_col, bar_col = bfields.index("Quarter count"), bfields.index("Bar count")
            maxima = defaultdict(int)
            for row in b_rows:
                maxima[int(float(row[bar_col]))] = max(maxima[int(float(row[bar_col]))], int(float(row[quarter_col])))
            beats_per_bar = Counter(maxima.values()).most_common(1)[0][0] if maxima else 4
            beats_per_bar = max(1, beats_per_bar)
            sfields, s_rows = annotations["segments"]
            onsets_col = onset_fields.index("Onset time in seconds")
            composition_end = max([float(row[onsets_col]) for row in onset_rows] +
                                  [float(row[bfields.index("Start time in seconds")]) for row in b_rows] +
                                  [float(row[sfields.index("Start time in seconds")]) for row in s_rows] + [.01])
            source_entry = {"source_id": sid, "split": assignment[sid], "midi_files": [],
                            "annotation_files": {kind: f"{sid:04d}_{kind}.arff" for kind in annotations}}
            for archive_name, member in sorted(midi_paths[sid]):
                archive_path = next(path for path in midi_archives.values() if path.name == archive_name)
                try:
                    events = parse_midi_bytes(midi_zip_cache[archive_path].read(member))
                    if len(events) < 3:
                        raise ValueError("fewer than three note events")
                    filename_label = Path(member).stem.split("_", 1)[1]
                    instrument = "MIXED_OR_UNSPECIFIED" if filename_label.lower() == "demo" else filename_label
                    align, local_times = _align_events(events, instrument, onset_fields, onset_rows, alignment_tolerance)
                    alignment_counts[align["status"]] += 1
                    instrument_id = vocab["instrument"].get(instrument, 0)
                    cats, nums, actual_end = _make_features(events, local_times, align, annotations,
                                                            instrument_id, vocab, composition_end, beats_per_bar)
                    for feature_index, feature_name in enumerate(CATEGORICAL_FIELDS):
                        unique_tokens, unique_counts = np.unique(cats[:, feature_index], return_counts=True)
                        token_counts[assignment[sid]][feature_name].update(
                            {int(token): int(count) for token, count in zip(unique_tokens, unique_counts)})
                    start = cursors[assignment[sid]]
                    cat_stream, num_stream = raw_streams[assignment[sid]]
                    cats.tofile(cat_stream); nums.tofile(num_stream)
                    count = len(events)
                    cursors[assignment[sid]] += count
                    total_by_split[assignment[sid]] += count
                    event_counts[member] = count
                    # Preserve overlaps; count same-line interval intersections for the audit.
                    onsets = np.cumsum([float(e.delta_time) for e in events])
                    ends = onsets + np.asarray([max(0.0, float(e.duration)) for e in events])
                    active_ends: list[float] = []
                    from heapq import heappop, heappush
                    local_overlap = 0
                    for note_start, note_end in zip(onsets, ends):
                        while active_ends and active_ends[0] <= note_start:
                            heappop(active_ends)
                        local_overlap += len(active_ends)
                        heappush(active_ends, float(note_end))
                    overlap_notes[assignment[sid]] += local_overlap
                    file_record = {"source_id": sid, "archive": archive_name, "member": member,
                                   "split": assignment[sid], "instrument": instrument,
                                   "start": start, "count": count,
                                   "alignment": align, "absolute_onset_origin_seconds": align["offset_seconds"],
                                   "preserved_overlapping_note_pairs": local_overlap,
                                   "validation_status": "valid"}
                    index_records[assignment[sid]].append(file_record)
                    source_entry["midi_files"].append(file_record)
                    valid_files[assignment[sid]] += 1
                except Exception as exc:
                    invalid_files.append({"source_id": sid, "member": member,
                                          "error": f"{type(exc).__name__}: {exc}"})
                    source_entry["midi_files"].append({"archive": archive_name, "member": member,
                                                        "split": assignment[sid], "validation_status": "invalid",
                                                        "error": str(exc)})
            manifest_sources.append(source_entry)
    finally:
        for streams in raw_streams.values():
            for stream in streams:
                stream.close()
        for archive in (*midi_zip_cache.values(), *ann_zip_cache.values()):
            archive.close()

    for split in SPLIT_NAMES:
        n = int(total_by_split[split])
        (raw_data / f"{split}_categorical.shape.json").write_text(json.dumps({"events": n, "features": len(CATEGORICAL_FIELDS), "dtype": "uint16"}, indent=2), encoding="utf-8")
        (raw_data / f"{split}_numeric.shape.json").write_text(json.dumps({"events": n, "features": len(NUMERIC_FIELDS), "dtype": "float32"}, indent=2), encoding="utf-8")
        with (raw_data / f"{split}_index.jsonl").open("w", encoding="utf-8") as stream:
            for row in index_records[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    mappings = {"unknown_id": 0, "categorical_features": CATEGORICAL_FIELDS,
                "numeric_features": NUMERIC_FIELDS, "vocab_sizes": {
                    "pitch": 128, "duration": len(DURATION_EDGES) + 1,
                    "velocity": len(VELOCITY_EDGES) + 1, **vocab_sizes},
                "values": vocab, "alignment": {
                    "onset_annotations_are_alignment_validation_only": True,
                    "timestamp_shift": "median(annotation_attack_time - relative_MIDI_onset) per matching MIDI member",
                    "instrument_alignment_rule": "instrument members match their onset column; Demo matches aggregate onset columns",
                    "max_pitch_time_residual_seconds": alignment_tolerance}}
    mappings_path = output / "feature_mappings.json"
    mappings_path.write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {"dataset": "AAM MIDI + AAM annotations v1.1.0", "raw_dir": str(raw),
              "processed_midi_dir": str(processed), "output_dir": str(output),
              "split_seed": seed, "split_source": "existing MIDI-only split_files source-ID assignments",
              "sequence_length": sequence_length, "alignment_tolerance_seconds": alignment_tolerance,
              "audio_used": False, "onset_annotations_used_as_pitch_input": False,
              "invalid_timing_policy": "reject malformed MIDI/ARFF; retain valid polyphony; mark annotation context unknown when onset alignment fails",
              "timing": {"MIDI event onset": "relative MIDI event time + validated source onset offset",
                         "beat alignment": "continuous interpolation between beatinfo timestamps",
                         "delta feature": "log-normalized inter-onset interval in beats",
                         "absolute onset": "normalized progress through annotation timeline"},
              "categorical_features": CATEGORICAL_FIELDS, "numeric_features": NUMERIC_FIELDS}
    (output / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    file_windows = {}
    for split in SPLIT_NAMES:
        file_windows[split] = sum(max(0, (row["count"] - 3) // sequence_length + 1)
                                  for row in index_records[split] if row.get("validation_status") == "valid")
    per_split = {}
    for split in SPLIT_NAMES:
        per_split[split] = {"source_groups": len({r["source_id"] for r in manifest_sources if r["split"] == split}),
                            "midi_files_valid": valid_files[split], "events": total_by_split[split],
                            "sequences": file_windows[split],
                            "target_tokens_per_epoch": file_windows[split] * sequence_length,
                            "preserved_overlap_pairs": overlap_notes[split],
                            "annotation_aligned_midi_files": sum(1 for r in index_records[split]
                                                                 if r["alignment"]["status"] == "aligned"),
                            "annotation_unaligned_midi_files": sum(1 for r in index_records[split]
                                                                   if r["alignment"]["status"] != "aligned")}
    disk_bytes = sum(path.stat().st_size for path in raw_data.iterdir() if path.is_file())
    stats = {"dataset": "AAM MIDI + AAM annotations v1.1.0; audio excluded",
             "seed": seed, "sequence_length": sequence_length,
             "source_id_split_leaks": 0, "source_groups_total": len(assignment),
             "valid_midi_files": sum(valid_files.values()), "invalid_midi_files": len(invalid_files),
             "total_events": sum(total_by_split.values()), "malformed_annotation_rows": 0,
             "onset_alignment_tolerance_seconds": alignment_tolerance,
             "alignment_status_counts": dict(alignment_counts), "splits": per_split,
             "events_per_midi": {"min": min(event_counts.values(), default=0),
                                 "median": statistics.median(event_counts.values()) if event_counts else 0,
                                 "mean": statistics.mean(event_counts.values()) if event_counts else 0,
                                 "max": max(event_counts.values(), default=0)},
             "feature_vocabulary_sizes": mappings["vocab_sizes"],
             "feature_value_counts": {key: len(value) for key, value in vocab.items()},
             "categorical_token_frequency_by_split": {
                 split: {feature: {str(token): count for token, count in sorted(counter.items())}
                         for feature, counter in features.items()}
                 for split, features in token_counts.items()},
             "categorical_feature_order": CATEGORICAL_FIELDS,
             "numeric_feature_order": NUMERIC_FIELDS,
             "tensor_shapes_per_batch": {"categorical_input": ["B", sequence_length, len(CATEGORICAL_FIELDS)],
                                          "numeric_input": ["B", sequence_length, len(NUMERIC_FIELDS)],
                                          "categorical_target": ["B", sequence_length, 3],
                                          "timing_target": ["B", sequence_length]},
             "disk_bytes": disk_bytes,
             "invalid_examples": invalid_files[:100]}
    output_manifest = {"dataset": "AAM MIDI + annotations v1.1.0", "audio_used": False,
                       "source_groups": manifest_sources,
                       "split_assignments": {str(sid): split for sid, split in sorted(assignment.items())},
                       "feature_mapping": str(mappings_path.relative_to(output)),
                       "preprocessing_config": str((output / "preprocessing_config.json").relative_to(output))}
    (output / "dataset_manifest.json").write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Persist a few real encoded examples for review, not arbitrary integerized labels.
    samples = []
    for split in SPLIT_NAMES:
        for row in index_records[split][:2]:
            if row["count"] < 3:
                continue
            cat_path = raw_data / f"{split}_categorical.bin"
            num_path = raw_data / f"{split}_numeric.bin"
            cat = np.memmap(cat_path, dtype=np.uint16, mode="r", shape=(int(total_by_split[split]), len(CATEGORICAL_FIELDS)))
            num = np.memmap(num_path, dtype=np.float32, mode="r", shape=(int(total_by_split[split]), len(NUMERIC_FIELDS)))
            a, b = row["start"], min(row["start"] + 5, row["start"] + row["count"])
            samples.append({"split": split, "source_id": row["source_id"], "member": row["member"],
                            "alignment": row["alignment"], "categorical_features": CATEGORICAL_FIELDS,
                            "numeric_features": NUMERIC_FIELDS, "categorical": cat[a:b].tolist(),
                            "numeric": np.round(num[a:b], 6).tolist()})
            del cat, num
    (output / "sample_encoded_sequences.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/aam_annotated")
    parser.add_argument("--stats", type=Path, default=ROOT / "reports/aam_annotated_dataset_stats.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--alignment-tolerance", type=float, default=.06)
    args = parser.parse_args()
    result = build_dataset(args.raw, args.processed, args.output, args.stats,
                           args.seed, args.sequence_length, args.alignment_tolerance)
    print(json.dumps({"valid_midi_files": result["valid_midi_files"],
                      "invalid_midi_files": result["invalid_midi_files"],
                      "total_events": result["total_events"],
                      "alignment": result["alignment_status_counts"],
                      "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
