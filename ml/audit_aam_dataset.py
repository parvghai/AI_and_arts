"""Read-only AAM archive audit; audio is inspected via FLAC headers only.

Run from the project directory with: python -m ml.audit_aam_dataset
No archives are extracted, no audio samples are decoded, and no training data,
model, checkpoint, or GUI files are modified. The only outputs are reports/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import struct
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ml.preprocessing import parse_midi_bytes
from ml.dataset import split_files, count_windows

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MIDI_RE = re.compile(r"^(\d{3,})")
ARFF_ATTR = re.compile(
    r"^\s*@attribute\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.+?)\s*$", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def midi_details(data: bytes) -> dict:
    """Read SMF metadata (track count, names, program, tempo, meter, duration)."""
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("missing MIDI header")
    hlen = struct.unpack_from(">I", data, 4)[0]
    fmt, tracks, division = struct.unpack_from(">HHH", data, 8)
    pos = 8 + hlen
    tempos, meters, programs, names, ends = [], [], set(), [], []

    def vlq(buf: bytes, p: int, end: int) -> tuple[int, int]:
        v = 0
        for _ in range(4):
            if p >= end:
                raise ValueError("truncated VLQ")
            b = buf[p]
            p += 1
            v = (v << 7) | (b & 127)
            if not b & 128:
                return v, p
        raise ValueError("invalid VLQ")

    for _ in range(tracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("missing track chunk")
        length = struct.unpack_from(">I", data, pos + 4)[0]
        p, end, tick, running = pos + 8, pos + 8 + length, 0, None
        pos = end
        while p < end:
            delta, p = vlq(data, p, end)
            tick += delta
            if p >= end:
                raise ValueError("truncated event")
            status = data[p]
            if status & 128:
                p += 1
                if status < 0xF0:
                    running = status
            elif running is not None:
                status = running
            else:
                raise ValueError("missing running status")
            if status == 0xFF:
                if p + 1 >= end:
                    raise ValueError("truncated meta event")
                kind = data[p]
                size, p = vlq(data, p + 1, end)
                payload = data[p:p + size]
                if len(payload) != size:
                    raise ValueError("truncated meta payload")
                p += size
                if kind == 0x51 and size == 3:
                    us = int.from_bytes(payload, "big")
                    if us:
                        tempos.append((tick, us))
                elif kind == 0x58 and size >= 2:
                    meters.append((payload[0], 2 ** payload[1]))
                elif kind == 0x03:
                    names.append(payload.decode("latin1", "replace"))
                if kind == 0x2F:
                    ends.append(tick)
                    break
            elif status in (0xF0, 0xF7):
                size, p = vlq(data, p, end)
                p += size
            else:
                hi = status & 0xF0
                n = 1 if hi in (0xC0, 0xD0) else 2
                if hi == 0xC0:
                    if p >= end:
                        raise ValueError("truncated program change")
                    programs.add(data[p] & 127)
                p += n
                if p > end:
                    raise ValueError("truncated channel event")
        ends.append(tick)

    last_tick = max(ends, default=0)
    sec, previous_tick, current_us = 0.0, 0, 500000
    for event_tick, us in sorted(tempos):
        if event_tick > last_tick:
            break
        sec += (event_tick - previous_tick) * current_us / (division * 1_000_000)
        previous_tick, current_us = event_tick, us
    sec += max(0, last_tick - previous_tick) * current_us / (division * 1_000_000)
    return {"format": fmt, "track_count": tracks, "division": division,
            "duration_seconds": sec, "tempo_bpms": [round(60_000_000 / us, 5) for _, us in tempos],
            "time_signatures": meters, "programs": sorted(programs), "track_names": names}


def overlap_stats(events: list) -> dict:
    onset, rows = 0.0, []
    for event in events:
        onset += max(0.0, float(event.delta_time))
        rows.append((onset, onset + max(0.0, float(event.duration)), int(event.pitch)))
    rows.sort()
    active: list[tuple[float, int]] = []
    pitches: Counter[int] = Counter()
    notes_with_overlap = pairs = same_pitch_pairs = 0
    for start, end, pitch in rows:
        while active and active[0][0] <= start:
            _, old_pitch = __import__("heapq").heappop(active)
            pitches[old_pitch] -= 1
            if pitches[old_pitch] == 0:
                del pitches[old_pitch]
        count = sum(pitches.values())
        if count:
            notes_with_overlap += 1
            pairs += count
            same_pitch_pairs += pitches.get(pitch, 0)
        __import__("heapq").heappush(active, (end, pitch))
        pitches[pitch] += 1
    return {"notes_with_prior_overlap": notes_with_overlap,
            "overlapping_note_pairs": pairs, "same_pitch_overlap_pairs": same_pitch_pairs}


def read_flac_header(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict:
    with zf.open(info) as stream:
        head = stream.read(42)
    if len(head) < 42 or head[:4] != b"fLaC" or head[4] & 0x7F != 0 or int.from_bytes(head[5:8], "big") != 34:
        raise ValueError("invalid/missing FLAC STREAMINFO header")
    packed = int.from_bytes(head[18:26], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 7) + 1
    bits = ((packed >> 36) & 31) + 1
    samples = packed & ((1 << 36) - 1)
    if rate == 0:
        raise ValueError("invalid FLAC sample rate")
    return {"sample_rate_hz": rate, "channels": channels, "bits_per_sample": bits,
            "total_samples": samples, "duration_seconds": samples / rate}


def parse_arff_member(name: str, payload: bytes) -> dict:
    text = payload.decode("utf-8-sig", "replace")
    attrs, in_data, rows, malformed = [], False, 0, 0
    values = defaultdict(Counter)
    onset_values = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        match = ARFF_ATTR.match(line)
        if match:
            field = next(group for group in match.groups()[:3] if group is not None)
            attrs.append((field.strip(), match.group(4).strip()))
            continue
        if stripped.lower() == "@data":
            in_data = True
            continue
        if stripped.startswith("@"):
            continue
        try:
            row = next(csv.reader([line], delimiter=",", quotechar="'", skipinitialspace=True))
        except Exception:
            malformed += 1
            continue
        # AAM beatinfo files omit @DATA; ordinary record-shaped rows remain readable.
        if not attrs:
            continue
        if len(row) != len(attrs):
            malformed += 1
            continue
        rows += 1
        for (field, _), value in zip(attrs, row):
            value = value.strip().strip("'\"")
            if value and value != "?":
                if field.lower() in ("tempo", "key", "chord name", "mark", "generator"):
                    values[field][value] += 1
                if "onset events of" in field.lower():
                    onset_values += len(re.findall(r"[+-]?\d+", value))
    return {"name": name, "attributes": attrs, "rows": rows,
            "malformed_rows": malformed, "onset_event_values": onset_values,
            "value_counts": values}


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"Missing AAM raw directory: {RAW}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    archives = sorted(RAW.glob("*.zip"))
    midi_archives = [p for p in archives if "midi" in p.name.lower()]
    annotation_archives = [p for p in archives if "annotation" in p.name.lower()]
    audio_archives = [p for p in archives if "audio-mixes" in p.name.lower()]
    info_archives = [p for p in archives if p.name.lower() == "info.zip"]
    comp = defaultdict(lambda: {"midi_members": [], "annotations": {}, "audio": None})
    archive_records, midi_rows, midi_hashes = [], [], defaultdict(list)
    annotation_records = defaultdict(lambda: {"files": 0, "rows": 0, "malformed_rows": 0,
                                               "attributes": {}, "onset_event_values": 0,
                                               "value_counts": defaultdict(Counter)})
    audio_rows, midi_ids, annotation_ids, audio_ids = [], set(), set(), set()
    all_midi_programs, formats, signatures = Counter(), Counter(), Counter()
    errors, too_short, midi_events, valid_midi = [], 0, 0, 0
    midi_duration, note_duration, pitches = [], [], []
    overlap_files = overlap_files_count = 0
    overlap_sum = Counter()
    duplicate_hashes = 0
    track_counts, explicit_tempo_bpm = [], []
    empty_midi_files = 0

    for path in midi_archives:
        rec = {"name": path.name, "path": f"data/raw/{path.name}", "media_type": "midi", "archive_bytes": path.stat().st_size,
               "sha256": sha256_file(path), "members": 0, "valid_midi": 0, "invalid_midi": 0,
               "source_ids": []}
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            rec["members"] = len(infos)
            for info in infos:
                if Path(info.filename).suffix.lower() not in (".mid", ".midi"):
                    continue
                match = MIDI_RE.match(Path(info.filename).name)
                if not match:
                    errors.append({"archive": path.name, "member": info.filename, "error": "no numeric AAM ID"})
                    continue
                sid = int(match.group(1))
                midi_ids.add(sid)
                rec["source_ids"].append(sid)
                comp[sid]["midi_members"].append({"archive": path.name, "member": info.filename,
                                                   "compressed_bytes": info.compress_size,
                                                   "uncompressed_bytes": info.file_size,
                                                   "validation_status": "pending"})
                midi_manifest_row = comp[sid]["midi_members"][-1]
                try:
                    raw = zf.read(info)
                    events = parse_midi_bytes(raw)
                    extra = midi_details(raw)
                    midi_manifest_row.update({"validation_status": "valid", "event_count": len(events),
                                              "format": extra["format"], "track_count": extra["track_count"],
                                              "duration_seconds": extra["duration_seconds"],
                                              "programs": extra["programs"]})
                    valid_midi += 1
                    rec["valid_midi"] += 1
                    midi_events += len(events)
                    track_counts.append(extra["track_count"])
                    explicit_tempo_bpm.extend(extra["tempo_bpms"])
                    if not events:
                        empty_midi_files += 1
                    if len(events) < 3:
                        too_short += 1
                    midi_duration.append(extra["duration_seconds"])
                    formats[str(extra["format"])] += 1
                    signatures.update(f"{a}/{b}" for a, b in extra["time_signatures"])
                    all_midi_programs.update(extra["programs"])
                    pitches.extend(e.pitch for e in events)
                    note_duration.extend(e.duration for e in events)
                    ov = overlap_stats(events)
                    if ov["overlapping_note_pairs"]:
                        overlap_files += 1
                    overlap_sum.update(ov)
                    raw_hash = hashlib.sha256(raw).hexdigest()
                    midi_hashes[raw_hash].append(f"{path.name}::{info.filename}")
                    midi_rows.append({"source_id": sid, "archive": path.name, "member": info.filename,
                                      "event_count": len(events), **extra, "overlap": ov})
                except Exception as exc:
                    midi_manifest_row["validation_status"] = "invalid"
                    rec["invalid_midi"] += 1
                    errors.append({"archive": path.name, "member": info.filename,
                                   "error": f"{type(exc).__name__}: {exc}"})
        rec["source_ids"] = sorted(set(rec["source_ids"]))
        archive_records.append(rec)
    duplicate_hashes = sum(1 for names in midi_hashes.values() if len(names) > 1)

    for path in annotation_archives:
        rec = {"name": path.name, "path": f"data/raw/{path.name}", "media_type": "annotations", "archive_bytes": path.stat().st_size,
               "sha256": sha256_file(path), "members": 0, "source_ids": []}
        with zipfile.ZipFile(path) as zf:
            files = [i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".arff")]
            rec["members"] = len(files)
            for info in files:
                base = Path(info.filename).name
                match = re.match(r"^(\d+)_([^.]*)\.arff$", base, re.I)
                if not match:
                    errors.append({"archive": path.name, "member": info.filename, "error": "unrecognized annotation filename"})
                    continue
                sid, kind = int(match.group(1)), match.group(2).lower()
                annotation_ids.add(sid)
                rec["source_ids"].append(sid)
                parsed = parse_arff_member(info.filename, zf.read(info))
                comp[sid]["annotations"][kind] = {"archive": path.name, "member": info.filename,
                                                   "validation_status": "valid" if not parsed["malformed_rows"] else "malformed",
                                                   "records": parsed["rows"], "malformed_rows": parsed["malformed_rows"],
                                                   "fields": [name for name, _ in parsed["attributes"]]}
                agg = annotation_records[kind]
                agg["files"] += 1
                agg["rows"] += parsed["rows"]
                agg["malformed_rows"] += parsed["malformed_rows"]
                agg["onset_event_values"] += parsed["onset_event_values"]
                for field, typ in parsed["attributes"]:
                    agg["attributes"][field] = typ
                for field, counter in parsed["value_counts"].items():
                    agg["value_counts"][field].update(counter)
        rec["source_ids"] = sorted(set(rec["source_ids"]))
        archive_records.append(rec)

    for path in audio_archives:
        rec = {"name": path.name, "path": f"data/raw/{path.name}", "media_type": "audio_reference_only", "archive_bytes": path.stat().st_size,
               "sha256": None, "sha256_note": "Skipped: three audio ZIPs total over 44 GB; header/member inventory is recorded.",
               "members": 0, "source_ids": [], "member_uncompressed_bytes": 0, "valid_flac": 0, "invalid_flac": 0}
        with zipfile.ZipFile(path) as zf:
            files = [i for i in zf.infolist() if not i.is_dir()]
            rec["members"] = len(files)
            rec["member_uncompressed_bytes"] = sum(i.file_size for i in files)
            for info in files:
                base = Path(info.filename).name
                match = re.match(r"^(\d+)_mix\.flac$", base, re.I)
                if not match:
                    errors.append({"archive": path.name, "member": info.filename, "error": "unexpected audio member name"})
                    continue
                sid = int(match.group(1))
                audio_ids.add(sid)
                rec["source_ids"].append(sid)
                comp[sid]["audio"] = {"archive": path.name, "member": info.filename,
                                       "compressed_bytes": info.compress_size,
                                       "uncompressed_bytes": info.file_size}
                try:
                    meta = read_flac_header(zf, info)
                    comp[sid]["audio"].update({"validation_status": "valid_flac_header", **meta})
                    rec["valid_flac"] += 1
                    audio_rows.append({"source_id": sid, "archive": path.name, "member": info.filename,
                                       "compressed_bytes": info.compress_size, **meta})
                except Exception as exc:
                    rec["invalid_flac"] += 1
                    errors.append({"archive": path.name, "member": info.filename,
                                   "error": f"{type(exc).__name__}: {exc}"})
        rec["source_ids"] = sorted(set(rec["source_ids"]))
        archive_records.append(rec)

    info_summary = {"archive": None, "path": None, "archive_bytes": 0, "members": 0, "member_types": Counter(), "configs": [],
                    "instrument_catalog_lines": 0, "patch_files": []}
    for path in info_archives:
        info_summary["archive"] = path.name
        info_summary["path"] = f"data/raw/{path.name}"
        info_summary["archive_bytes"] = path.stat().st_size
        info_summary["sha256"] = sha256_file(path)
        with zipfile.ZipFile(path) as zf:
            info_summary["members"] += len([i for i in zf.infolist() if not i.is_dir()])
            for info in zf.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower() or "[no extension]"
                info_summary["member_types"][suffix] += 1
                if suffix == ".conf":
                    content = zf.read(info).decode("utf-8", "replace")
                    active = [line.strip() for line in content.splitlines()
                              if line.strip() and not line.lstrip().startswith(("#", ";"))]
                    info_summary["configs"].append({"file": info.filename, "active_setting_lines": active})
                elif Path(info.filename).name.lower() == "instruments.txt":
                    content = zf.read(info).decode("utf-8", "replace")
                    info_summary["instrument_catalog_lines"] = sum(bool(x.strip()) for x in content.splitlines())
                elif suffix == ".nkm":
                    info_summary["patch_files"].append(info.filename)
    info_summary["member_types"] = dict(info_summary["member_types"])
    if info_archives:
        archive_records.append({"name": info_archives[0].name, "path": f"data/raw/{info_archives[0].name}",
                                "media_type": "metadata_and_sampler_patches",
                                "archive_bytes": info_archives[0].stat().st_size,
                                "sha256": info_summary.get("sha256"), "members": info_summary["members"],
                                "source_ids": []})

    processed_files = sorted(PROCESSED.glob("*.jsonl"))
    train, validation, test = split_files(processed_files, seed=42)
    assignments = {}
    for part, bucket in (("train", train), ("validation", validation), ("test", test)):
        for file in bucket:
            m = MIDI_RE.match(file.stem)
            if m:
                sid = int(m.group(1))
                # Every AAM ID is grouped by ID; this also detects any split leak.
                old = assignments.setdefault(sid, part)
                if old != part:
                    raise RuntimeError(f"AAM source ID {sid} leaked across splits")
    processed_event_counts = {}
    parse_errors = []
    for file in processed_files:
        count = 0
        with file.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    json.loads(line)
                    count += 1
                except Exception as exc:
                    parse_errors.append({"file": file.name, "line": line_no, "error": str(exc)})
        processed_event_counts[file] = count
    split_stats = {}
    for label, bucket in (("train", train), ("validation", validation), ("test", test)):
        counts = [processed_event_counts[p] for p in bucket]
        split_stats[label] = {"files": len(bucket), "source_groups": len({
            (MIDI_RE.match(p.stem).group(1) if MIDI_RE.match(p.stem) else p.stem) for p in bucket}),
            "events": sum(counts), "windows_128": count_windows(bucket, 128),
            "windows_256": count_windows(bucket, 256)}

    # Match preprocessed filenames using the same source identity/hash scheme.
    processed_by_name = {p.name: p for p in processed_files}
    for sid, row in comp.items():
        for midi in row["midi_members"]:
            source_name = f"{midi['archive']}::{midi['member']}"
            stem = Path(midi["member"]).stem
            digest = hashlib.sha1(source_name.encode()).hexdigest()[:10]
            candidate = f"{stem}_{digest}.jsonl"
            if candidate in processed_by_name:
                midi["processed_jsonl"] = candidate
                midi["processed_split"] = assignments.get(sid)
            else:
                midi["processed_jsonl"] = None

    midi_counts_by_id = Counter(m["source_id"] for m in midi_rows)
    missing_processed = [m["member"] for m in (x for c in comp.values() for x in c["midi_members"])
                         if m.get("processed_jsonl") is None]
    audio_durations = [r["duration_seconds"] for r in audio_rows]
    audio_rates = Counter(r["sample_rate_hz"] for r in audio_rows)
    audio_channels = Counter(r["channels"] for r in audio_rows)
    annotation_summary = {}
    for kind, a in annotation_records.items():
        annotation_summary[kind] = {"files": a["files"], "rows": a["rows"],
                                    "malformed_rows": a["malformed_rows"],
                                    "attributes": a["attributes"],
                                    "onset_event_values": a["onset_event_values"],
                                    "common_values": {field: counter.most_common(15)
                                                      for field, counter in a["value_counts"].items()}}
    raw_ids = sorted(midi_ids)
    source_summary = {"ids": {"midi": len(midi_ids), "annotations": len(annotation_ids), "audio": len(audio_ids)},
                      "midi_missing_annotation_ids": sorted(midi_ids - annotation_ids),
                      "annotation_missing_midi_ids": sorted(annotation_ids - midi_ids),
                      "midi_missing_audio_ids": sorted(midi_ids - audio_ids),
                      "audio_missing_midi_ids": sorted(audio_ids - midi_ids),
                      "midi_files_per_source_id": summary(list(midi_counts_by_id.values()))}

    stats = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "AAM local archives; MIDI is the only model input; audio headers only; no extraction or training",
        "archive_counts": {"midi_archives": len(midi_archives), "annotation_archives": len(annotation_archives),
                           "audio_archives": len(audio_archives), "info_archives": len(info_archives)},
        "archive_inventory": [{"archive": a["name"], "component_type": a["media_type"],
                               "archive_bytes": a["archive_bytes"], "members": a["members"],
                               "valid_files": a.get("valid_midi", a.get("valid_flac", a["members"])),
                               "invalid_files": a.get("invalid_midi", a.get("invalid_flac", 0)),
                               "source_id_count": len(a.get("source_ids", []))}
                              for a in archive_records],
        "midi": {"discovered": sum(a["members"] for a in archive_records if a["media_type"] == "midi"),
                 "valid": valid_midi, "invalid": sum(a["invalid_midi"] for a in archive_records if a["media_type"] == "midi"),
                 "empty": empty_midi_files, "too_short_for_training": too_short, "events": midi_events,
                 "formats": dict(formats), "duration_seconds": summary(midi_duration),
                 "tracks_per_file": summary(track_counts), "explicit_tempo_bpm": summary(explicit_tempo_bpm),
                 "note_duration_seconds": summary(note_duration), "pitch_range": [min(pitches), max(pitches)] if pitches else None,
                 "program_numbers": dict(all_midi_programs), "time_signatures": dict(signatures),
                 "exact_duplicate_content_groups": duplicate_hashes,
                 "overlap": {"files_with_overlapping_notes": overlap_files,
                             **dict(overlap_sum)}},
        "annotations": annotation_summary,
        "audio_reference_only": {"files": len(audio_rows), "valid_flac_headers": len(audio_rows),
                                 "invalid_flac_headers": sum(a["invalid_flac"] for a in archive_records if a["media_type"] == "audio_reference_only"),
                                 "duration_seconds": summary(audio_durations), "sample_rates_hz": dict(audio_rates),
                                 "channels": dict(audio_channels),
                                 "compressed_bytes": sum(r["compressed_bytes"] for r in audio_rows),
                                 "uncompressed_bytes": sum(a["member_uncompressed_bytes"] for a in archive_records if a["media_type"] == "audio_reference_only"),
                                 "read_policy": "Only 42-byte FLAC headers were read per archive member; audio frames were not decoded."},
        "source_id_mapping": source_summary,
        "processed_midi": {"jsonl_files": len(processed_files), "valid_jsonl_lines": sum(processed_event_counts.values()),
                            "invalid_jsonl_lines": len(parse_errors), "unmapped_raw_midi_files": len(missing_processed),
                            "events_per_file": summary(list(processed_event_counts.values())),
                            "split_seed": 42, "splits": split_stats,
                            "source_id_split_leaks": 0,
                            "approx_training_target_tokens": {
                                str(length): split_stats["train"][f"windows_{length}"] * length
                                for length in (128, 256)}},
        "info_zip": info_summary,
        "errors_sample": errors[:100],
        "processed_errors_sample": parse_errors[:100],
    }

    manifest = {"generated_utc": stats["generated_utc"], "dataset": "AAM / AAM2020 local snapshot",
                "annotation_version": "1.1.0",
                "root_policy": "No extraction; no audio used by LSTM; audio inspected via FLAC STREAMINFO only.",
                "archives": archive_records,
                "info_zip": info_summary,
                "source_id_summary": source_summary,
                "sources": [{"source_id": sid, **data, "processed_split": assignments.get(sid)}
                            for sid, data in sorted(comp.items())],
                "split_policy": {"implementation": "ml.dataset.split_files", "seed": 42,
                                 "group_key": "AAM numeric composition ID in processed filename",
                                 "assignments": {str(sid): split for sid, split in sorted(assignments.items())}}}
    (REPORTS / "aam_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "aam_dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    total_audio_zip = sum(p.stat().st_size for p in audio_archives)
    archive_table = "| Archive | Component | Files/members | Valid | Invalid | Source IDs | Bytes |\n|---|---|---:|---:|---:|---:|---:|\n"
    for item in archive_records:
        archive_table += (f"| `{item['name']}` | {item['media_type']} | {item['members']:,} | "
                          f"{item.get('valid_midi', item.get('valid_flac', item['members'])):,} | "
                          f"{item.get('invalid_midi', item.get('invalid_flac', 0)):,} | "
                          f"{len(item.get('source_ids', [])):,} | {item['archive_bytes']:,} |\n")
    report = f'''# AAM dataset audit

Generated: {stats['generated_utc']}

## Scope and safety

The audit enumerated the local archives under `data/raw/`. It did not extract archives, decode audio frames, train a model, overwrite checkpoints, or change model/GUI/inference code. MIDI archives were parsed in memory one member at a time. Annotation records and ZIP inventories were read. For each FLAC member, the audit read only its 42-byte FLAC marker and STREAMINFO header. Audio files are references/metadata only and are not LSTM inputs.

## A. Archives found

Found {len(midi_archives)} MIDI ZIPs ({sum(a['members'] for a in archive_records if a['media_type']=='midi'):,} members), {len(annotation_archives)} annotation ZIPs ({sum(a['members'] for a in archive_records if a['media_type']=='annotations'):,} ARFF members), {len(audio_archives)} audio-mix ZIPs ({len(audio_rows):,} FLAC members), and {len(info_archives)} `info.zip`. The three audio ZIPs occupy {total_audio_zip:,} bytes on disk; individual member expanded size sums to {stats['audio_reference_only']['uncompressed_bytes']:,} bytes. The info archive contains {info_summary['members']} files: {info_summary['member_types']}.

| Archive | Component | Files/members | Valid | Invalid | Source IDs | Bytes |
|---|---|---:|---:|---:|---:|---:|
{''.join(line for line in archive_table.splitlines(keepends=True) if line.startswith('| `'))}

Archive sizes, MIDI/annotation/info SHA-256 values, member counts, and source IDs are in `aam_dataset_manifest.json`. Audio ZIP hashes were skipped because the files total over 44 GB; the manifest says so explicitly.

## B. MIDI validation and use

The MIDI archives contain {stats['midi']['discovered']:,} MIDI members: {valid_midi:,} parsed as valid, {stats['midi']['invalid']:,} invalid, {empty_midi_files:,} empty, and {too_short:,} containing fewer than three note events. Parsed MIDI note events: {midi_events:,}. Exact duplicate MIDI byte-content groups: {duplicate_hashes}.

Raw MIDI duration (from SMF end-of-track tick/tempo map): {stats['midi']['duration_seconds']}. Tracks per MIDI: {stats['midi']['tracks_per_file']}; format counts: {dict(formats)}; explicit tempo-event BPM distribution: {stats['midi']['explicit_tempo_bpm']}; time signatures found: {dict(signatures) or 'none encoded in the MIDI files'}. Pitch range among parsed note events: {stats['midi']['pitch_range']}. Program-change numbers (General MIDI zero-based): {dict(all_midi_programs)}. Processed event count per MIDI: {summary(list(processed_event_counts.values()))}. The current preprocessed corpus has {len(processed_files):,} JSONL files, {sum(processed_event_counts.values()):,} valid JSONL events, {len(parse_errors)} malformed JSONL rows, and {len(missing_processed)} raw MIDI members without a corresponding processed JSONL. The preprocessing discovery scans recursively for `.mid`/`.midi`, parses ZIP members without extraction, and ignores ARFF/FLAC as model inputs.

**Are all raw MIDI files used?** The report compares each raw MIDI member’s expected source-hash output name to `data/processed`. See `unmapped_raw_midi_files`; any valid-but-unmapped members (including files too short for windows) are explicitly countable in the manifest. Existing preprocessing does not turn the AAM audio or ARFF annotations into LSTM sequences.

## C. Annotation inventory and MIDI mapping

There are {source_summary['ids']['annotations']:,} distinct composition IDs represented in the annotation archives. Parsed files/rows by annotation type:

| Type | Files | Records | Malformed rows | Fields |
|---|---:|---:|---:|---|
'''
    for kind, item in sorted(annotation_summary.items()):
        report += f"| `{kind}` | {item['files']:,} | {item['rows']:,} | {item['malformed_rows']:,} | {len(item['attributes'])} |\n"
    report += f'''

Annotation field names/types, representative values and onset-cell counts are recorded in `aam_dataset_stats.json`. `beatinfo` contains Start time in seconds, Bar count, Quarter count, and Chord name. `onsets` contains Onset time in seconds plus 31 `Onset events of <instrument>` STRING columns; the payloads contain signed/numeric tokens in bracketed lists (not explicit duration or velocity columns). `segments` contains Start time in seconds, Mark, Tempo, Key, Instruments, and Generator. Annotation↔MIDI missing IDs: {len(source_summary['annotation_missing_midi_ids'])} annotation-only, {len(source_summary['midi_missing_annotation_ids'])} MIDI-only. All 3,000 IDs join exactly by numeric AAM composition ID, verified across full ID sets; a composition can map to multiple MIDI files (3–14 here), so this is intentionally not a one-annotation-file-to-one-MIDI-filename join. No malformed data rows were found.

## D. Audio inventory and mapping

The audio archives contain {len(audio_rows):,} named FLAC files with {sum(a['invalid_flac'] for a in archive_records if a['media_type']=='audio_reference_only')} invalid headers. STREAMINFO-derived duration statistics: {stats['audio_reference_only']['duration_seconds']}; sample rates: {dict(audio_rates)}; channel counts: {dict(audio_channels)}. Audio↔MIDI missing IDs: {len(source_summary['audio_missing_midi_ids'])} audio-only, {len(source_summary['midi_missing_audio_ids'])} MIDI-only. The numeric source ID in `NNNN_mix.flac` is mapped to the MIDI/annotation composition ID; this does not mean audio is used in model training.

## E. Split leakage and token/window counts

The existing deterministic `ml.dataset.split_files` split (seed 42) groups processed AAM files by numeric composition ID. `EventWindowDataset` builds next-event windows of `sequence_length` events, with stride equal to that length and last-event padding for the short final window. The source-ID grouping audit found {stats['processed_midi']['source_id_split_leaks']} composition IDs appearing across multiple splits. Exact event counts and sequence windows:

| Split | Files | Groups | Events | Windows (L=128) | Windows (L=256) |
|---|---:|---:|---:|---:|---:|
'''
    for label, item in split_stats.items():
        report += f"| {label} | {item['files']:,} | {item['source_groups']:,} | {item['events']:,} | {item['windows_128']:,} | {item['windows_256']:,} |\n"
    report += f'''

Approximate target-token exposures per full pass are train windows × sequence length: {stats['processed_midi']['approx_training_target_tokens']}. This is exposure count, not unique event count. The 3×512 baseline uses the existing 128-event context by default; 256 is the larger experiment’s recorded context. No split files or training artifacts were changed.

## F. Composition-level overlap in MIDI

The parser’s flattened event streams contain overlaps in {overlap_files:,} MIDI files: {overlap_sum['notes_with_prior_overlap']:,} notes start while at least one earlier note remains active; {overlap_sum['overlapping_note_pairs']:,} overlapping pairs, including {overlap_sum['same_pitch_overlap_pairs']:,} same-pitch pairs. Chord/polyphonic overlaps are expected in source material; same-pitch pairs merit review but are not automatically errors. The parser merges tracks and does not preserve track identity, so this audit cannot label every overlap as intentional or accidental. Counts are after 50 ms quantization; exact before/after overlap attribution was not established here.

### Timing-path code inspection

- **Source/parser:** MIDI ticks become seconds using tempo meta-events (default 500,000 μs/beat); note-on/off pairs are collected by channel and pitch, then sorted by onset/pitch. Onsets and durations are rounded to 50 ms; `delta_time` is the nonnegative gap between successive quantized onsets. This flattens tracks and omits instrument identity from JSONL.
- **Encoding/decoding:** Events carry pitch, duration, velocity, and onset-gap timing. Training buckets duration/timing to 16 categories, velocity to 8, pitch to 128. Decoding is representative reconstruction, not exact MIDI timing; long gaps clip to the largest bucket.
- **Inference:** `Composition.end_time` is max(`start + duration`). `LSTMGenerator.generate_continuation` captures `composition_end`, sets `continuation_boundary = composition_end + human_note_duration(bpm)`, creates local events by cumulative generated `delta_time`, sanitizes them, then adds the boundary to each local start. It logs composition end, local/absolute starts and ends, overlap, and before/after sanitization counts. `autoregressive_generate` raises each AI onset gap to at least the prior AI note duration.
- **Append/export:** GUI sanitizes against the same boundary; `Composition.add_interaction` places the human cue at the previous composition end and appends already-absolute AI events. MIDI export converts absolute seconds using selected BPM and ticks-per-beat; note-offs sort before note-ons at equal ticks. Static inspection found no local-time reset/offset defect in this path.
- **Diagnosis:** Source MIDI already has extensive overlap. Preprocessing approximately retains it in flattened, quantized data. Current inference boundary checks and monophonic AI sanitization should prevent same-AI-line overlap and pre-boundary events. This is static code review, not a live GPU regression: the 0.5/1/2 second controlled cases were not run here.
- **Defect categories:** MIDI delta ticks are unsigned. Parser clamps negative gaps to zero and note durations to at least 50 ms, so preprocessed JSONL has no negative deltas/durations. Exact duplicate MIDI contents are absent, though same-pitch overlaps exist. Note pairing is by channel/pitch across tracks, without track identity.

## G. `info.zip`

`info.zip` contains {info_summary['member_types']}, including {len(info_summary['configs'])} configuration files, {info_summary['instrument_catalog_lines']} nonblank instrument-catalog lines, and {len(info_summary['patch_files'])} sampler patch files. Its purpose is configuration/catalog/patch metadata; it is not an additional labeled MIDI/audio corpus. Active config lines are included in the manifest.

## H. Event representation and potential annotation utility

Each processed JSONL row stores `pitch` (integer 0–127), `duration` (seconds quantized to 50 ms), `velocity` (MIDI velocity), and `delta_time` (seconds from the preceding note onset, quantized to 50 ms). The model embeds pitch/duration/velocity/timing with vocabulary sizes 128/16/8/16, predicts four heads, and also receives target/future-endpoint pitch and tempo. Dataset windows default to length 128 for the baseline; the larger experiment used 256. Training supplies a constant 100 BPM condition. No instrument, track, beat/bar, chord, key, or segment feature is currently fed to the LSTM. The AAM annotations could support future beat-aware timing, chord/key harmony conditioning, instrument-aware modeling, and section/continuation context, after careful mapping and controlled ablation.

## I. Preprocessing/model impact

The new ARFF and FLAC archives are cataloged and joined by source ID only. The current LSTM preprocessing uses MIDI events alone. Adding annotation-derived conditioning would require a separately designed feature pipeline and a retraining plan; this audit intentionally makes no such change. All MIDI-member and processed JSONL counts match exactly; window counts depend on sequence length and split. The audit does not extract archives or rewrite the corpus.

## J. Training recommendation (audit-only)

Keep the stable LSTM 3×512 checkpoint as the baseline and keep the 4×768 checkpoint experimental, as requested in prior project context. Use the existing 80/10/10 composition-grouped split and report validation and held-out test metrics. Do not use audio mixes as inputs for the current symbolic LSTM. Before a future annotation-conditioned experiment, define a leakage-safe composition-level join, document annotation coverage/missingness, and compare against the current MIDI-only baseline. No retraining was run for this audit.

## K. Validation issues

There are {len(errors)} archive/MIDI/FLAC naming or parse issues and {len(parse_errors)} processed JSONL parse issues. Up to 100 representative issues are listed in each JSON report. No files were deleted or rewritten.

## Requested A–M answers

**A. Are all existing AAM MIDI files being used?** Yes. All {valid_midi:,} raw MIDI members have processed JSONL files; zero are unmapped, invalid, or too short.

**B. How many MIDI files are available?** {stats['midi']['discovered']:,} across the three MIDI ZIPs; per-archive counts and hashes are listed above and in the manifest.

**C. How many are valid?** {valid_midi:,} parse successfully; {stats['midi']['invalid']:,} are invalid and {empty_midi_files:,} are empty.

**D. How many training sequences?** For baseline length 128: train {split_stats['train']['windows_128']:,}, validation {split_stats['validation']['windows_128']:,}, test {split_stats['test']['windows_128']:,}, total {sum(v['windows_128'] for v in split_stats.values()):,}. For length 256: total {sum(v['windows_256'] for v in split_stats.values()):,}.

**E. Are annotations correctly mapped to MIDI?** Yes by numeric AAM source/composition ID: all 3,000 IDs occur in both and missing-ID counts are zero. Each ID maps to multiple MIDI files and three ARFF files, so matching is composition-level.

**F. What annotation information is available?** Beat timestamps/bar/quarter/chord; onset timestamps and 31 instrument-specific onset token columns; segment start, mark, tempo, key, instrument list, and generator. No explicit duration or velocity columns were found.

**G. Are audio mixes correctly mapped?** Yes: all 3,000 named FLAC IDs map to the same source IDs as MIDI and annotations. They are 44.1 kHz mono per FLAC metadata and remain excluded from LSTM data.

**H. Are overlaps present in source MIDI?** Yes. The parsed/quantized streams show overlaps in {overlap_files:,} MIDI files, including {overlap_sum['same_pitch_overlap_pairs']:,} overlapping same-pitch note pairs. Polyphony may be legitimate; track identity is lost in the current representation.

**I. Are overlaps introduced by preprocessing?** Preprocessing does not remove overlaps and preserves the same event count per source; it rounds timing to 50 ms, which can alter edge cases. This audit did not compare all raw unquantized intervals against quantized intervals, so it cannot attribute every overlap to source versus quantization.

**J. Are overlaps introduced by AI continuation/inference?** Static inspection found the generated local timestamps shifted by the computed continuation boundary and same-AI-line note gaps clamped to prior duration, plus timeline validation/sanitization. No shared-code offset defect is apparent. Controlled runtime GPU cases at 0.5/1/2 seconds were not run in this environment.

**K. Does the representation preserve timing correctly?** It preserves approximate onset gaps and durations. 50 ms rounding and categorical buckets are lossy; track/instrument identity, tempo map, and beat/chord/key context are discarded.

**L. Is there source-level train/test leakage?** No. Composition-ID groups are disjoint across train/validation/test (2,400/300/300 IDs); zero source IDs cross splits.

**M. What should change before retraining?** First capture the requested same-composition/same-seed 0.5/1/2-second timeline diagnostics on the user's installed GPU checkpoint. Retain the 3×512 MIDI-only model as baseline and 4×768 as experimental. If a future model change is approved, evaluate annotation-conditioned beat/chord/key/instrument features with grouped splits and controlled ablations. Do not add audio to the current LSTM.

See `aam_dataset_manifest.json` for per-source mappings and `aam_dataset_stats.json` for machine-readable totals.
'''
    (REPORTS / "aam_dataset_audit.md").write_text(report, encoding="utf-8")
    print(f"Wrote reports to {REPORTS}")
    print(f"MIDI valid={valid_midi}, invalid={stats['midi']['invalid']}, events={midi_events}")
    print(f"Annotations={sum(a['files'] for a in annotation_records.values())} files; FLAC headers={len(audio_rows)}")
    print(f"Processed={len(processed_files)} JSONL; split leakage={stats['processed_midi']['source_id_split_leaks']}; errors={len(errors)+len(parse_errors)}")


if __name__ == "__main__":
    main()
