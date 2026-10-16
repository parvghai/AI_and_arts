"""MIDI parsing and normalized event quantization."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import io
import struct
import zipfile

QUANTUM = 0.05
DURATION_EDGES = (0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0,
                  1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
VELOCITY_EDGES = (16, 32, 48, 64, 80, 96, 112)
TIMING_EDGES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0,
                1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


@dataclass
class Event:
    pitch: int
    duration: float
    velocity: int
    delta_time: float


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    return min(len(edges), next((i for i, edge in enumerate(edges) if value <= edge), len(edges)))


def encode_event(event: Event) -> tuple[int, int, int, int]:
    return (max(0, min(127, int(event.pitch))), _bucket(event.duration, DURATION_EDGES),
            min(7, max(0, _bucket(event.velocity, VELOCITY_EDGES))),
            _bucket(max(0.0, event.delta_time), TIMING_EDGES))


def decode_duration(token: int) -> float:
    token = min(max(0, token), len(DURATION_EDGES))
    if token == 0:
        return DURATION_EDGES[0] / 2
    if token == len(DURATION_EDGES):
        return DURATION_EDGES[-1]
    return (DURATION_EDGES[token - 1] + DURATION_EDGES[token]) / 2


def decode_velocity(token: int) -> int:
    token = min(max(0, token), len(VELOCITY_EDGES))
    lower = 1 if token == 0 else VELOCITY_EDGES[token - 1]
    upper = VELOCITY_EDGES[token] if token < len(VELOCITY_EDGES) else 127
    return round((lower + upper) / 2)


def decode_timing(token: int) -> float:
    token = min(max(0, token), len(TIMING_EDGES))
    if token == 0:
        return 0.0
    if token == len(TIMING_EDGES):
        return TIMING_EDGES[-1]
    return (TIMING_EDGES[token - 1] + TIMING_EDGES[token]) / 2


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise ValueError("truncated MIDI variable-length quantity")
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos
    raise ValueError("invalid MIDI variable-length quantity")


def parse_midi_bytes(data: bytes) -> list[Event]:
    """Validate an SMF byte stream and return quantized note events.

    Only MIDI channel/meta messages are parsed. ZIP members with audio extensions
    are never opened, decoded, or extracted.
    """
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("missing or truncated Standard MIDI header")
    header_size = struct.unpack_from(">I", data, 4)[0]
    if header_size < 6 or 8 + header_size > len(data):
        raise ValueError("invalid MIDI header length")
    fmt, track_count, division = struct.unpack_from(">HHH", data, 8)
    if fmt not in (0, 1, 2) or track_count < 1 or division == 0 or division & 0x8000:
        raise ValueError("unsupported MIDI format or time division")
    pos = 8 + header_size
    tracks: list[list[tuple[int, int, str, int, int]]] = []
    for track_index in range(track_count):
        if pos + 8 > len(data) or data[pos:pos+4] != b"MTrk":
            raise ValueError(f"missing MIDI track chunk {track_index + 1}")
        chunk_len = struct.unpack_from(">I", data, pos + 4)[0]
        start, end = pos + 8, pos + 8 + chunk_len
        if end > len(data):
            raise ValueError(f"truncated MIDI track chunk {track_index + 1}")
        pos = end
        cursor = start
        tick = 0
        running_status = None
        events: list[tuple[int, int, str, int, int]] = []
        order = 0
        while cursor < end:
            delta, cursor = _read_vlq(data, cursor)
            tick += delta
            if cursor >= end:
                raise ValueError("truncated MIDI event")
            status = data[cursor]
            if status & 0x80:
                cursor += 1
                if status < 0xF0:
                    running_status = status
            elif running_status is not None:
                status = running_status
            else:
                raise ValueError("MIDI running status without prior channel event")
            if status == 0xFF:
                if cursor >= end:
                    raise ValueError("truncated MIDI meta event")
                kind = data[cursor]; cursor += 1
                length, cursor = _read_vlq(data, cursor)
                if cursor + length > end:
                    raise ValueError("truncated MIDI meta payload")
                payload = data[cursor:cursor+length]; cursor += length
                if kind == 0x51 and length == 3:
                    tempo = int.from_bytes(payload, "big")
                    if tempo:
                        events.append((tick, order, "tempo", tempo, 0)); order += 1
                elif kind == 0x2F:
                    events.append((tick, order, "end", 0, 0))
                    break
                continue
            if status in (0xF0, 0xF7):
                length, cursor = _read_vlq(data, cursor)
                if cursor + length > end:
                    raise ValueError("truncated MIDI system-exclusive payload")
                cursor += length
                continue
            if status >= 0xF0:
                raise ValueError("unsupported MIDI system message")
            message = status & 0xF0
            size = 1 if message in (0xC0, 0xD0) else 2
            if cursor + size > end:
                raise ValueError("truncated MIDI channel message")
            a = data[cursor]; b = data[cursor+1] if size == 2 else 0
            cursor += size
            if a > 127 or b > 127:
                raise ValueError("invalid MIDI data byte")
            if message == 0x90 and b > 0:
                events.append((tick, order, "on", (status & 0x0F) * 128 + a, b)); order += 1
            elif message == 0x80 or (message == 0x90 and b == 0):
                events.append((tick, order, "off", (status & 0x0F) * 128 + a, b)); order += 1
        tracks.append(events)
    if pos < len(data) and any(data[pos:]):
        raise ValueError("unexpected trailing data after MIDI tracks")

    merged = sorted((tick, track_i, order, kind, value, velocity)
                    for track_i, track in enumerate(tracks)
                    for tick, order, kind, value, velocity in track)
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    notes: list[tuple[float, float, int, int]] = []
    seconds, prev_tick, tempo = 0.0, 0, 500_000
    for tick, _track_i, _order, kind, value, velocity in merged:
        seconds += (tick - prev_tick) * tempo / (division * 1_000_000)
        prev_tick = tick
        if kind == "tempo":
            tempo = value
        elif kind == "on":
            channel, pitch = divmod(value, 128)
            active.setdefault((channel, pitch), []).append((seconds, velocity))
        elif kind == "off":
            channel, pitch = divmod(value, 128)
            key = (channel, pitch)
            if active.get(key):
                start, vel = active[key].pop(0)
                notes.append((start, max(QUANTUM, seconds - start), pitch, vel))
    # Some multitrack exports omit note-offs at end-of-track. Close these notes at
    # the validated track end instead of discarding an otherwise useful MIDI file.
    for (_channel, pitch), starts in active.items():
        for start, vel in starts:
            notes.append((start, max(QUANTUM, seconds - start), pitch, vel))
    notes.sort(key=lambda row: (row[0], row[2]))
    result = []
    origin = notes[0][0] if notes else 0.0
    previous = origin
    for start, length, pitch, velocity in notes:
        qstart = round((start - origin) / QUANTUM) * QUANTUM
        qlength = max(QUANTUM, round(length / QUANTUM) * QUANTUM)
        result.append(Event(pitch, qlength, velocity, max(0, qstart - previous)))
        previous = qstart
    return result


def _midi_sources(source: Path):
    """Yield MIDI files from recursive folders and ZIPs without extracting them."""
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".mid", ".midi"):
            yield path.relative_to(source).as_posix(), path.read_bytes()
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if info.is_dir() or Path(info.filename).suffix.lower() not in (".mid", ".midi"):
                            continue
                        # Refuse suspiciously large members; ordinary MIDI is far smaller.
                        if info.file_size > 64 * 1024 * 1024:
                            print(f"Skipping oversized MIDI member: {path.name}:{info.filename}")
                            continue
                        try:
                            yield f"{path.relative_to(source).as_posix()}::{info.filename}", archive.read(info)
                        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                            print(f"Skipping unreadable MIDI member {path.name}:{info.filename}: {exc}")
            except (OSError, zipfile.BadZipFile) as exc:
                print(f"Skipping invalid ZIP {path}: {exc}")


def preprocess_directory(source: Path, destination: Path) -> int:
    """Recursively validate MIDI files and write normalized JSONL training records."""
    destination.mkdir(parents=True, exist_ok=True)
    count = invalid = too_short = discovered = total_events = 0
    approx_bytes = 0
    for source_name, contents in _midi_sources(source):
        discovered += 1
        try:
            events = parse_midi_bytes(contents)
        except Exception as exc:
            invalid += 1
            print(f"Skipping invalid MIDI {source_name}: {exc}")
            continue
        if len(events) < 3:
            too_short += 1
            continue
        basename = source_name.rsplit("::", 1)[-1]
        stem = Path(basename).stem
        digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:10]
        out = destination / f"{stem}_{digest}.jsonl"
        out.write_text("\n".join(json.dumps(asdict(e)) for e in events), encoding="utf-8")
        count += 1
        total_events += len(events)
        approx_bytes += out.stat().st_size
    report = {"source": str(source), "discovered_midi_files": discovered,
              "converted_files": count, "invalid_files": invalid,
              "files_with_fewer_than_3_notes": too_short,
              "audio_files_read": 0, "total_events": total_events,
              "sequences": count, "approximate_dataset_bytes": approx_bytes,
              "quantization_seconds": QUANTUM}
    (destination / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MIDI preprocessing: discovered={discovered}, converted={count}, invalid={invalid}, too_short={too_short}; audio files read=0")
    return count


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Local directory containing MIDI files and/or ZIP archives")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(f"Preprocessed {preprocess_directory(args.data, args.output)} MIDI files")


if __name__ == "__main__":
    main()
