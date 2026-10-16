"""Generate a continuation MIDI from a selected AAM source track."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from ml.aam_annotated.model import AAMAnnotatedMusicLSTM
from ml.preprocessing import Event, decode_duration, decode_velocity


def _sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1).item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _write_midi(path: Path, events: list[Event], bpm: float) -> None:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("MIDI export requires mido. Install it with: python -m pip install mido") from exc
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    microseconds = round(60_000_000 / bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=microseconds, time=0))
    ticks_per_second = midi.ticks_per_beat * bpm / 60.0
    scheduled = []
    onset = 0.0
    for event in events:
        onset += max(0.0, event.delta_time)
        start = onset
        end = start + max(.02, event.duration)
        scheduled.append((round(start * ticks_per_second), 1,
                          mido.Message("note_on", note=event.pitch, velocity=event.velocity)))
        scheduled.append((round(end * ticks_per_second), 0,
                          mido.Message("note_off", note=event.pitch, velocity=0)))
    previous = 0
    for tick, _, message in sorted(scheduled, key=lambda row: (row[0], row[1])):
        track.append(message.copy(time=max(0, tick - previous)))
        previous = tick
    midi.save(path)


def generate(args) -> dict:
    root = Path(args.experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or root / "models" / "aam_annotated_best.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    mapping = json.loads((root / "feature_mappings.json").read_text(encoding="utf-8"))
    inv = {name: {int(index): label for label, index in labels.items()}
           for name, labels in mapping["values"].items()}
    instrument_id = mapping["values"]["instrument"].get(args.instrument, 0) if args.instrument else None
    split = args.split
    data_dir = root / "data"
    records = []
    with (data_dir / f"{split}_index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("source_id") == args.source_id and row.get("validation_status") == "valid":
                if args.member is None or row["member"] == args.member:
                    if instrument_id is None or mapping["values"]["instrument"].get(row["instrument"], 0) == instrument_id:
                        records.append(row)
    if not records:
        raise SystemExit("No matching valid member found for that source ID, split, and instrument.")
    if len(records) > 1 and args.member is None and args.instrument is None:
        raise SystemExit("Source has multiple MIDI members; supply --member or --instrument to select the seed track.")
    row = records[0]
    cat_shape = json.loads((data_dir / f"{split}_categorical.shape.json").read_text(encoding="utf-8"))
    num_shape = json.loads((data_dir / f"{split}_numeric.shape.json").read_text(encoding="utf-8"))
    import numpy as np
    cat_all = np.memmap(data_dir / f"{split}_categorical.bin", dtype=np.uint16, mode="r",
                        shape=(cat_shape["events"], cat_shape["features"]))
    num_all = np.memmap(data_dir / f"{split}_numeric.bin", dtype=np.float32, mode="r",
                        shape=(num_shape["events"], num_shape["features"]))
    begin, end = int(row["start"]), int(row["start"] + row["count"])
    cats = np.asarray(cat_all[begin:end], dtype=np.int64)
    nums = np.asarray(num_all[begin:end], dtype=np.float32)
    del cat_all, num_all
    context_size = min(max(4, args.context_events), len(cats))
    context_cat = torch.tensor(cats[-context_size:], dtype=torch.long, device=device)
    context_num = torch.tensor(nums[-context_size:], dtype=torch.float32, device=device)

    source_rows = []
    with (data_dir / f"{split}_index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("source_id") == args.source_id and item.get("validation_status") == "valid":
                source_rows.append(item)
    project_root = root.resolve().parents[1]
    processed_dir = Path(args.processed) if args.processed else project_root / "data" / "processed"
    absolute_starts, composition_end = [], 0.0
    selected_processed = None
    for item in source_rows:
        choices = sorted(processed_dir.glob(Path(item["member"]).stem + "*.jsonl"))
        if not choices:
            continue
        path = choices[0]
        events = [Event(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
        cursor = 0.0
        offset = item.get("absolute_onset_origin_seconds") or 0.0
        for event in events:
            cursor += float(event.delta_time)
            composition_end = max(composition_end, offset + cursor + float(event.duration))
        if item is row or item.get("member") == row.get("member"):
            selected_processed = path
            absolute_starts = []
            cursor = 0.0
            for event in events:
                cursor += float(event.delta_time)
                absolute_starts.append(float(offset) + cursor)
    if not absolute_starts:
        raise SystemExit(f"Could not resolve processed source events under {processed_dir}")
    if row["alignment"]["status"] != "aligned":
        print("WARNING: selected track's onset annotations were unaligned; using MIDI-relative time and unknown annotation context.")
    composition_end = max(composition_end, absolute_starts[-1] + float(Event(**json.loads(selected_processed.read_text(encoding="utf-8").splitlines()[-1])).duration))
    last_source_start = absolute_starts[-1]

    model = AAMAnnotatedMusicLSTM(mapping["vocab_sizes"], cfg["embedding_size"], cfg["hidden_size"],
                                  cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    target_note = torch.tensor([args.target_pitch], dtype=torch.long, device=device)
    tempo = min(240.0, max(30.0, float(nums[-1, 4]) * 240.0))
    elapsed = composition_end
    previous_end = composition_end
    generated: list[Event] = []
    generated_abs = []
    current_cat, current_num = context_cat, context_num
    max_events = min(args.max_events, 1024)
    with torch.inference_mode():
        for _ in range(max_events):
            outputs, _ = model(current_cat.unsqueeze(0), current_num.unsqueeze(0), target_note)
            last_logits = {key: value[0, -1] for key, value in outputs.items()}
            pitch = _sample(last_logits["pitch"], args.temperature, rng)
            duration_token = _sample(last_logits["duration"], args.temperature, rng)
            velocity_token = _sample(last_logits["velocity"], args.temperature, rng)
            gap_norm = float(last_logits["timing"].item())
            delta_beats = math.expm1(min(1.0, max(0.0, gap_norm)) * math.log1p(16.0))
            gap_seconds = delta_beats * 60.0 / tempo
            duration = decode_duration(duration_token)
            previous_start = generated_abs[-1] if generated_abs else last_source_start
            onset = max(previous_start + gap_seconds, previous_end)
            if not generated:
                onset = max(onset, composition_end)
            actual_gap_seconds = max(0.0, onset - previous_start)
            actual_gap_beats = actual_gap_seconds * tempo / 60.0
            actual_gap_norm = min(1.0, math.log1p(min(16.0, actual_gap_beats)) / math.log1p(16.0))
            relative_onset = onset - composition_end
            if generated and relative_onset >= args.seconds:
                break
            if relative_onset >= args.seconds:
                break
            velocity = decode_velocity(velocity_token)
            previous_end = onset + duration
            elapsed = onset
            event = Event(pitch, duration, velocity, relative_onset if not generated else onset - generated_abs[-1])
            generated.append(event)
            generated_abs.append(onset)

            phase = math.atan2(float(current_num[-1, 1]), float(current_num[-1, 2]))
            advanced_beats = actual_gap_beats
            phase = (phase + 2.0 * math.pi * advanced_beats / 4.0) % (2.0 * math.pi)
            within = (float(current_num[-1, 3]) + advanced_beats) % 1.0
            next_cat = current_cat[-1].clone()
            next_cat[0] = pitch
            next_cat[1] = duration_token
            next_cat[2] = velocity_token
            next_num = current_num[-1].clone()
            next_num[0] = actual_gap_norm
            next_num[1] = math.sin(phase)
            next_num[2] = math.cos(phase)
            next_num[3] = within
            next_num[4] = tempo / 240.0
            next_num[5] = min(1.0, max(0.0, (onset - composition_end) / max(args.seconds, 1e-6)))
            next_num[6] = min(1.0, duration / 8.0)
            current_cat = torch.cat((current_cat, next_cat.unsqueeze(0)), dim=0)[-cfg["sequence_length"]:]
            current_num = torch.cat((current_num, next_num.unsqueeze(0)), dim=0)[-cfg["sequence_length"]:]

    if not generated:
        raise RuntimeError("Model generated no notes within the requested continuation duration.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_midi(args.output, generated, tempo)
    result = {"checkpoint": str(checkpoint_path), "source_id": args.source_id,
              "source_member": row["member"], "split": split, "target_pitch": args.target_pitch,
              "composition_end_seconds": composition_end, "tempo_bpm": tempo,
              "requested_seconds": args.seconds, "generated_notes": len(generated),
              "events": [{"pitch": event.pitch, "duration_seconds": event.duration,
                          "velocity": event.velocity, "delta_seconds": event.delta_time,
                          "absolute_start_seconds": start, "absolute_end_seconds": start + event.duration}
                         for event, start in zip(generated, generated_abs)],
              "output_midi": str(args.output), "device": str(device)}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/aam_annotated"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--member", default=None, help="Exact AAM MIDI archive member name, e.g. 0001_Drums.mid")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--target-pitch", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--context-events", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/aam_annotated/generated/continuation.mid"))
    args = parser.parse_args()
    if not 0 <= args.target_pitch <= 127:
        parser.error("--target-pitch must be between 0 and 127")
    if args.seconds <= 0 or args.context_events < 1 or args.max_events < 1:
        parser.error("seconds, context-events, and max-events must be positive")
    random.seed(args.seed)
    generate(args)


if __name__ == "__main__":
    main()
