"""Same-seed timeline regression for baseline and experimental LSTMs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch

from composition import Composition, NoteEvent, human_note_duration, validate_timeline
from ml.inference import LSTMGenerator

EXPECTED = {
    "3x512": ("best_model_3x512.pt", (3, 512, 256, .25)),
    "4x768": ("best_model_4x768.pt", (4, 768, 384, .30)),
}


def fixture() -> Composition:
    """A fixed, valid, monophonic prefix shared by every model/duration run."""
    composition = Composition()
    composition.events = [
        NoteEvent(60, 0.00, .18, 96, "human", "C"),
        NoteEvent(62, 0.22, .16, 78, "ai", "D"),
        NoteEvent(64, 0.42, .18, 82, "ai", "E"),
        NoteEvent(67, 0.64, .18, 86, "ai", "G"),
    ]
    return composition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("logs/timing_regression.json"))
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--target", choices=("C", "D", "E", "F", "G", "A", "B"), default="A")
    parser.add_argument("--bpm", type=int, default=100)
    args = parser.parse_args()
    report = {"seed": args.seed, "target": args.target, "bpm": args.bpm,
              "starting_composition": [event.__dict__ for event in fixture().events], "runs": []}
    for label, (filename, expected) in EXPECTED.items():
        generator = LSTMGenerator(args.models / filename)
        if not generator.loaded:
            raise SystemExit(f"Failed to load exact {label} checkpoint: {generator.load_error}")
        actual = (int(generator.config.get("layers", -1)), int(generator.config.get("hidden_size", -1)),
                  int(generator.config.get("embedding_size", -1)), float(generator.config.get("dropout", -1)))
        if actual != expected:
            raise SystemExit(f"{filename} architecture mismatch: got {actual}, expected {expected}")
        for seconds in (.5, 1.0, 2.0):
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            composition = fixture()
            boundary = composition.end_time + human_note_duration(args.bpm)
            started = time.perf_counter()
            generated = generator.generate_continuation(composition, args.target, seconds, args.bpm)
            latency = time.perf_counter() - started
            timeline = validate_timeline(generated, boundary, ("ai",))
            generated_payload = [event.__dict__.copy() for event in generated]
            interaction = composition.add_interaction(args.target, generated, seconds, args.bpm)
            appended_timeline = validate_timeline(interaction.events, monophonic_sources=("ai",))
            global_timeline = validate_timeline(composition.events, monophonic_sources=("ai",))
            run = {"model": label, "checkpoint": str(generator.checkpoint),
                   "architecture": actual, "duration_seconds": seconds, "latency_seconds": latency,
                   "composition_end": composition.end_time,
                   "continuation_boundary": boundary,
                   "events": generated_payload,
                   "timeline_validation": timeline,
                   "interaction_after_append_validation": appended_timeline,
                   "absolute_composition_validation": global_timeline,
                   "inference_diagnostics": getattr(generator, "last_timeline_diagnostics", None)}
            report["runs"].append(run)
            print(f"REGRESSION {label} {seconds:.1f}s: events={len(generated)} "
                  f"valid={timeline['valid']} overlap={timeline['overlap_detected']} "
                  f"latency={latency:.3f}s")
            if not timeline["valid"] or not appended_timeline["valid"] or not global_timeline["valid"]:
                raise SystemExit(f"Timeline regression failed for {label} {seconds:.1f}s: "
                                 f"{json.dumps({'generated': timeline, 'interaction': appended_timeline, 'composition': global_timeline})}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Same-seed baseline/experimental regression passed; report saved to {args.output}")


if __name__ == "__main__":
    main()
