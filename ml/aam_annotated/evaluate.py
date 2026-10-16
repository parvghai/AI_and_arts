"""Evaluate a trained AAM annotation-aware checkpoint on the untouched test split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.aam_annotated.dataset import AAMWindowDataset
from ml.aam_annotated.model import AAMAnnotatedMusicLSTM
from ml.preprocessing import DURATION_EDGES, VELOCITY_EDGES


def _decode_duration(token: torch.Tensor) -> torch.Tensor:
    edges = (0.0, *DURATION_EDGES)
    values = []
    for value in range(len(DURATION_EDGES) + 1):
        if value == 0:
            values.append(DURATION_EDGES[0] / 2)
        elif value == len(DURATION_EDGES):
            values.append(DURATION_EDGES[-1])
        else:
            values.append((edges[value] + edges[value + 1]) / 2)
    return torch.tensor(values, dtype=torch.float32, device=token.device)[token.long()]


def _decode_velocity(token: torch.Tensor) -> torch.Tensor:
    values = [8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0]
    return torch.tensor(values, dtype=torch.float32, device=token.device)[token.long()]


def _decode_gap_seconds(norm: torch.Tensor, tempo_norm: torch.Tensor) -> torch.Tensor:
    beats = torch.expm1(norm.clamp(0.0, 1.0) * math.log1p(16.0))
    bpm = (tempo_norm * 240.0).clamp(30.0, 240.0)
    return beats * (60.0 / bpm)


def evaluate_checkpoint(root: Path, checkpoint_path: Path, batch_size: int = 16) -> dict:
    root = Path(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    mapping = json.loads((root / "feature_mappings.json").read_text(encoding="utf-8"))
    model = AAMAnnotatedMusicLSTM(mapping["vocab_sizes"], cfg["embedding_size"],
                                  cfg["hidden_size"], cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    sequence_length = int(cfg["sequence_length"])
    test_ds = AAMWindowDataset(root, "test", sequence_length)
    if not len(test_ds):
        raise RuntimeError("Held-out test split contains no usable windows.")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    loss_weights = {"pitch": 1.0, "duration": 1.0, "velocity": 0.5, "timing": 0.5}
    sums = {key: 0.0 for key in ("loss", "pitch_loss", "duration_loss", "velocity_loss",
                                  "timing_loss", "pitch_correct", "duration_abs_error_seconds",
                                  "velocity_abs_error", "timing_abs_error_seconds")}
    count = 0
    with torch.inference_mode():
        for batch in loader:
            cat, numeric, targets, timing_target, target_note = [v.to(device, non_blocking=device.type == "cuda") for v in batch]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs, _ = model(cat, numeric, target_note)
                pitch_loss = nn.functional.cross_entropy(outputs["pitch"].flatten(0, 1), targets[:, :, 0].reshape(-1))
                duration_loss = nn.functional.cross_entropy(outputs["duration"].flatten(0, 1), targets[:, :, 1].reshape(-1))
                velocity_loss = nn.functional.cross_entropy(outputs["velocity"].flatten(0, 1), targets[:, :, 2].reshape(-1))
                timing_loss = nn.functional.smooth_l1_loss(outputs["timing"].float(), timing_target.float())
                total_loss = (loss_weights["pitch"] * pitch_loss + loss_weights["duration"] * duration_loss
                              + loss_weights["velocity"] * velocity_loss + loss_weights["timing"] * timing_loss)
            n = int(targets[:, :, 0].numel())
            count += n
            for key, value in (("loss", total_loss), ("pitch_loss", pitch_loss),
                               ("duration_loss", duration_loss), ("velocity_loss", velocity_loss),
                               ("timing_loss", timing_loss)):
                sums[key] += float(value) * n
            pred_pitch = outputs["pitch"].argmax(-1)
            pred_duration = outputs["duration"].argmax(-1)
            pred_velocity = outputs["velocity"].argmax(-1)
            sums["pitch_correct"] += int((pred_pitch == targets[:, :, 0]).sum())
            sums["duration_abs_error_seconds"] += float((_decode_duration(pred_duration) - _decode_duration(targets[:, :, 1])).abs().sum())
            sums["velocity_abs_error"] += float((_decode_velocity(pred_velocity) - _decode_velocity(targets[:, :, 2])).abs().sum())
            predicted_gap = _decode_gap_seconds(outputs["timing"].float(), numeric[:, :, 4])
            target_gap = _decode_gap_seconds(timing_target.float(), numeric[:, :, 4])
            sums["timing_abs_error_seconds"] += float((predicted_gap - target_gap).abs().sum())
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    report = {
        "status": "completed", "checkpoint": str(checkpoint_path), "device": str(device),
        "cuda_available": torch.cuda.is_available(), "test_windows": len(test_ds),
        "test_target_events": count, "test_loss": metrics["loss"],
        "pitch_loss": metrics["pitch_loss"], "duration_loss": metrics["duration_loss"],
        "velocity_loss": metrics["velocity_loss"], "timing_loss": metrics["timing_loss"],
        "pitch_accuracy": metrics["pitch_correct"],
        "duration_mae_seconds": metrics["duration_abs_error_seconds"],
        "velocity_mae": metrics["velocity_abs_error"],
        "timing_mae_seconds": metrics["timing_abs_error_seconds"],
        "validation_loss_at_best_epoch": float(state["validation_loss"]),
        "best_epoch": int(state["epoch"]), "source_level_test_groups": 300,
        "full_training_started": True,
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/aam_annotated"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    checkpoint = args.checkpoint or (args.experiment / "models" / "aam_annotated_best.pt")
    evaluate_checkpoint(args.experiment, checkpoint, args.batch_size)


if __name__ == "__main__":
    main()
