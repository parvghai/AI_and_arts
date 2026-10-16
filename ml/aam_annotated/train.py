"""Smoke-test or, when explicitly requested, train the isolated AAM 4x768 model."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ml.aam_annotated.dataset import AAMWindowDataset
from ml.aam_annotated.model import AAMAnnotatedMusicLSTM


def _batch_loss(model, batch, device):
    cat, numeric, targets, timing, target_note = [value.to(device) for value in batch]
    outputs, _ = model(cat, numeric, target_note)
    pitch = nn.functional.cross_entropy(outputs["pitch"].reshape(-1, 128), targets[:, :, 0].reshape(-1))
    duration = nn.functional.cross_entropy(outputs["duration"].reshape(-1, 16), targets[:, :, 1].reshape(-1))
    velocity = nn.functional.cross_entropy(outputs["velocity"].reshape(-1, 8), targets[:, :, 2].reshape(-1))
    timing = nn.functional.smooth_l1_loss(outputs["timing"], timing)
    return pitch + duration + .5 * velocity + .5 * timing, {
        "pitch": float(pitch.detach()), "duration": float(duration.detach()),
        "velocity": float(velocity.detach()), "timing": float(timing.detach())}


def _load_vocab(root: Path):
    mapping = json.loads((root / "feature_mappings.json").read_text(encoding="utf-8"))
    return mapping, mapping["vocab_sizes"]


def smoke(root: Path, args) -> None:
    mapping, vocab_sizes = _load_vocab(root)
    sequence_length = int(json.loads((root / "preprocessing_config.json").read_text(encoding="utf-8"))["sequence_length"])
    train_ds = AAMWindowDataset(root, "train", sequence_length)
    val_ds = AAMWindowDataset(root, "validation", sequence_length)
    if len(train_ds) < 1 or len(val_ds) < 1:
        raise SystemExit("Smoke test requires at least one train and validation sequence.")
    batch_size = min(args.smoke_batch_size, len(train_ds))
    train_loader = DataLoader(Subset(train_ds, range(batch_size)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(Subset(val_ds, range(min(batch_size, len(val_ds)))),
                            batch_size=min(batch_size, len(val_ds)), shuffle=False)
    train_batch, val_batch = next(iter(train_loader)), next(iter(val_loader))
    cat, numeric, targets, timing, target_note = train_batch
    expected = (batch_size, sequence_length, 6)
    if tuple(cat.shape) != expected or tuple(numeric.shape) != (batch_size, sequence_length, 7):
        raise RuntimeError(f"Unexpected input tensor shapes: cat={tuple(cat.shape)}, numeric={tuple(numeric.shape)}")
    if tuple(targets.shape) != (batch_size, sequence_length, 3) or tuple(timing.shape) != (batch_size, sequence_length):
        raise RuntimeError("Unexpected output target tensor shapes.")
    if not torch.isfinite(numeric).all() or not torch.isfinite(timing).all():
        raise RuntimeError("Non-finite timing/context tensor values found.")

    use_cuda = torch.cuda.is_available()
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA smoke test requested, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    model = AAMAnnotatedMusicLSTM(vocab_sizes, embedding_size=384, hidden_size=768,
                                  num_layers=4, dropout=.30).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.smoke_learning_rate, weight_decay=.01)
    model.eval()
    with torch.no_grad():
        initial_train_loss = float(_batch_loss(model, train_batch, device)[0])
        initial_val_loss = float(_batch_loss(model, val_batch, device)[0])
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    history = []
    start = time.perf_counter()
    for step in range(1, args.smoke_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = _batch_loss(model, train_batch, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        row = {"step": step, "train_loss": float(loss.detach()), **parts}
        history.append(row)
        print(f"SMOKE step {step}/{args.smoke_steps}: loss={row['train_loss']:.4f} "
              f"pitch={row['pitch']:.4f} timing={row['timing']:.4f}")
    model.eval()
    with torch.no_grad():
        final_train_loss = float(_batch_loss(model, train_batch, device)[0])
        validation_loss = float(_batch_loss(model, val_batch, device)[0])
    if final_train_loss >= initial_train_loss:
        raise RuntimeError(f"Smoke overfit check did not reduce loss ({initial_train_loss:.4f} -> {final_train_loss:.4f}).")

    exp_models = root / "smoke" / "models"
    logs = root / "logs"
    exp_models.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    config = {"mode": "smoke_only", "layers": 4, "hidden_size": 768, "embedding_size": 384,
              "dropout": .30, "sequence_length": sequence_length, "batch_size": batch_size,
              "optimizer": "AdamW", "learning_rate": args.smoke_learning_rate,
              "seed": args.seed, "vocab_sizes": vocab_sizes,
              "categorical_features": mapping["categorical_features"],
              "numeric_features": mapping["numeric_features"], "device": str(device),
              "target_conditioning": "future_window_final_pitch",
              "loss_weights": {"pitch": 1, "duration": 1, "velocity": .5, "timing": .5}}
    checkpoint = exp_models / "aam_annotated_smoke.pt"
    torch.save({"model_state": model.state_dict(), "config": config,
                "smoke_history": history, "initial_train_loss": initial_train_loss,
                "final_train_loss": final_train_loss, "validation_loss": validation_loss}, checkpoint)
    restored = AAMAnnotatedMusicLSTM(vocab_sizes, 384, 768, 4, .30).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(saved["model_state"])
    restored.eval()
    with torch.no_grad():
        restored_loss = float(_batch_loss(restored, val_batch, device)[0])
    if abs(restored_loss - validation_loss) > 1e-5:
        raise RuntimeError("Reloaded smoke checkpoint validation output differs from the saved model.")
    peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if use_cuda else 0
    report = {"status": "passed", "cuda_available": use_cuda,
              "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
              "layers": 4, "hidden_size": 768, "embedding_size": 384,
              "dropout": .30, "trainable_parameters": parameter_count,
              "batch_size": batch_size, "sequence_length": sequence_length,
              "input_shapes": {"categorical": list(cat.shape), "numeric": list(numeric.shape)},
              "target_shapes": {"categorical": list(targets.shape), "timing": list(timing.shape)},
              "initial_train_loss": initial_train_loss, "final_train_loss": final_train_loss,
              "initial_validation_loss": initial_val_loss, "final_validation_loss": validation_loss,
              "checkpoint_reload_validation_loss": restored_loss,
              "peak_cuda_memory_bytes": peak_cuda_bytes,
              "smoke_seconds": time.perf_counter() - start,
              "checkpoint": str(checkpoint), "full_training_started": False}
    (logs / "smoke_training.json").write_text(json.dumps({"config": config, "steps": history, **report}, indent=2), encoding="utf-8")
    (logs / "smoke_validation.json").write_text(json.dumps({"initial_validation_loss": initial_val_loss,
                                                             "final_validation_loss": validation_loss,
                                                             "checkpoint_reload_validation_loss": restored_loss}, indent=2), encoding="utf-8")
    (root / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model_config = {**config, "trainable_parameters": parameter_count,
                    "model_class": "AAMAnnotatedMusicLSTM", "numeric_input_features": 7,
                    "categorical_input_features": 6}
    (root / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["status"] == "passed":
        raise SystemExit("Smoke test failed")


def full_train(root: Path, args) -> None:
    """Train the isolated annotation-aware model and evaluate its held-out test set."""
    mapping, vocab_sizes = _load_vocab(root)
    cfg = json.loads((root / "preprocessing_config.json").read_text(encoding="utf-8"))
    sequence_length = int(cfg["sequence_length"])
    train_ds = AAMWindowDataset(root, "train", sequence_length)
    val_ds = AAMWindowDataset(root, "validation", sequence_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AAMAnnotatedMusicLSTM(vocab_sizes, 384, 768, 4, .30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=.5)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
    model_dir, logs_dir = root / "models", root / "logs"
    model_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "full", "layers": 4, "hidden_size": 768, "embedding_size": 384,
              "dropout": .30, "sequence_length": sequence_length, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "vocab_sizes": vocab_sizes, "categorical_features": mapping["categorical_features"],
              "numeric_features": mapping["numeric_features"], "target_conditioning": "future_window_final_pitch",
              "optimizer": "AdamW", "weight_decay": .01,
              "loss_weights": {"pitch": 1.0, "duration": 1.0, "velocity": .5, "timing": .5},
              "mixed_precision": "float16 autocast + GradScaler on CUDA"}
    (root / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model_config = {**config,
                    "model_class": "AAMAnnotatedMusicLSTM",
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "categorical_input_features": 6, "numeric_input_features": 7,
                    "device": str(device)}
    (root / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    best, stale = float("inf"), 0
    best_epoch = 0
    train_log = (logs_dir / "training.jsonl").open("w", encoding="utf-8")
    validation_log = (logs_dir / "validation.jsonl").open("w", encoding="utf-8")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); total = torch.zeros((), device=device)
            for batch in train_loader:
                batch = [v.to(device, non_blocking=use_cuda) for v in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                    loss, _ = _batch_loss(model, batch, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); total += loss.detach()
            train_loss = float(total.cpu()) / max(1, len(train_loader))
            model.eval(); vtotal = torch.zeros((), device=device)
            with torch.no_grad():
                for batch in val_loader:
                    batch = [v.to(device, non_blocking=use_cuda) for v in batch]
                    vloss, _ = _batch_loss(model, batch, device)
                    if not torch.isfinite(vloss):
                        raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
                    vtotal += vloss
            val_loss = float(vtotal.cpu()) / max(1, len(val_loader)); scheduler.step(val_loss)
            epoch_row = {"epoch": epoch, "train_loss": train_loss,
                         "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
            train_log.write(json.dumps(epoch_row) + "\n"); train_log.flush()
            validation_log.write(json.dumps(epoch_row) + "\n"); validation_log.flush()
            checkpoint = {"model_state": model.state_dict(),
                          "epoch": epoch, "config": config, "validation_loss": val_loss,
                          "best_epoch_so_far": best_epoch}
            tmp_latest = model_dir / "aam_annotated_latest.tmp.pt"
            torch.save(checkpoint, tmp_latest)
            tmp_latest.replace(model_dir / "aam_annotated_latest.pt")
            if val_loss < best:
                best, stale = val_loss, 0
                best_epoch = epoch
                checkpoint["best_epoch_so_far"] = best_epoch
                tmp_best = model_dir / "aam_annotated_best.tmp.pt"
                torch.save(checkpoint, tmp_best)
                tmp_best.replace(model_dir / "aam_annotated_best.pt")
            else:
                stale += 1
            print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.4f} val={val_loss:.4f}")
            if stale >= 6:
                print("Early stopping after six validation epochs without improvement.")
                break
    finally:
        train_log.close(); validation_log.close()
    if best_epoch == 0 or not (model_dir / "aam_annotated_best.pt").exists():
        raise RuntimeError("Training ended without a best-validation checkpoint.")
    print(f"Best validation checkpoint: epoch {best_epoch}, loss={best:.4f}")
    from ml.aam_annotated.evaluate import evaluate_checkpoint
    evaluate_checkpoint(root, model_dir / "aam_annotated_best.pt", args.evaluation_batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/aam_annotated"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke",
                        help="Defaults to the bounded smoke test; full mode is explicit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=12)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--smoke-learning-rate", type=float, default=.001)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.mode == "smoke":
        smoke(args.experiment, args)
    else:
        full_train(args.experiment, args)


if __name__ == "__main__":
    main()
