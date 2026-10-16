"""Train the conditional symbolic MusicLSTM from local MIDI files."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from ml.dataset import EventWindowDataset, split_files, count_event_rows, count_windows
from ml.model import MusicLSTM
from ml.preprocessing import preprocess_directory


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Local MIDI directory or preprocessed JSONL directory")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=256); p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--embedding-size", type=int, default=384); p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.30); p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--top-k", type=int, default=20); p.add_argument("--top-p", type=float, default=.90)
    p.add_argument("--model-name", default=None, help="Checkpoint stem, e.g. best_model_4x768")
    p.add_argument("--temperature", type=float, default=.8); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("."))
    args = p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print(f"Training GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    else:
        print("CUDA unavailable; training on CPU")
    processed = args.data
    if (any(args.data.rglob("*.mid")) or any(args.data.rglob("*.midi"))
            or any(args.data.rglob("*.zip"))):
        processed = args.output / "data" / "processed"
        n = preprocess_directory(args.data, processed)
        print(f"Preprocessed {n} MIDI files")
    files = sorted(processed.glob("*.jsonl"))
    if len(files) < 3:
        raise SystemExit("Need at least 3 processed MIDI files for train/validation/test splits.")
    train_files, val_files, test_files = split_files(files, args.seed)
    report_path = processed / "preprocessing_report.json"
    prep_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    event_total = count_event_rows(files)
    sequence_counts = {"train":count_windows(train_files,args.sequence_length),
                       "validation":count_windows(val_files,args.sequence_length),
                       "test":count_windows(test_files,args.sequence_length)}
    dataset_bytes = sum(p.stat().st_size for p in files)
    dataset_summary = {"midi_files_discovered":prep_report.get("discovered_midi_files",len(files)),
                       "valid_midi_files":len(files),"invalid_midi_files":prep_report.get("invalid_files",0),
                       "training_files":len(train_files),"validation_files":len(val_files),
                       "test_files":len(test_files),"total_events":event_total,
                       "training_sequences":sequence_counts["train"],
                       "validation_sequences":sequence_counts["validation"],
                       "test_sequences":sequence_counts["test"],
                       "total_sequences":sum(sequence_counts.values()),
                       "vocabulary_sizes":{"pitch":128,"duration":16,"velocity":8,"timing":16},
                       "approximate_dataset_bytes":dataset_bytes,"seed":args.seed}
    print("Dataset summary:")
    print(f"  MIDI: {dataset_summary['midi_files_discovered']} discovered, {len(files)} valid, {dataset_summary['invalid_midi_files']} invalid")
    print(f"  Split files: train={len(train_files)}, validation={len(val_files)}, test={len(test_files)}")
    print(f"  Events: {event_total:,}; windows/sequences: train={sequence_counts['train']:,}, validation={sequence_counts['validation']:,}, test={sequence_counts['test']:,}")
    print(f"  Vocabulary: {dataset_summary['vocabulary_sizes']}; processed size≈{dataset_bytes/1024/1024:.1f} MiB")
    # Pick the longest context that fits a conservative activation-memory estimate.
    if cuda and args.sequence_length == 256:
        free_bytes, _ = torch.cuda.mem_get_info()
        estimate = args.batch_size * args.sequence_length * args.hidden_size * args.layers * 64
        if estimate > free_bytes * .68:
            args.sequence_length = 192 if estimate * .75 <= free_bytes * .68 else 128
            print(f"Estimated activation memory exceeds free VRAM; using sequence_length={args.sequence_length}")
    sequence_counts = {"train":count_windows(train_files,args.sequence_length),
                       "validation":count_windows(val_files,args.sequence_length),
                       "test":count_windows(test_files,args.sequence_length)}
    dataset_summary.update({"sequence_length":args.sequence_length,
                            "training_sequences":sequence_counts["train"],
                            "validation_sequences":sequence_counts["validation"],
                            "test_sequences":sequence_counts["test"],
                            "total_sequences":sum(sequence_counts.values())})
    train_ds = EventWindowDataset(train_files, args.sequence_length)
    val_ds = EventWindowDataset(val_files, args.sequence_length)
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("Training or validation split has no usable event windows; add MIDI files.")
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=cuda)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model = MusicLSTM(embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                      num_layers=args.layers, dropout=args.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model parameters: {parameter_count / 1_000_000:.2f} million")
    print(f"Batch size: {args.batch_size}; sequence length: {args.sequence_length}; CUDA: {cuda}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=.5)
    ce = nn.CrossEntropyLoss()
    weights = {"pitch": 1.0, "duration": 1.0, "velocity": .5, "timing": .5}
    amp_enabled = cuda
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # Backward compatible with earlier supported PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def epoch(loader: DataLoader, train: bool) -> float:
        model.train(train); total = seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (x, y, target, tempo) in enumerate(loader):
            x, y = x.to(device, non_blocking=cuda), y.to(device, non_blocking=cuda)
            target, tempo = target.to(device, non_blocking=cuda), tempo.to(device, non_blocking=cuda)
            kwargs = dict(pitch=x[:,:,0], duration=x[:,:,1], velocity=x[:,:,2], timing=x[:,:,3],
                          target_note=target, tempo=tempo)
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    out, _ = model(**kwargs)
                    losses = (weights["pitch"]*ce(out["pitch"].reshape(-1,128), y[:,:,0].reshape(-1))
                              + weights["duration"]*ce(out["duration"].reshape(-1,16), y[:,:,1].reshape(-1))
                              + weights["velocity"]*ce(out["velocity"].reshape(-1,8), y[:,:,2].reshape(-1))
                              + weights["timing"]*nn.functional.smooth_l1_loss(out["timing"], y[:,:,3].float()/7.5))
                if train:
                    scaler.scale(losses / args.gradient_accumulation).backward()
                    if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                        optimizer.zero_grad(set_to_none=True)
            total += float(losses.detach()) * len(x); seen += len(x)
        return total / max(1, seen)

    models = args.output / "models"; logs = args.output / "logs"
    models.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    (logs / "dataset_summary.json").write_text(json.dumps(dataset_summary,indent=2),encoding="utf-8")
    config = vars(args).copy(); config["data"] = str(args.data); config["output"] = str(args.output)
    config["top_k"] = args.top_k; config["top_p"] = args.top_p
    config.update({"vocab_size":128,"duration_vocab":16,"velocity_vocab":8,"timing_vocab":16,
                   "temperature":args.temperature,"device":str(device),"loss_weights":weights,
                   "target_conditioning":"future_window_final_pitch"})
    (models / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = logs / "training_history.csv"
    best, stale = float("inf"), 0
    model_stem = args.model_name or f"best_model_{args.layers}x{args.hidden_size}"
    best_path = models / f"{model_stem}.pt"
    latest_path = models / f"latest_model_{args.layers}x{args.hidden_size}.pt"
    legacy_best = models / "best_model.pt"
    if args.layers == 4 and args.hidden_size == 768 and legacy_best.exists():
        legacy_state = torch.load(legacy_best, map_location="cpu", weights_only=False)
        legacy_cfg = legacy_state.get("config", {})
        if legacy_cfg.get("layers") == 3 and legacy_cfg.get("hidden_size") == 512:
            torch.save(legacy_state, models / "best_model_3x512.pt")
    with history.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["epoch","training_loss","validation_loss","learning_rate"])
        for e in range(1, args.epochs + 1):
            tr, vl = epoch(train_loader, True), epoch(val_loader, False)
            scheduler.step(vl); lr = optimizer.param_groups[0]["lr"]
            writer.writerow([e, tr, vl, lr]); f.flush()
            checkpoint = {"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                          "scheduler_state":scheduler.state_dict(),"epoch":e,"validation_loss":vl,"config":config}
            torch.save(checkpoint, latest_path)
            if vl < best:
                best, stale = vl, 0; torch.save(checkpoint, best_path)
            else:
                stale += 1
            print(f"Epoch {e}/{args.epochs}: train={tr:.4f} val={vl:.4f} lr={lr:.2g}")
            if stale >= 6:
                print("Early stopping after 6 epochs without validation improvement."); break
    try:
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(history.open(encoding="utf-8")))
        plt.plot([float(r["training_loss"]) for r in rows], label="train")
        plt.plot([float(r["validation_loss"]) for r in rows], label="validation")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(logs / "training_loss.png"); plt.close()
    except ImportError:
        print("matplotlib unavailable; skipped loss plot")
    if args.layers == 4 and args.hidden_size == 768 and best_path.exists():
        # Keep the legacy default path usable while retaining explicit model files.
        torch.save(torch.load(best_path, map_location="cpu", weights_only=False), legacy_best)
    print(f"Test files reserved: {len(test_files)}. Checkpoint: {best_path}")
    print(f"Run python -m ml.evaluate --data {processed} --models {models} --checkpoint {best_path}")


if __name__ == "__main__":
    main()
