"""Windowed symbolic-event dataset and deterministic file-level splits."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep split inspection available when only preprocessing is installed.
    torch = None
    class Dataset:  # type: ignore[no-redef]
        pass
from ml.preprocessing import Event, encode_event


def split_files(files: list[Path], seed: int = 42) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic 80/10/10 file split; group by inferred track stem when possible."""
    groups: dict[str, list[Path]] = {}
    for f in files:
        # Multi-track exports commonly share a stem prefix before track/part suffixes.
        aam_id = re.match(r"^(\d{3,})[_-]", f.stem)
        key = f"aam:{aam_id.group(1)}" if aam_id else f.stem.split("_track")[0].split("_part")[0]
        groups.setdefault(key, []).append(f)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) >= 3:
        train_cut = min(len(keys) - 2, max(1, int(len(keys) * .8)))
        val_cut = min(len(keys) - 1, max(train_cut + 1, int(len(keys) * .9)))
    else:
        train_cut, val_cut = max(1, len(keys) - 2), max(1, len(keys) - 1)
    parts = [keys[:train_cut], keys[train_cut:val_cut], keys[val_cut:]]
    return tuple([p for key in bucket for p in groups[key]] for bucket in parts)  # type: ignore[return-value]


def count_event_rows(files: list[Path]) -> int:
    """Count JSONL records in chunks without parsing or retaining event data."""
    total = 0
    for path in files:
        with path.open("rb") as stream:
            last = b""
            while chunk := stream.read(1024 * 1024):
                total += chunk.count(b"\n")
                last = chunk[-1:]
            if last and last != b"\n":
                total += 1
    return total


def count_windows(files: list[Path], sequence_length: int) -> int:
    """Count generated training windows using the same stride as EventWindowDataset."""
    windows = 0
    for path in files:
        with path.open("rb") as stream:
            rows, last = 0, b""
            while chunk := stream.read(1024 * 1024):
                rows += chunk.count(b"\n")
                last = chunk[-1:]
            if last and last != b"\n":
                rows += 1
        if rows >= 3:
            windows += (rows - 3) // sequence_length + 1
    return windows


class EventWindowDataset(Dataset):
    def __init__(self, files: list[Path], sequence_length: int = 128) -> None:
        if torch is None:
            raise RuntimeError("Training sequences require PyTorch; install requirements.txt first.")
        self.windows: list[torch.Tensor] = []
        self.sequence_length = sequence_length
        for path in files:
            try:
                events = [Event(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
                encoded = [encode_event(event) for event in events]
            except (OSError, ValueError, TypeError):
                continue
            # Include a final short track by discarding too-short files only.
            for start in range(0, max(0, len(encoded) - 1), sequence_length):
                seq = encoded[start:start + sequence_length + 1]
                if len(seq) >= 3:
                    if len(seq) < sequence_length + 1:
                        seq += [seq[-1]] * (sequence_length + 1 - len(seq))
                    self.windows.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        row = self.windows[index]
        x, y = row[:-1], row[1:]
        # Condition on the phrase's future endpoint so the LSTM learns to shape
        # the continuation toward a target that is not already in its context.
        target = y[-1, 0].clone()
        tempo = torch.tensor(100.0)
        return x, y, target, tempo
