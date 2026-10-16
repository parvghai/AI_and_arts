"""Memory-mapped grouped windows for the AAM annotation-aware experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AAMWindowDataset(Dataset):
    def __init__(self, experiment_dir: Path, split: str, sequence_length: int = 256):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(experiment_dir)
        self.split = split
        self.sequence_length = int(sequence_length)
        data_dir = self.root / "data"
        cat_shape = json.loads((data_dir / f"{split}_categorical.shape.json").read_text(encoding="utf-8"))
        num_shape = json.loads((data_dir / f"{split}_numeric.shape.json").read_text(encoding="utf-8"))
        self.event_count = int(cat_shape["events"])
        self.cat_features = int(cat_shape["features"])
        self.num_features = int(num_shape["features"])
        self.cat = np.memmap(data_dir / f"{split}_categorical.bin", dtype=np.uint16,
                             mode="r", shape=(self.event_count, self.cat_features))
        self.numeric = np.memmap(data_dir / f"{split}_numeric.bin", dtype=np.float32,
                                 mode="r", shape=(self.event_count, self.num_features))
        self.windows: list[tuple[int, int]] = []
        with (data_dir / f"{split}_index.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                count, start = int(record["count"]), int(record["start"])
                if record.get("validation_status") != "valid" or count < 3:
                    continue
                for offset in range(0, max(0, count - 1), self.sequence_length):
                    if min(self.sequence_length + 1, count - offset) >= 3:
                        self.windows.append((start + offset, min(self.sequence_length + 1, count - offset)))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        start, available = self.windows[index]
        cat = np.array(self.cat[start:start + available], dtype=np.int64, copy=True)
        numeric = np.array(self.numeric[start:start + available], dtype=np.float32, copy=True)
        needed = self.sequence_length + 1
        if available < needed:
            cat = np.concatenate((cat, np.repeat(cat[-1:], needed - available, axis=0)), axis=0)
            numeric = np.concatenate((numeric, np.repeat(numeric[-1:], needed - available, axis=0)), axis=0)
        x_cat, y_cat = cat[:-1], cat[1:]
        x_num, y_num = numeric[:-1], numeric[1:]
        # The existing target-note conditioning concept is retained, but every
        # musical context feature on x remains causal (features for event t only).
        target_note = int(y_cat[-1, 0])
        return (torch.from_numpy(x_cat), torch.from_numpy(x_num),
                torch.from_numpy(y_cat[:, :3].copy()),
                torch.from_numpy(y_num[:, 0].copy()),
                torch.tensor(target_note, dtype=torch.long))
