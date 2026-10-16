"""Annotation-aware 4x768 LSTM with explicit categorical and numeric encoders."""
from __future__ import annotations

import torch
from torch import nn


class AAMAnnotatedMusicLSTM(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], embedding_size: int = 384,
                 hidden_size: int = 768, num_layers: int = 4, dropout: float = .30,
                 numeric_features: int = 7):
        super().__init__()
        self.vocab_sizes = dict(vocab_sizes)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pitch_embedding = nn.Embedding(vocab_sizes["pitch"], embedding_size)
        self.duration_embedding = nn.Embedding(vocab_sizes["duration"], embedding_size)
        self.velocity_embedding = nn.Embedding(vocab_sizes["velocity"], embedding_size)
        self.instrument_embedding = nn.Embedding(vocab_sizes["instrument"], embedding_size, padding_idx=0)
        self.chord_embedding = nn.Embedding(vocab_sizes["chord"], embedding_size, padding_idx=0)
        self.key_embedding = nn.Embedding(vocab_sizes["key"], embedding_size, padding_idx=0)
        self.numeric_projection = nn.Linear(numeric_features, embedding_size)
        self.target_embedding = nn.Embedding(vocab_sizes["pitch"], embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(hidden_size, vocab_sizes["pitch"])
        self.duration_head = nn.Linear(hidden_size, vocab_sizes["duration"])
        self.velocity_head = nn.Linear(hidden_size, vocab_sizes["velocity"])
        self.timing_head = nn.Linear(hidden_size, 1)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor,
                target_note: torch.Tensor, hidden=None):
        """categorical [B,T,6], numeric [B,T,7], scalar target pitch [B]."""
        x = (self.pitch_embedding(categorical[:, :, 0])
             + self.duration_embedding(categorical[:, :, 1])
             + self.velocity_embedding(categorical[:, :, 2])
             + self.instrument_embedding(categorical[:, :, 3])
             + self.chord_embedding(categorical[:, :, 4])
             + self.key_embedding(categorical[:, :, 5])
             + self.numeric_projection(numeric.float())
             + self.target_embedding(target_note.long()).unsqueeze(1))
        out, hidden = self.lstm(self.dropout(x), hidden)
        out = self.dropout(out)
        return {"pitch": self.pitch_head(out),
                "duration": self.duration_head(out),
                "velocity": self.velocity_head(out),
                "timing": self.timing_head(out).squeeze(-1)}, hidden
