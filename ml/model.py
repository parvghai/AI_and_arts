"""Conditional multi-layer LSTM with separate symbolic event prediction heads."""
from __future__ import annotations

import torch
from torch import nn


class MusicLSTM(nn.Module):
    """Predict pitch, duration, velocity and delta-time for each MIDI event."""
    def __init__(self, vocab_size: int = 128, embedding_size: int = 256,
                 hidden_size: int = 512, num_layers: int = 3,
                 dropout: float = 0.25, duration_vocab: int = 16,
                 velocity_vocab: int = 8, timing_vocab: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.duration_vocab = duration_vocab
        self.velocity_vocab = velocity_vocab
        self.timing_vocab = timing_vocab
        self.pitch_embedding = nn.Embedding(vocab_size, embedding_size)
        self.duration_embedding = nn.Embedding(duration_vocab, embedding_size)
        self.velocity_embedding = nn.Embedding(velocity_vocab, embedding_size)
        self.timing_embedding = nn.Embedding(timing_vocab, embedding_size)
        self.target_embedding = nn.Embedding(vocab_size, embedding_size)
        self.tempo_projection = nn.Linear(1, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(hidden_size, vocab_size)
        self.duration_head = nn.Linear(hidden_size, duration_vocab)
        self.velocity_head = nn.Linear(hidden_size, velocity_vocab)
        self.timing_head = nn.Linear(hidden_size, 1)

    def forward(self, pitch: torch.Tensor, duration: torch.Tensor,
                velocity: torch.Tensor, timing: torch.Tensor,
                target_note: torch.Tensor, tempo: torch.Tensor,
                hidden=None):
        """Inputs are [batch, time]; tempo is BPM and target_note is MIDI pitch."""
        tempo_norm = (tempo.float().reshape(-1, 1, 1) / 120.0).expand(-1, pitch.size(1), 1)
        x = (self.pitch_embedding(pitch) + self.duration_embedding(duration)
             + self.velocity_embedding(velocity) + self.timing_embedding(timing)
             + self.target_embedding(target_note.long()).unsqueeze(1)
             + self.tempo_projection(tempo_norm))
        out, hidden = self.lstm(self.dropout(x), hidden)
        out = self.dropout(out)
        return {"pitch": self.pitch_head(out), "duration": self.duration_head(out),
                "velocity": self.velocity_head(out), "timing": self.timing_head(out).squeeze(-1)}, hidden

    def generate(self, *args, **kwargs):
        """Generation is implemented by ml.inference to share event safeguards."""
        from ml.inference import autoregressive_generate
        return autoregressive_generate(self, *args, **kwargs)
