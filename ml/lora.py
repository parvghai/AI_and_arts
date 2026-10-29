"""Low-rank updates for MusicLSTM; original checkpoint tensors stay frozen."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import parametrize
from ml.model import MusicLSTM

FORMAT = 'music-lstm-lora-v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        return weight + (self.B @ self.A) * self.scale


def target_names(model):
    return [f'lstm.weight_{kind}_l{i}' for i in range(model.lstm.num_layers)
            for kind in ('ih', 'hh')] + [f'{name}.weight' for name in
            ('tempo_projection', 'pitch_head', 'duration_head', 'velocity_head', 'timing_head')]


def attach_lora(model, rank=8, alpha=16.0):
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 64:
        raise ValueError('LoRA rank must be an integer between 1 and 64')
    if not math.isfinite(alpha) or not 0 < alpha <= 1024:
        raise ValueError('LoRA alpha must be finite and in (0, 1024]')
    targets = target_names(model)
    if any(parametrize.is_parametrized(model.get_submodule(n.rsplit('.', 1)[0])) for n in targets):
        raise ValueError('Model already has adapters')
    model.requires_grad_(False)
    for name in targets:
        parent, attr = name.rsplit('.', 1)
        module = model.get_submodule(parent)
        parametrize.register_parametrization(module, attr, LowRankWeight(getattr(module, attr), rank, alpha))
    return targets


def adapter_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if '.parametrizations.' in name and name.endswith(('.0.A', '.0.B'))}


def save_adapter(path, model, base_checkpoint, rank, alpha, metadata=None):
    """Save only adapters, never base weights or optimizer state."""
    state = {k: v.detach().cpu().clone() for k, v in adapter_parameters(model).items()}
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Missing or non-finite adapter weights')
    payload = {'format': FORMAT, 'base_sha256': sha256(base_checkpoint),
               'base_filename': Path(base_checkpoint).name, 'rank': rank, 'alpha': float(alpha),
               'targets': target_names(model), 'adapter_state': state, 'metadata': metadata or {}}
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Adapter already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)


def read_adapter(path):
    if Path(path).stat().st_size > 128 * 1024**2:
        raise ValueError('Adapter exceeds 128 MiB limit')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('Unsupported LoRA adapter format')
    return payload


def apply_adapter(model, path, base_checkpoint):
    """Validate on a temporary model, then merge deltas into the inference model."""
    payload = read_adapter(path)
    if payload.get('base_sha256') != sha256(base_checkpoint):
        raise ValueError('LoRA adapter does not match this base checkpoint (SHA-256)')
    if payload.get('targets') != target_names(model):
        raise ValueError('LoRA targets do not match this model')
    # Construct adapters on a copy: malformed files cannot partly mutate the caller.
    import copy
    adapted = copy.deepcopy(model)
    attach_lora(adapted, payload['rank'], payload['alpha'])
    parameters = adapter_parameters(adapted)
    state = payload.get('adapter_state', {})
    if set(state) != set(parameters):
        raise ValueError('Missing or unexpected adapter tensors')
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[name].shape:
            raise ValueError(f'Invalid adapter tensor: {name}')
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite or non-floating adapter tensor: {name}')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])
    for name in payload['targets']:
        parent, attr = name.rsplit('.', 1)
        parametrize.remove_parametrizations(adapted.get_submodule(parent), attr, leave_parametrized=True)
    merged = adapted.state_dict()
    if not all(torch.isfinite(t).all() for t in merged.values()):
        raise ValueError('Merged LoRA weights are non-finite')
    model.load_state_dict(merged)
    model.lstm.flatten_parameters()
    return payload


def load_base(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = state['config']
    if config.get('target_conditioning') != 'future_window_final_pitch':
        raise ValueError('Expected a MIDI-only MusicLSTM checkpoint')
    model = MusicLSTM(embedding_size=config['embedding_size'], hidden_size=config['hidden_size'],
                      num_layers=config['layers'], dropout=config['dropout'])
    model.load_state_dict(state['model_state'], strict=True)
    return model.to(device), config
