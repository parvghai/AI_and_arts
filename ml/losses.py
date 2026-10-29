"""Losses over real target positions only (protocol masked-v1)."""
import torch
from torch.nn import functional as F


def masked_mean(values, mask):
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("Batch has no real targets")
    return values.masked_select(mask).mean()


def event_loss(outputs, targets, timing, mask):
    parts = {name: masked_mean(F.cross_entropy(outputs[name].transpose(1, 2),
                                               targets[:, :, i], reduction='none'), mask)
             for i, name in enumerate(('pitch', 'duration', 'velocity'))}
    parts['timing'] = masked_mean(F.smooth_l1_loss(outputs['timing'].float(), timing.float(),
                                                 reduction='none'), mask)
    total = parts['pitch'] + parts['duration'] + .5*parts['velocity'] + .5*parts['timing']
    return total, parts
