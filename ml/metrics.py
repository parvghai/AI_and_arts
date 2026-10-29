"""Shared real-target and legacy padded metrics from the same forward passes."""
import torch
from ml.losses import event_loss
from ml.preprocessing import decode_duration, decode_velocity


class Metrics:
    def __init__(self):
        self.count = 0
        self.sums = {k: 0. for k in ('loss','pitch_loss','duration_loss','velocity_loss',
                                    'timing_loss','pitch_accuracy','duration_mae_seconds',
                                    'velocity_mae','timing_mae_seconds')}

    def add(self,out,targets,timing,mask,predicted_gap,target_gap):
        count = int(mask.sum())
        loss,parts = event_loss(out,targets,timing,mask)
        self.count += count
        values = {'loss':loss, **{k+'_loss':v for k,v in parts.items()}}
        duration=torch.tensor([decode_duration(i) for i in range(16)],device=targets.device,dtype=torch.float32)
        velocity=torch.tensor([decode_velocity(i) for i in range(8)],device=targets.device,dtype=torch.float32)
        errors={'pitch_accuracy':(out['pitch'].argmax(-1)==targets[:,:,0]).float(),
                'duration_mae_seconds':(duration[out['duration'].argmax(-1)]-duration[targets[:,:,1]]).abs(),
                'velocity_mae':(velocity[out['velocity'].argmax(-1)]-velocity[targets[:,:,2]]).abs(),
                'timing_mae_seconds':(predicted_gap-target_gap).abs()}
        values.update({k:v[mask].mean() for k,v in errors.items()})
        for key,value in values.items():self.sums[key]+=float(value)*count

    def report(self):
        if not self.count:raise ValueError('No usable test targets')
        return {'target_events':self.count,**{k:v/self.count for k,v in self.sums.items()}}
