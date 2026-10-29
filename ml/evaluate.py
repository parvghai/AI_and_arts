"""Evaluate frozen checkpoints with masks; preserve historical padded reports."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ml.dataset import EventWindowDataset, split_files
from ml.model import MusicLSTM
from ml.metrics import Metrics
from ml.preprocessing import decode_timing

def evaluate(data, checkpoint, output, batch_size=32, adapter=None):
    output=Path(output)
    if output.exists(): raise ValueError("Choose a new report path; historical results are preserved")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=state['config']
    if cfg.get('target_conditioning')!='future_window_final_pitch':raise ValueError('Incompatible endpoint conditioning')
    model=MusicLSTM(embedding_size=cfg['embedding_size'],hidden_size=cfg['hidden_size'],
                    num_layers=cfg['layers'],dropout=cfg['dropout']).to(device)
    model.load_state_dict(state['model_state']);model.eval()
    if adapter:
        from ml.lora import apply_adapter
        apply_adapter(model,adapter,checkpoint)
    parts=split_files(sorted(Path(data).glob('*.jsonl')),cfg.get('seed',42))
    ds=EventWindowDataset(parts[2],cfg['sequence_length'])
    masked,legacy=Metrics(),Metrics()
    lut=torch.tensor([decode_timing(i) for i in range(16)],device=device)
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=batch_size):
            x,y,target,tempo,mask=[v.to(device) for v in batch]
            out,_=model(x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],target,tempo)
            predicted=lut[(out['timing']*7.5).round().long().clamp(0,15)]
            target_gap=lut[y[:,:,3]]
            for metrics,valid in ((masked,mask),(legacy,torch.ones_like(mask))):
                metrics.add(out,y[:,:,:3],y[:,:,3].float()/7.5,valid,predicted,target_gap)
    report={'protocol':'masked-v1','checkpoint':str(checkpoint),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'split_files':[len(p) for p in parts], 'test_windows':len(ds),
            'split_filename_sha256':hashlib.sha256('\n'.join(p.name for p in parts[2]).encode()).hexdigest(),
            'config':cfg,'device':str(device),'torch_version':str(torch.__version__),
            'real_targets':masked.report(),'legacy_including_padding':legacy.report(),
            'adapter':str(adapter) if adapter else None,
            'adapter_sha256':hashlib.sha256(Path(adapter).read_bytes()).hexdigest() if adapter else None,
            'note':'Teacher forced, gold endpoint conditioned; MAE uses quantized representatives. Evaluation does not train weights.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,default=Path('reports/masked_baseline'))
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--adapter',type=Path)
    a=p.parse_args();evaluate(a.data,a.checkpoint or a.models/'best_model_3x512.pt',a.output/'evaluation_report.json',a.batch_size,a.adapter)
if __name__=='__main__':main()
