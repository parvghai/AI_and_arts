"""Real-target annotation evaluation; original reports are never overwritten."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ml.aam_annotated.dataset import AAMWindowDataset
from ml.aam_annotated.model import AAMAnnotatedMusicLSTM
from ml.metrics import Metrics

def _decode_gap_seconds(norm,tempo_norm):
    return torch.expm1(norm.clamp(0,1)*math.log1p(16)) * 60/(tempo_norm*240).clamp(30,240)

def evaluate_checkpoint(root,checkpoint_path,batch_size=16,output=None):
    root=Path(root);checkpoint_path=Path(checkpoint_path)
    output=Path(output) if output else root/'reports/evaluation_masked.json'
    if output.exists():raise ValueError('Choose a new output report path; historical results are preserved')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True);cfg=state['config']
    mapping_path=root/'feature_mappings.json';mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    model=AAMAnnotatedMusicLSTM(mapping['vocab_sizes'],cfg['embedding_size'],cfg['hidden_size'],cfg['layers'],cfg['dropout']).to(device)
    model.load_state_dict(state['model_state']);model.eval()
    ds=AAMWindowDataset(root,'test',cfg['sequence_length'])
    masked,legacy=Metrics(),Metrics()
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=batch_size):
            cat,num,targets,timing,target_note,mask,target_tempo=[v.to(device) for v in batch]
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                out,_=model(cat,num,target_note)
            # Physical ground truth uses the TARGET event tempo; prediction has only causal input tempo.
            pg=_decode_gap_seconds(out['timing'].float(),num[:,:,4])
            tg=_decode_gap_seconds(timing,target_tempo)
            masked.add(out,targets,timing,mask,pg,tg)
            # Exact old timing metric retained solely for comparison with the historical report.
            legacy.add(out,targets,timing,torch.ones_like(mask),pg,_decode_gap_seconds(timing,num[:,:,4]))
    ids={json.loads(line)['source_id'] for line in (root/'data/test_index.jsonl').read_text().splitlines()}
    report={'protocol':'masked-v1','checkpoint':str(checkpoint_path),'best_epoch':state['epoch'],
            'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'mapping_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            'source_level_test_groups':len(ids),'test_windows':len(ds),'device':str(device),
            'real_targets':masked.report(),'legacy_including_padding':legacy.report(),
            'note':'Teacher forced, gold endpoint; real timing target decoded with target tempo. No retraining.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('experiments/aam_annotated'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();evaluate_checkpoint(a.experiment,a.checkpoint or a.experiment/'models/aam_annotated_best.pt',a.batch_size,a.output)
if __name__=='__main__':main()
