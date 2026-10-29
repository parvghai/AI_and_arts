"""Fine-tune frozen MIDI MusicLSTM checkpoints with low-rank adapters."""
from __future__ import annotations
import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ml.dataset import EventWindowDataset, split_files
from ml.losses import event_loss
from ml.lora import attach_lora, adapter_parameters, load_base, save_adapter, sha256


def measure(model, loader, device, max_batches=0):
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y, target, tempo, mask = [t.to(device) for t in batch]
            out, _ = model(x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3], target, tempo)
            loss, _ = event_loss(out, y[:,:,:3], y[:,:,3].float()/7.5, mask)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite validation loss')
            n = int(mask.sum()); count += n; loss_sum += float(loss) * n
            correct += int(((out['pitch'].argmax(-1) == y[:,:,0]) & mask).sum())
    if not count:
        raise ValueError('No valid validation targets')
    return {'loss': loss_sum/count, 'pitch_accuracy': correct/count, 'real_targets': count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('models/best_model_3x512.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--smoke', action='store_true', help='Only 3 updates on at most 12 training files and 2 validation batches')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('epochs, batch-size and learning-rate must be positive')
    output = args.output or Path('runs/lora') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if output.exists():
        parser.error('Choose a new output directory; existing runs are never overwritten')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(2)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, config = load_base(args.base, device)
    base_hash = sha256(args.base)
    # Preserve the base model's split, even if the fine-tuning sampling seed changes.
    split_seed = int(config.get('seed', 42))
    parts = split_files(sorted(args.data.glob('*.jsonl')), split_seed)
    if not all(parts):
        parser.error('Need at least three composition groups with train/validation/test files')
    train_files, val_files, test_files = parts
    if args.smoke:
        train_files = random.Random(args.seed).sample(train_files, min(12, len(train_files)))
        val_files = val_files[:12]
    length = int(config.get('sequence_length',128))
    train_ds = EventWindowDataset(train_files, length)
    val_ds = EventWindowDataset(val_files, length)
    if not len(train_ds) or not len(val_ds):
        parser.error('No usable train/validation windows')
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(val_ds, batch_size=args.batch_size)
    base_parameters = sum(p.numel() for p in model.parameters())
    targets = attach_lora(model, args.rank, args.alpha)
    # Zero-initialized adapters equal the base, evaluated with the same backend
    # as subsequent validation to avoid cuDNN/native numerical differences.
    baseline = measure(model, val, device, 2 if args.smoke else 0)
    params = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    frozen = {name: p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    output.mkdir(parents=True)
    manifest = {name: [{'file':str(p.relative_to(args.data)), 'sha256':sha256(p)} for p in paths]
                for name,paths in zip(('train','validation','reserved_test'), (train_files,val_files,test_files))}
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    history = []; best_loss = float('inf'); best_file = None; steps = 0
    print(f'Base parameters: {base_parameters:,}; trainable LoRA: {sum(p.numel() for p in params):,}; device: {device}',flush=True)
    for epoch in range(1, (1 if args.smoke else args.epochs)+1):
        model.train(); total = seen = 0
        for batch in train:
            x,y,target,tempo,mask = [t.to(device) for t in batch]
            optimizer.zero_grad(set_to_none=True)
            out,_ = model(x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],target,tempo)
            loss,_ = event_loss(out,y[:,:,:3],y[:,:,3].float()/7.5,mask)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            optimizer.step(); steps += 1
            count = int(mask.sum()); total += float(loss.detach())*count; seen += count
            if args.smoke and steps >= 3:
                break
        validation = measure(model,val,device,2 if args.smoke else 0)
        row = {'epoch':epoch,'train_loss':total/seen,'validation':validation,'steps':steps}
        history.append(row); print(json.dumps(row),flush=True)
        if validation['loss'] < best_loss:
            best_loss = validation['loss']; best_file = f'adapter_epoch_{epoch:03d}.pt'
            save_adapter(output/best_file,model,args.base,args.rank,args.alpha,
                         {'epoch':epoch,'validation':validation,'smoke_only':args.smoke,'split_seed':split_seed})
    frozen_unchanged = all(torch.equal(dict(model.named_parameters())[k],v) for k,v in frozen.items())
    if not frozen_unchanged or sha256(args.base) != base_hash:
        raise RuntimeError('Base weight preservation check failed')
    report = {'mode':'smoke' if args.smoke else 'fine-tune','base_checkpoint':str(args.base),
              'base_sha256':base_hash,'base_parameters':base_parameters,
              'trainable_parameters':sum(p.numel() for p in params),'rank':args.rank,'alpha':args.alpha,
              'targets':targets,'device':str(device),'seed':args.seed,'split_seed':split_seed,
              'sequence_length':length,'learning_rate':args.learning_rate,'batch_size':args.batch_size,
              'baseline_validation':baseline,'history':history,'best_adapter':best_file,
              'validation_loss_improved':best_loss < baseline['loss'],
              'base_weights_unchanged':frozen_unchanged,'test_used_for_training':False,
              'quality_claim':'Mechanics check only' if args.smoke else 'Validation metrics; listening assessment still needed'}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Adapter and report saved under {output}. No adapter has been activated automatically.')
    return report


if __name__ == '__main__':
    main()
