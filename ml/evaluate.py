"""Evaluate a saved checkpoint and export target-conditioned MIDI examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from ml.dataset import EventWindowDataset, split_files
from ml.model import MusicLSTM
from ml.inference import autoregressive_generate
from ml.preprocessing import Event, decode_duration, decode_timing, decode_velocity


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--models",type=Path,default=Path("models")); p.add_argument("--checkpoint",type=Path,default=None)
    p.add_argument("--output",type=Path,default=Path("test_outputs"))
    args=p.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=args.checkpoint or (args.models/"best_model.pt")
    state=torch.load(checkpoint,map_location=device,weights_only=False); cfg=state["config"]
    if cfg.get("target_conditioning") != "future_window_final_pitch":
        raise SystemExit("Checkpoint predates future-target conditioning. Retrain with train_full.ps1 before evaluation.")
    model=MusicLSTM(embedding_size=cfg["embedding_size"],hidden_size=cfg["hidden_size"],
                    num_layers=cfg["layers"],dropout=cfg["dropout"]).to(device)
    model.load_state_dict(state["model_state"]); model.eval()
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {parameters/1_000_000:.2f} million; device={device}; CUDA={torch.cuda.is_available()}")
    files=sorted(args.data.glob("*.jsonl")); _,val,test=split_files(files,cfg.get("seed",42))
    test_ds=EventWindowDataset(test,cfg["sequence_length"])
    if not len(test_ds):
        raise SystemExit("Test split has no usable event windows.")
    ce=nn.CrossEntropyLoss()
    weights=cfg.get("loss_weights",{"pitch":1.0,"duration":1.0,"velocity":.5,"timing":.5})
    duration_lut=torch.tensor([decode_duration(i) for i in range(16)],device=device)
    velocity_lut=torch.tensor([decode_velocity(i) for i in range(8)],device=device)
    timing_lut=torch.tensor([decode_timing(i) for i in range(16)],device=device)
    totals={k:0. for k in ("loss","pitch_loss","pitch_correct","pitch_count","duration_error","velocity_error","timing_error")}; n=0
    with torch.inference_mode():
        for x,y,target,tempo in DataLoader(test_ds,batch_size=cfg.get("batch_size",32)):
            x,y=x.to(device),y.to(device); target,tempo=target.to(device),tempo.to(device)
            out,_=model(x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],target,tempo)
            count=y[:,:,0].numel()
            lp=ce(out["pitch"].reshape(-1,128),y[:,:,0].reshape(-1))
            ld=ce(out["duration"].reshape(-1,16),y[:,:,1].reshape(-1))
            lv=ce(out["velocity"].reshape(-1,8),y[:,:,2].reshape(-1))
            lt=nn.functional.smooth_l1_loss(out["timing"],y[:,:,3].float()/7.5)
            totals["loss"]+=float((weights["pitch"]*lp+weights["duration"]*ld+weights["velocity"]*lv+weights["timing"]*lt))*count
            totals["pitch_loss"]+=float(lp)*count
            pp=out["pitch"].argmax(-1); dd=out["duration"].argmax(-1); vv=out["velocity"].argmax(-1)
            totals["pitch_correct"]+=int((pp==y[:,:,0]).sum()); totals["pitch_count"]+=count
            totals["duration_error"]+=float((duration_lut[dd]-duration_lut[y[:,:,1]]).abs().sum())
            totals["velocity_error"]+=float((velocity_lut[vv]-velocity_lut[y[:,:,2]]).abs().sum())
            predicted_timing=timing_lut[(out["timing"]*7.5).round().long().clamp(0,15)]
            totals["timing_error"]+=float((predicted_timing-timing_lut[y[:,:,3]]).abs().sum()); n+=count
    print(f"validation_loss={state['validation_loss']:.4f} test_loss={totals['loss']/max(1,n):.4f} test_pitch_loss={totals['pitch_loss']/max(1,n):.4f}")
    print(f"pitch_accuracy={totals['pitch_correct']/max(1,n):.3f}; duration_MAE_seconds={totals['duration_error']/max(1,n):.3f}; velocity_MAE={totals['velocity_error']/max(1,n):.1f}; timing_MAE_seconds={totals['timing_error']/max(1,n):.3f}")
    args.output.mkdir(parents=True,exist_ok=True)
    report={"checkpoint":str(checkpoint),"device":str(device),"parameter_count":parameters,
            "validation_loss":float(state["validation_loss"]),"test_loss":totals["loss"]/n,
            "test_pitch_loss":totals["pitch_loss"]/n,"pitch_accuracy":totals["pitch_correct"]/n,
            "duration_mae_seconds":totals["duration_error"]/n,"velocity_mae":totals["velocity_error"]/n,
            "timing_mae_seconds":totals["timing_error"]/n,"test_events":int(n),
            "validation_files":len(val),"test_files":len(test),
            "model":{"layers":cfg["layers"],"hidden_size":cfg["hidden_size"],
                     "embedding_size":cfg["embedding_size"],"sequence_length":cfg["sequence_length"]}}
    (args.output/"evaluation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    try: import mido
    except ImportError as exc: raise SystemExit("MIDI export requires mido (pip install mido)") from exc
    targets={"C":60,"E":64,"G":67,"A":69}
    for name,pitch in targets.items():
        ctx=[Event(60,.25,80,0),Event(64,.25,80,.25),Event(67,.25,80,.25)]
        events=autoregressive_generate(model,ctx,pitch,1.0,100,cfg.get("temperature",.8),device,cfg["sequence_length"])
        mid=mido.MidiFile(); track=mido.MidiTrack(); mid.tracks.append(track)
        us_per_beat=600_000  # 100 BPM, matching the generation conditioning.
        track.append(mido.MetaMessage("set_tempo",tempo=us_per_beat,time=0))
        track.append(mido.Message("program_change",program=0,time=0))
        ticks_per_second=mid.ticks_per_beat*1_000_000/us_per_beat
        scheduled=[]; onset_seconds=0.0
        for e in events:
            onset_seconds+=e.delta_time
            start=round(onset_seconds*ticks_per_second); end=start+max(1,round(e.duration*ticks_per_second))
            scheduled.extend(((start,1,mido.Message("note_on",note=e.pitch,velocity=e.velocity)),
                              (end,0,mido.Message("note_off",note=e.pitch,velocity=0))))
        previous_tick=0
        for tick,_,message in sorted(scheduled,key=lambda row:(row[0],row[1])):
            track.append(message.copy(time=max(0,tick-previous_tick))); previous_tick=tick
        mid.save(args.output/f"example_{name}.mid")
    print(f"Generated MIDI examples and evaluation_report.json in {args.output}")


if __name__=="__main__": main()
