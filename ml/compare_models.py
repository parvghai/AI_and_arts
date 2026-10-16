"""Generate matched human-conditioned MIDI comparisons for 3x512 and 4x768."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from ml.inference import LSTMGenerator
from composition import Composition

NOTES = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71}

def save_midi(path: Path, composition: Composition, bpm: int = 100) -> None:
    import mido
    midi=mido.MidiFile(); track=mido.MidiTrack(); midi.tracks.append(track)
    tempo=round(60_000_000/bpm); track.append(mido.MetaMessage("set_tempo",tempo=tempo,time=0))
    per_second=midi.ticks_per_beat*1_000_000/tempo
    scheduled=[]
    for e in composition.events:
        start=max(0,round(e.start*per_second)); end=start+max(1,round(e.duration*per_second))
        scheduled.extend(((start,1,mido.Message("note_on",note=e.pitch,velocity=max(1,min(127,e.velocity)))),
                          (end,0,mido.Message("note_off",note=e.pitch,velocity=0))))
    prev=0
    for tick,_,msg in sorted(scheduled,key=lambda r:(r[0],r[1])):
        track.append(msg.copy(time=max(0,tick-prev))); prev=tick
    midi.save(path)

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",type=Path,default=Path("models")); p.add_argument("--output",type=Path,default=Path("evaluation"))
    p.add_argument("--sequence",default="C,E,G,A"); p.add_argument("--durations",default="0.5,1,2")
    p.add_argument("--bpm",type=int,default=100); p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--old-report",type=Path); p.add_argument("--new-report",type=Path)
    a=p.parse_args(); names=[x.strip().upper() for x in a.sequence.split(",")]
    if not names or any(x not in NOTES for x in names): raise SystemExit("Use comma-separated natural notes, e.g. C,E,G,A")
    durations=[float(x) for x in a.durations.split(",")]
    if any(x not in (.5,1.,2.) for x in durations): raise SystemExit("Durations must be selected from 0.5, 1, or 2 seconds")
    reports={}
    for label,filename in (("old_lstm","best_model_3x512.pt"),("large_lstm","best_model_4x768.pt")):
        generator=LSTMGenerator(a.models/filename)
        if not generator.loaded: raise SystemExit(f"Cannot compare: {generator.load_error}")
        count=sum(p.numel() for p in generator.model.parameters() if p.requires_grad)
        model_dir=a.output/label; model_dir.mkdir(parents=True,exist_ok=True)
        results={"checkpoint":str(generator.checkpoint),"parameters":count,"device":str(generator.device),"durations":{}}
        for duration in durations:
            torch.manual_seed(a.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
            composition=Composition(); latencies=[]; interaction_records=[]
            for note in names:
                start=time.perf_counter()
                generated=generator.generate_continuation(composition,note,duration,a.bpm)
                latency=time.perf_counter()-start; latencies.append(latency)
                interaction_records.append({"human_target":note,"continuation_seconds":duration,
                    "bpm":a.bpm,"temperature":generator.temperature,"top_k":generator.top_k,
                    "top_p":generator.top_p,"generation_latency_seconds":latency,
                    "ai_pitches":[event.pitch for event in generated]})
                composition.add_interaction(note,generated,duration,a.bpm)
            pitches=[e.pitch for e in composition.events if e.source=="ai"]
            run=max((len(list(g)) for _,g in __import__('itertools').groupby(pitches)),default=0)
            target_completion={note:any(e.pitch==NOTES[note] for e in composition.interactions[i].events if e.source=="ai")
                              for i,note in enumerate(names)}
            save_midi(model_dir/f"sequence_{duration:g}s.mid",composition,a.bpm)
            results["durations"][str(duration)]={"mean_latency_seconds":sum(latencies)/max(1,len(latencies)),
                "p95_latency_seconds":(sorted(latencies)[min(len(latencies)-1,max(0,round(.95*(len(latencies)-1))))] if latencies else 0),
                "ai_event_count":len(pitches),"pitch_diversity":len(set(pitches)),
                "repeated_pitch_fraction":sum(x==y for x,y in zip(pitches,pitches[1:]))/max(1,len(pitches)-1),
                "max_identical_pitch_run":run,"target_pitch_present_by_input":target_completion,
                "duration_mean_seconds":sum(e.duration for e in composition.events if e.source=="ai")/max(1,len(pitches)),
                "human_ai_interactions":interaction_records}
        reports[label]=results
    for key,path in (("old_lstm",a.old_report),("large_lstm",a.new_report)):
        if path and path.exists(): reports[key]["loss_report"]=json.loads(path.read_text(encoding="utf-8"))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"comparison_report.json").write_text(json.dumps({"human_sequence":names,"bpm":a.bpm,"results":reports},indent=2),encoding="utf-8")
    print(json.dumps({key:{"parameters":value["parameters"],"generation":value["durations"]} for key,value in reports.items()},indent=2))
    print(f"Comparison report and MIDI files saved in {a.output}")

if __name__=="__main__": main()
