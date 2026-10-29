"""Bounded frozen-model comparison and synthetic-audio guidance verification; no training."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import statistics
import tempfile
from unittest.mock import patch
import zipfile
import numpy as np
import soundfile as sf
import torch
from composition import Composition,NoteEvent,human_note_duration
from guided_generation import GenerationSettings,GuidedComposer,musical_metrics
from ml.inference import LSTMGenerator
from music_io import midi_file
from reference_audio import ReferenceAnalyzer


def average(rows,key):return statistics.mean(r[key] for r in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-source',type=Path,required=True,help='Trusted local source snapshot ZIP created before the upgrade')
    parser.add_argument('--output',type=Path,default=Path('reports/generation_upgrade_20261029'))
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a new output directory')
    # Execute only the explicitly supplied trusted project snapshot, not downloaded code.
    with zipfile.ZipFile(args.original_source) as archive:
        source=archive.read('ml/inference.py').decode('utf-8')
    original={};exec(compile(source,'original_project_inference','exec'),original)
    load=torch.load
    def restricted(*a,**kw):kw['weights_only']=True;return load(*a,**kw)
    with patch('torch.load',restricted):old=original['LSTMGenerator'](Path('models/best_model_3x512.pt'))
    new=LSTMGenerator(Path('models/best_model_3x512.pt'))
    if not old.loaded or not new.loaded:raise RuntimeError('Local baseline checkpoint required')
    prefix=Composition();prefix.events=[NoteEvent(60,0,.18,96,'human','C'),NoteEvent(62,.22,.16),NoteEvent(64,.42,.18),NoteEvent(67,.64,.18)]
    boundary=prefix.end_time+human_note_duration(100)
    rows=[];samples={}
    for seconds in (.5,1.,2.,8.):
        for seed in (11,22,33,44,55):
            torch.manual_seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):raw=old.generate_continuation(prefix,'A',seconds,100)
            with contextlib.redirect_stdout(io.StringIO()):
                candidate=GuidedComposer(new).generate(prefix,GenerationSettings(target='A',seconds=seconds,candidates=1,seed=seed))[0]
            for label,events in [('original',raw),('guided',candidate.events)]:
                metrics=musical_metrics(events,100,boundary=boundary)
                metrics['duration_violation']=metrics['span_seconds']>seconds+1e-7
                rows.append({'engine':label,'seconds':seconds,'seed':seed,**metrics})
                if seconds==8 and seed==11:samples[label]=events
    guidance=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,semitones in [('C_major',0),('Fsharp_major',6)]:
            sr=22050;t=np.arange(sr*8)/sr
            signal=sum(.08*np.sin(2*np.pi*(f*2**(semitones/12))*t) for f in (261.63,329.63,392.))
            for onset in np.arange(0,8,.5):
                idx=int(onset*sr);signal[idx:idx+200]+=.4*np.exp(-np.arange(200)/35)
            path=Path(tmp)/(name+'.wav');sf.write(path,signal,sr)
            ref=ReferenceAnalyzer().analyze(path)
            for influence in (0.,1.):
                batch=GuidedComposer(new).generate(prefix,GenerationSettings(target='A',seconds=8,candidates=5,seed=101,
                                                                            influence=influence,features=('harmony',)),ref)
                metrics=[c.metrics for c in batch]
                guidance.append({'reference':name,'influence':influence,'estimated_key':ref.key,
                                 'reference_pitch_mass':average(metrics,'mean_reference_pitch_mass'),
                                 'unique_pitches':average(metrics,'unique_pitches'),
                                 'candidate_signatures':[list(e.pitch for e in c.events if e.source=='ai') for c in batch]})
                if influence==1:samples[name]=batch[0].events
    summary={}
    for label in ('original','guided'):
        selected=[r for r in rows if r['engine']==label]
        summary[label]={'cases':len(selected),'duration_violations':sum(r['duration_violation'] for r in selected),
                        'mean_grid_error_seconds':average(selected,'mean_grid_error_seconds'),
                        'mean_adjacent_repeat_fraction':average(selected,'adjacent_repeat_fraction'),
                        'mean_unique_pitches':average(selected,'unique_pitches')}
    report={'scope':'20 seeded cases per generator; same frozen 3x512 weights, target A and prefix; synthetic reference fixtures',
            'limitations':'Output constraints and feature adherence only; not a listening study or proof of aesthetic improvement.',
            'summary':summary,'cases':rows,'reference_guidance':guidance}
    args.output.mkdir(parents=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,events in samples.items():midi_file(events,100).save(args.output/(name+'.mid'))
    print(json.dumps({'summary':summary,'reference_guidance':[{k:v for k,v in row.items() if k!='candidate_signatures'} for row in guidance]},indent=2))


if __name__=='__main__':main()
