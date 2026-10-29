"""Hybrid composition: existing LSTM proposals plus explicit, testable musical guidance.

No audio-trained model is implied. Accompaniment objectives are symbolic generators.
Only aggregate reference characteristics are used; source melodies are never replayed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import contextlib
import io
import math
import random
from composition import Composition, NoteEvent, human_note_duration
from music_io import validate_events

OBJECTIVES=('Continue melody','New melody','Chords','Bassline','Rhythm','Complementary part')
NAMES={'C':60,'D':62,'E':64,'F':65,'G':67,'A':69,'B':71}


@dataclass
class GenerationSettings:
    objective: str='Continue melody'
    target: str='C'
    bpm: float=100.
    seconds: float=2.
    influence: float=.6
    features: tuple[str,...]=('harmony','rhythm','energy','contour')
    temperature: float=.75
    seed: int=42
    candidates: int=3

    def validate(self):
        if self.objective not in OBJECTIVES or self.target not in NAMES:raise ValueError('Unknown objective or cue')
        if not all(math.isfinite(v) for v in (self.bpm,self.seconds,self.influence,self.temperature)):
            raise ValueError('Controls must be finite')
        if not 30<=self.bpm<=240 or not .25<=self.seconds<=30:raise ValueError('Use 30–240 BPM and 0.25–30 seconds')
        if not 0<=self.influence<=1 or not .05<=self.temperature<=2 or not 1<=self.candidates<=5:
            raise ValueError('Invalid influence, temperature or candidate count')
        if set(self.features)-{'harmony','rhythm','energy','contour'}:raise ValueError('Unknown reference feature')


@dataclass
class Candidate:
    events: list[NoteEvent]
    seed: int
    engine: str
    metrics: dict=field(default_factory=dict)


def musical_metrics(events,bpm,reference=None,boundary=0.):
    notes=[e for e in events if e.source=='ai']
    pitches=[e.pitch for e in notes];grid=60/bpm/4
    return {'notes':len(notes),'unique_pitches':len(set(pitches)),
            'adjacent_repeat_fraction':sum(a==b for a,b in zip(pitches,pitches[1:]))/max(1,len(pitches)-1),
            'mean_grid_error_seconds':sum(abs((e.start-boundary)/grid-round((e.start-boundary)/grid))*grid for e in notes)/max(1,len(notes)),
            'mean_reference_pitch_mass':(sum(reference.pitch_classes[p%12] for p in pitches)/max(1,len(pitches)) if reference else None),
            'span_seconds':max((e.start+e.duration-boundary for e in notes),default=0.)}


class GuidedComposer:
    def __init__(self,learned=None):self.learned=learned

    def generate(self,composition:Composition,settings:GenerationSettings,reference=None,start=None,include_cue=True):
        settings.validate()
        if reference and (len(reference.pitch_classes)!=12 or
                          any(not math.isfinite(v) or v<0 for v in reference.pitch_classes)):
            raise ValueError('Invalid reference pitch distribution')
        origin=composition.end_time if start is None else float(start)
        if origin<0 or not math.isfinite(origin):raise ValueError('Invalid phrase start')
        boundary=origin+(human_note_duration(int(settings.bpm)) if include_cue else 0.)
        result=[]
        for index in range(settings.candidates):
            seed=settings.seed+index*1009;rng=random.Random(seed);hints=[];engine='Symbolic composer'
            if self.learned and self.learned.loaded and settings.objective in ('Continue melody','New melody','Complementary part'):
                try:
                    import torch
                    devices=[self.learned.device.index or 0] if self.learned.device.type=='cuda' else []
                    with torch.random.fork_rng(devices=devices),contextlib.redirect_stdout(io.StringIO()):
                        torch.manual_seed(seed)
                        self.learned.temperature=settings.temperature
                        prefix=composition if settings.objective!='New melody' else Composition()
                        hints=self.learned.generate_continuation(prefix,settings.target,settings.seconds,int(settings.bpm))
                    engine='LSTM + musical constraints'
                except Exception as exc:
                    engine=f'Symbolic fallback ({type(exc).__name__})'
            notes=self._phrase(composition,settings,reference,boundary,hints,rng)
            if include_cue:notes.insert(0,NoteEvent(NAMES[settings.target],origin,boundary-origin,95,'human',settings.target))
            validate_events(notes)
            result.append(Candidate(notes,seed,engine,musical_metrics(notes,settings.bpm,reference,boundary)))
        return result

    @staticmethod
    def _phrase(composition,s,ref,boundary,hints,rng):
        beat=60/s.bpm;grid=beat/4;influence=s.influence if ref else 0.
        target=NAMES[s.target]
        previous=next((e.pitch for e in reversed(composition.events) if e.channel!=9),target)
        if s.objective=='New melody':previous=target
        low,high=(36,55) if s.objective=='Bassline' else ((72,88) if s.objective=='Complementary part' else (48,84))
        previous=max(low,min(high,previous));target=max(low,min(high,target))
        hist=ref.pitch_classes if ref else [1/12]*12
        follow_harmony=ref is not None and 'harmony' in s.features
        # A cue is an endpoint, not a request to change key. Infer a loose scale
        # from existing material; use C major only when no tonal context exists.
        context=[e.pitch%12 for e in composition.events[-48:] if e.channel!=9]
        root,minor=0,False
        if context:
            options=[]
            for candidate_root in range(12):
                for is_minor,intervals in ((False,(0,2,4,5,7,9,11)),(True,(0,2,3,5,7,8,10))):
                    pcs={(candidate_root+i)%12 for i in intervals}
                    fit=sum(pc in pcs for pc in context)+.05*context.count(candidate_root)
                    options.append((fit,-candidate_root,not is_minor,candidate_root,is_minor))
            _,_,_,root,minor=max(options)
        if follow_harmony and influence>=.35:
            root=max(range(12),key=lambda pc:hist[pc])
            minor=bool(ref.key and 'minor' in ref.key)
        scale={(root+i)%12 for i in ((0,2,3,5,7,8,10) if minor else (0,2,4,5,7,9,11))}
        density=2/beat
        if ref and 'rhythm' in s.features:density=(1-influence)*density+influence*max(.5,min(8,ref.onset_density))
        base_step=max(1,min(8,round(1/max(.2,density)/grid)))
        if s.objective=='Chords':base_step=max(4,round(2*beat/grid))
        elif s.objective=='Bassline':base_step=max(2,base_step)
        energy=82
        if ref and 'energy' in s.features:
            ref_energy=(40+80*min(1,ref.rms)) if ref.kind=='midi' else max(35,min(115,100+20*math.log10(max(ref.rms,1e-5))))
            energy=round((1-influence)*82+influence*ref_energy)
        events=[];onset=0.;index=0;recent=[];motif=[]
        while onset<s.seconds-1e-7 and index<256:
            remaining=s.seconds-onset
            step=base_step
            if index%4==3:step=min(16,base_step*2)  # phrase breathing space
            elif rng.random()<.25:step=max(1,base_step+rng.choice([-1,1]))
            span=min(remaining,step*grid)
            length=max(.001,min(remaining,span*(.9 if s.objective!='Chords' else .98)))
            velocity=max(1,min(127,energy+rng.randrange(-8,9)+(5 if index%4==0 else 0)))
            if s.objective=='Rhythm':
                pitch=rng.choices([36,38,42],[3 if index%4==0 else 1,2 if index%4==2 else 1,3])[0]
                events.append(NoteEvent(pitch,boundary+onset,min(length,.15),velocity,'ai',str(pitch),9,0))
            elif s.objective=='Chords':
                degrees=(0,5,7,3 if minor else 4)
                chord_root=(root+degrees[(index+rng.randrange(0,2))%4])%12
                base=48+chord_root
                third=3 if ((chord_root+3)%12 in scale and (chord_root+4)%12 not in scale) else 4
                for pitch in [base,base+third,base+7]:
                    events.append(NoteEvent(pitch,boundary+onset,length,velocity,'ai',str(pitch),1,0))
            else:
                choices=list(range(low,high+1));weights=[]
                hint=hints[index%len(hints)].pitch if hints else previous
                for pitch in choices:
                    distance=abs(pitch-previous)
                    score=math.exp(-distance/(3+3*s.temperature))
                    score*=2.5 if pitch%12 in scale else .35
                    score*=1+.8*math.exp(-abs(pitch-hint)/3)
                    if pitch in recent[-3:]:score*=.35
                    if follow_harmony:score*=((1-influence)+influence*(.08+12*hist[pitch%12]))
                    if ref and 'contour' in s.features:
                        score*=math.exp(influence*ref.contour*(pitch-previous)/6)
                    if s.objective=='Bassline':score*=3 if pitch%12 in (root,(root+7)%12) else .4
                    if s.objective=='Complementary part':
                        sounding={e.pitch%12 for e in composition.events if e.start<=boundary+onset<e.start+e.duration}
                        if pitch%12 in sounding:score*=.25
                    weights.append(score)
                pitch=rng.choices(choices,weights)[0]
                # Reuse a short generated motif occasionally, never the reference melody.
                if index>=4 and motif and index%4==0 and rng.random()<.35:pitch=motif[0]
                if remaining<=step*grid+1e-7:
                    resolution=[p for p in choices if p%12 in (root,NAMES[s.target]%12)]
                    pitch=min(resolution,key=lambda p:abs(p-previous))
                channel=2 if s.objective=='Bassline' else (3 if s.objective=='Complementary part' else 0)
                program=32 if s.objective=='Bassline' else 0
                events.append(NoteEvent(pitch,boundary+onset,length,velocity,'ai',str(pitch),channel,program))
                previous=pitch;recent.append(pitch)
                if index<3:motif.append(pitch)
            onset+=step*grid;index+=1
        return sorted(events,key=lambda e:(e.start,e.pitch))
