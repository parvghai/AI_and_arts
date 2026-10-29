"""Bounded local audio analysis. Estimates guide new notes, never transcribe a recording."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math

PITCH_NAMES=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')


@dataclass
class ReferenceFeatures:
    filename: str
    duration: float
    analyzed_seconds: float
    tempo: float | None
    tempo_confidence: float
    beat_positions: list[float]
    onset_density: float
    pitch_classes: list[float]
    key: str | None
    key_confidence: float
    rms: float
    dynamic_range: float
    contour: float
    chords: list[dict]
    warnings: list[str]
    kind: str = 'audio'

    def to_dict(self):return asdict(self)


class ReferenceAnalyzer:
    MAX_BYTES=256*1024*1024
    MAX_DURATION=1200
    ANALYSIS_SECONDS=90

    def __init__(self):self._cache={}

    def analyze(self,path,progress=lambda message:None):
        path=Path(path).resolve()
        if path.suffix.lower() not in ('.mp3','.wav','.mid','.midi'):
            raise ValueError('Choose MP3, WAV, or MIDI')
        if not path.is_file():raise ValueError('Reference file not found')
        stat=path.stat()
        if not 0 < stat.st_size <= self.MAX_BYTES:raise ValueError('Reference must be nonempty and at most 256 MiB')
        key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key in self._cache:return self._cache[key]
        progress('Reading reference locally…')
        if path.suffix.lower() in ('.mid','.midi'):
            result=self._midi(path)
        else:
            result=self._audio(path,progress)
        self._cache[key]=result
        if len(self._cache)>4:self._cache.pop(next(iter(self._cache)))
        return result

    def _audio(self,path,progress):
        try:
            import numpy as np
            import soundfile as sf
            from scipy import signal
        except ImportError as exc:
            raise ValueError('Audio analysis needs numpy, scipy and soundfile; see requirements-reference.txt') from exc
        import shutil
        if path.suffix.lower()=='.mp3' and shutil.which('ffmpeg') and shutil.which('ffprobe'):
            import subprocess,json,os
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            try:
                probe=subprocess.run([shutil.which('ffprobe'),'-v','error','-protocol_whitelist','file,pipe','-select_streams','a:0',
                                      '-show_entries','format=duration:stream=sample_rate,channels','-of','json',str(path)],
                                     capture_output=True,text=True,timeout=15,check=True,creationflags=flags)
                info=json.loads(probe.stdout);duration=float(info['format']['duration'])
                if not .25<=duration<=self.MAX_DURATION:raise ValueError('Reference must be between 0.25 seconds and 20 minutes')
                if int(info['streams'][0]['channels'])>8:raise ValueError('At most 8 channels are supported')
                decoded=subprocess.run([shutil.which('ffmpeg'),'-v','error','-nostdin','-protocol_whitelist','file,pipe','-i',str(path),
                                        '-t',str(self.ANALYSIS_SECONDS),'-ac','1','-ar','11025','-f','f32le','pipe:1'],
                                       capture_output=True,timeout=45,check=True,creationflags=flags)
                sr=11025;audio=np.frombuffer(decoded.stdout,dtype='<f4')
                if not len(audio) or not np.isfinite(audio).all():raise ValueError('MP3 decoder returned no valid samples')
            except (subprocess.SubprocessError,KeyError,IndexError) as exc:
                raise ValueError('Cannot decode MP3; file may be corrupt or unsupported') from exc
        else:
            try:
                with sf.SoundFile(path) as stream:
                    duration=len(stream)/stream.samplerate
                    if not .25 <= duration <= self.MAX_DURATION:
                        raise ValueError('Reference must be between 0.25 seconds and 20 minutes')
                    if not 8000 <= stream.samplerate <= 192000 or stream.channels>8:
                        raise ValueError('Supported audio: 8–192 kHz, at most 8 channels')
                    sr=11025;source_rate=stream.samplerate;g=math.gcd(sr,source_rate)
                    left=min(len(stream),round(self.ANALYSIS_SECONDS*source_rate));chunks=[]
                    while left:
                        block=stream.read(min(left,source_rate),dtype='float32',always_2d=True)
                        if not len(block):break
                        left-=len(block)
                        if not np.isfinite(block).all():raise ValueError('Audio contains invalid sample values')
                        chunks.append(signal.resample_poly(block.mean(axis=1),sr//g,source_rate//g))
                    audio=np.concatenate(chunks)
            except (RuntimeError,OSError) as exc:
                raise ValueError(f'Cannot decode audio. The file may be corrupt or this SoundFile build may lack MP3 support: {exc}') from exc
        rms=float(np.sqrt(np.mean(audio**2)))
        if rms<1e-5:raise ValueError('Reference is silent or too quiet to analyze')
        progress('Estimating rhythm, pitch classes and dynamics…')
        _,times,z=signal.stft(audio,fs=sr,nperseg=2048,noverlap=1536,boundary=None)
        mag=np.abs(z);hop=512/sr
        frame_energy=np.sqrt(np.mean(mag**2,axis=0))
        flux=np.maximum(np.diff(mag,axis=1),0).sum(axis=0)
        if not len(flux):raise ValueError('Reference is too short for analysis')
        peaks,_=signal.find_peaks(flux,height=float(np.median(flux)+.5*np.std(flux)),distance=max(1,int(.09/hop)))
        onsets=(peaks+1)*hop
        tempo,confidence=self._tempo(flux,hop)
        beats=[]
        if tempo:
            period=60/tempo
            origin=float(onsets[0]) if len(onsets) else 0.
            beats=np.arange(origin,len(audio)/sr,period).round(4).tolist()
        freq=np.fft.rfftfreq(2048,1/sr)
        useful=(freq>=65)&(freq<=2200)
        midi=np.rint(69+12*np.log2(freq[useful]/440)).astype(int)
        chroma=np.zeros((12,mag.shape[1]))
        for pc in range(12):chroma[pc]=np.sqrt(mag[useful][midi%12==pc]).sum(axis=0)
        histogram=chroma.sum(axis=1);histogram/=max(float(histogram.sum()),1e-12)
        key,key_conf=self._key(histogram)
        dominant=midi[np.argmax(mag[useful],axis=0)]
        # Broad direction only; no source melody or event sequence is retained.
        contour=float(np.clip((np.median(dominant[-max(1,len(dominant)//4):])-np.median(dominant[:max(1,len(dominant)//4)]))/12,-1,1))
        chords=[]
        for start in np.arange(0,min(len(audio)/sr,64),2):
            selected=(times>=start)&(times<start+2)
            if not selected.any():continue
            c=chroma[:,selected].sum(axis=1);c/=max(float(c.sum()),1e-12)
            scores=sorted([(float(c[[root,(root+third)%12,(root+7)%12]].sum()),root,quality)
                           for root in range(12) for third,quality in [(4,'major'),(3,'minor')]],reverse=True)
            margin=scores[0][0]-scores[1][0]
            if scores[0][0]>.38 and margin>.015:
                chords.append({'start':float(start),'root':scores[0][1],'quality':scores[0][2],
                               'confidence':round(min(1,margin*8),3)})
        warnings=['Tempo, key and chords are estimates; dense mixes, drums and tuning can mislead them.',
                  'Melodic direction is a spectral proxy, not a melody transcription. Instrument/style embeddings are not inferred.']
        if duration>self.ANALYSIS_SECONDS:warnings.append(f'Only the first {self.ANALYSIS_SECONDS} seconds were analyzed; no converted file was saved.')
        if not tempo:warnings.append('No reliable periodic pulse found; choose tempo manually.')
        if key_conf<.15:warnings.append('Key confidence is low; pitch-class guidance is safer than a fixed key.')
        return ReferenceFeatures(path.name,duration,len(audio)/sr,tempo,confidence,beats,
                                 len(onsets)/(len(audio)/sr),histogram.tolist(),key,key_conf,rms,
                                 float(np.percentile(frame_energy,90)/max(np.percentile(frame_energy,10),1e-9)),
                                 contour,chords,warnings)

    @staticmethod
    def _tempo(flux,hop):
        import numpy as np
        from scipy import signal
        centered=flux-np.mean(flux)
        correlation=signal.correlate(centered,centered,mode='full',method='fft')[len(centered)-1:]
        lo=max(1,round(60/180/hop));hi=min(len(correlation)-1,round(60/60/hop))
        if hi<=lo or correlation[0]<=1e-12:return None,0.
        lags,_=signal.find_peaks(correlation[lo:hi+1])
        if not len(lags):return None,0.
        lag=max((int(i+lo) for i in lags),key=lambda i:correlation[i])
        conf=float(np.clip(correlation[lag]/correlation[0],0,1))
        return (round(60/(lag*hop),1) if conf>.12 else None),conf

    @staticmethod
    def _key(histogram):
        import numpy as np
        h=np.asarray(histogram)
        if float(h.std())<1e-8:return None,0.
        templates={'major':[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
                   'minor':[6.33,2.68,3.52,5.38,2.6,3.53,2.54,4.75,3.98,2.69,3.34,3.17]}
        scores=sorted([(float(np.corrcoef(h,np.roll(profile,root))[0,1]),root,mode)
                       for mode,profile in templates.items() for root in range(12)],reverse=True)
        score,root,mode=scores[0];confidence=max(0.,min(1.,(score-scores[1][0])*3))
        return f'{PITCH_NAMES[root]} {mode}',confidence

    def _midi(self,path):
        import numpy as np
        import mido
        from music_io import read_midi
        events=read_midi(path);duration=max(e.start+e.duration for e in events)
        if duration>self.MAX_DURATION:raise ValueError('MIDI reference exceeds 20 minutes')
        hist=np.zeros(12)
        for e in events:
            if e.channel!=9:hist[e.pitch%12]+=e.duration*e.velocity
        hist/=max(float(hist.sum()),1.)
        key,conf=self._key(hist)
        tempos=[60_000_000/m.tempo for t in mido.MidiFile(path).tracks for m in t if m.type=='set_tempo']
        tempo=float(np.median(tempos)) if tempos else 120.
        return ReferenceFeatures(path.name,duration,duration,tempo,1. if tempos else 0.,
                                 np.arange(0,duration,60/tempo).tolist(),len({e.start for e in events})/max(duration,.01),
                                 hist.tolist(),key,conf,float(np.mean([e.velocity for e in events]))/127,
                                 max(e.velocity for e in events)/max(1,min(e.velocity for e in events)),
                                 float(np.clip((events[-1].pitch-events[0].pitch)/12,-1,1)),[],
                                 ['MIDI pitch/velocity are symbolic measurements; key is estimated.',
                                  'Tempo is the median declared MIDI tempo (120 BPM default if absent).'],kind='midi')


class AudioPreview:
    """Preview the original file through a local player; no conversion or upload."""
    def __init__(self):self.process=None

    def play(self,path):
        import subprocess,shutil,os
        self.stop()
        player=shutil.which('ffplay')
        if not player:raise ValueError('Reference preview needs ffplay on PATH. Analysis/generation do not require it.')
        self.process=subprocess.Popen([player,'-nodisp','-autoexit','-loglevel','error','-protocol_whitelist','file,pipe','-i',str(Path(path).resolve())],
                                      stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=.5)
            except __import__('subprocess').TimeoutExpired:self.process.kill()
        self.process=None
