from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import soundfile as sf
from reference_audio import ReferenceAnalyzer
from guided_generation import GuidedComposer,GenerationSettings
from composition import Composition


def fixture(path,duration=8):
    sr=22050;t=np.arange(int(sr*duration))/sr
    audio=sum(.06*np.sin(2*np.pi*f*t) for f in [261.63,329.63,392.0])
    for start in np.arange(0,duration,.5):
        index=int(start*sr);length=min(500,len(audio)-index)
        audio[index:index+length]+=.5*np.exp(-np.arange(length)/80)*np.cos(np.arange(length)*2)
    sf.write(path,audio,sr)


class ReferenceTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_wav_analysis_and_cache(self):
        p=self.root/'chord.wav';fixture(p)
        analyzer=ReferenceAnalyzer();ref=analyzer.analyze(p)
        self.assertAlmostEqual(ref.duration,8,places=2);self.assertEqual(len(ref.pitch_classes),12)
        self.assertAlmostEqual(sum(ref.pitch_classes),1,places=5)
        self.assertGreater(sum(ref.pitch_classes[i] for i in [0,4,7]),sum(ref.pitch_classes[i] for i in [1,6,10]))
        self.assertGreater(ref.onset_density,0);self.assertIsNotNone(ref.tempo)
        self.assertIs(analyzer.analyze(p),ref)

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg unavailable; MP3 encoding fixture skipped')
    def test_real_mp3_decode_and_guidance(self):
        wav=self.root/'reference.wav';mp3=self.root/'reference.mp3';fixture(wav)
        subprocess.run([shutil.which('ffmpeg'),'-y','-loglevel','error','-i',str(wav),str(mp3)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ref=ReferenceAnalyzer().analyze(mp3);self.assertGreater(ref.duration,7)
        engine=GuidedComposer();s=GenerationSettings(seconds=8,influence=0)
        before=engine.generate(Composition(),s,ref)[0].events
        s.influence=1;after=engine.generate(Composition(),s,ref)[0].events
        self.assertNotEqual(before,after)

    def test_silence_corrupt_unsupported_and_size(self):
        p=self.root/'silence.wav';sf.write(p,np.zeros(22050),22050)
        with self.assertRaisesRegex(ValueError,'silent'):ReferenceAnalyzer().analyze(p)
        p=self.root/'broken.mp3';p.write_bytes(b'not mp3')
        with self.assertRaises(ValueError):ReferenceAnalyzer().analyze(p)
        p=self.root/'unsupported.txt';p.write_text('x')
        with self.assertRaisesRegex(ValueError,'Choose'):ReferenceAnalyzer().analyze(p)
        p=self.root/'small.wav';fixture(p);a=ReferenceAnalyzer();a.MAX_BYTES=10
        with self.assertRaisesRegex(ValueError,'256 MiB'):a.analyze(p)

    def test_long_input_is_bounded(self):
        p=self.root/'long.wav';fixture(p,duration=4);a=ReferenceAnalyzer();a.ANALYSIS_SECONDS=2
        ref=a.analyze(p);self.assertLessEqual(ref.analyzed_seconds,2.01)
        self.assertAlmostEqual(ref.duration,4,places=2)

if __name__=='__main__':unittest.main()
