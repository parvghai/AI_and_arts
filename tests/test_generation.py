from dataclasses import replace
import unittest
from composition import Composition,NoteEvent,human_note_duration
from guided_generation import GuidedComposer,GenerationSettings,OBJECTIVES
from reference_audio import ReferenceFeatures


def reference(pc=1,density=6.,rms=.5,contour=1.):
    hist=[.001]*12;hist[pc]=.989
    return ReferenceFeatures('synthetic',10,10,120,.8,[],density,hist,'C# major',.6,rms,2.,contour,[],[])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.engine=GuidedComposer();self.c=Composition()
        self.c.events=[NoteEvent(60,0,.4)]

    def test_all_objectives_within_duration(self):
        for objective in OBJECTIVES:
            for seconds in [.5,1,2,8]:
                s=GenerationSettings(objective=objective,seconds=seconds)
                for candidate in self.engine.generate(self.c,s,reference()):
                    boundary=self.c.end_time+human_note_duration(int(s.bpm))
                    self.assertLessEqual(max(e.start+e.duration for e in candidate.events),boundary+seconds+1e-8)
                    self.assertTrue(all(e.duration>0 for e in candidate.events))

    def test_zero_reference_influence_exactly_ignores_reference(self):
        s=GenerationSettings(influence=0,seconds=8)
        self.assertEqual([c.events for c in self.engine.generate(self.c,s,None)],
                         [c.events for c in self.engine.generate(self.c,s,reference())])

    def test_harmony_controls_change_pitch_distribution(self):
        s=GenerationSettings(seconds=16,candidates=5,influence=1,features=('harmony',))
        ref=reference();weak=self.engine.generate(self.c,replace(s,influence=0),ref)
        strong=self.engine.generate(self.c,s,ref)
        score=lambda rows:sum(e.pitch%12==1 for c in rows for e in c.events if e.source=='ai')/sum(e.source=='ai' for c in rows for e in c.events)
        self.assertGreater(score(strong),score(weak)+.2)

    def test_rhythm_energy_contour_controls_are_effective(self):
        s=GenerationSettings(seconds=8,influence=1,features=('rhythm',))
        low=self.engine.generate(self.c,s,reference(density=.5))[0]
        high=self.engine.generate(self.c,s,reference(density=8))[0]
        self.assertGreater(len(high.events),len(low.events))
        s=replace(s,features=('energy',))
        low=self.engine.generate(self.c,s,reference(rms=.001))[0]
        high=self.engine.generate(self.c,s,reference(rms=.7))[0]
        self.assertGreater(sum(e.velocity for e in high.events)/len(high.events),sum(e.velocity for e in low.events)/len(low.events))
        s=replace(s,features=('contour',))
        self.assertNotEqual(self.engine.generate(self.c,s,reference(contour=-1))[0].events,
                            self.engine.generate(self.c,s,reference(contour=1))[0].events)

    def test_disabled_features_ignore_reference(self):
        s=GenerationSettings(features=(),influence=1,seconds=8)
        a=self.engine.generate(self.c,s,reference(pc=1))[0].events
        b=self.engine.generate(self.c,s,reference(pc=6,density=.5,rms=.001,contour=-1))[0].events
        self.assertEqual(a,b)

    def test_reproducibility_and_candidate_diversity(self):
        s=GenerationSettings(seconds=8,candidates=3)
        a=self.engine.generate(self.c,s);self.assertEqual(a,self.engine.generate(self.c,s))
        signatures={tuple(e.pitch for e in c.events) for c in a};self.assertGreater(len(signatures),1)

    def test_chords_preserve_polyphony_rhythm_uses_drums(self):
        chords=self.engine.generate(self.c,GenerationSettings(objective='Chords'))[0]
        ai=[e for e in chords.events if e.source=='ai']
        self.assertEqual(len([e for e in ai if e.start==ai[0].start]),3)
        drums=self.engine.generate(self.c,GenerationSettings(objective='Rhythm'))[0]
        self.assertTrue(all(e.channel==9 for e in drums.events if e.source=='ai'))

    def test_invalid_settings(self):
        for settings in [GenerationSettings(seconds=0),GenerationSettings(bpm=float('nan')),
                         GenerationSettings(influence=2),GenerationSettings(features=('fake',))]:
            with self.assertRaises(ValueError):self.engine.generate(self.c,settings)

if __name__=='__main__':unittest.main()
