import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ml.dataset import EventWindowDataset,split_files
from ml.aam_annotated.dataset import AAMWindowDataset
from ml.losses import event_loss
from ml.inference import LSTMGenerator
from composition import Composition,human_note_duration


class MLTests(unittest.TestCase):
    def test_padding_mask_real_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'fixture.jsonl'
            p.write_text('\n'.join(json.dumps(dict(pitch=n,duration=.2,velocity=80,delta_time=.2)) for n in [60,62,64]))
            ds=EventWindowDataset([p],8);x,y,target,tempo,mask=ds[0]
        self.assertEqual(mask.tolist(),[True,True,False,False,False,False,False,False])
        self.assertEqual(int(target),64);self.assertEqual(float(tempo),100)

    def test_loss_ignores_padded_values_and_gradients(self):
        torch.manual_seed(12)
        out={k:torch.randn(1,4,n,requires_grad=True) for k,n in [('pitch',128),('duration',16),('velocity',8)]}
        out['timing']=torch.randn(1,4,requires_grad=True)
        y=torch.zeros((1,4,3),dtype=torch.long);t=torch.zeros((1,4));mask=torch.tensor([[True,True,False,False]])
        loss,_=event_loss(out,y,t,mask)
        changed={k:v.clone() for k,v in out.items()}
        for v in changed.values():v[:,2:]=1000
        other,_=event_loss(changed,y,t,mask);self.assertAlmostEqual(float(loss.detach()),float(other.detach()),places=6)
        loss.backward()
        for v in out.values():self.assertEqual(float(v.grad[:,2:].abs().sum()),0)

    def test_composition_group_splits(self):
        files=[Path(f'{i:04d}_{instrument}_hash.jsonl') for i in range(30) for instrument in ['Drums','Piano']]
        splits=split_files(files);ids=[{p.name.split('_')[0] for p in s} for s in splits]
        self.assertTrue(all(not(ids[a]&ids[b]) for a,b in [(0,1),(0,2),(1,2)]))

    def test_missing_checkpoint_is_safe(self):
        g=LSTMGenerator(Path('__nonexistent_audit_checkpoint__.pt'))
        self.assertFalse(g.loaded);self.assertIn('unavailable',g.load_error)

    def test_annotated_mask_fixture(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';data.mkdir()
            np.zeros((3,6),dtype=np.uint16).tofile(data/'test_categorical.bin')
            np.zeros((3,7),dtype=np.float32).tofile(data/'test_numeric.bin')
            for name,n in [('categorical',6),('numeric',7)]:
                (data/f'test_{name}.shape.json').write_text(json.dumps({'events':3,'features':n}))
            (data/'test_index.jsonl').write_text(json.dumps({'start':0,'count':3,'validation_status':'valid'}))
            ds=AAMWindowDataset(root,'test',8);row=ds[0]
            self.assertEqual(int(row[5].sum()),2);self.assertEqual(row[6].shape,(8,))
            ds.cat._mmap.close();ds.numeric._mmap.close()

    @unittest.skipUnless(Path('models/best_model_3x512.pt').exists(),'local checkpoint not distributed')
    def test_local_frozen_checkpoints_duration(self):
        for stem in ['best_model_3x512.pt','best_model_4x768.pt']:
            if not Path('models',stem).exists():continue
            g=LSTMGenerator(Path('models',stem));self.assertTrue(g.loaded,g.load_error)
            for bpm in [60,100,160]:
                for seconds in [.5,1,2]:
                    c=Composition();torch.manual_seed(41027)
                    with contextlib.redirect_stdout(io.StringIO()):events=g.generate_continuation(c,'A',seconds,bpm)
                    self.assertLessEqual(max(e.start+e.duration for e in events),human_note_duration(bpm)+seconds+1e-7)

if __name__=='__main__':unittest.main()
