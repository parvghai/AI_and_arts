"""LoRA learning, frozen-weight, serialization and deployment regressions."""
import copy
import contextlib
import io
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
import torch
from ml.model import MusicLSTM
from ml.lora import attach_lora, adapter_parameters, save_adapter, apply_adapter, load_base
from ml.losses import event_loss
from ml.inference import LSTMGenerator


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.config = {'embedding_size':8,'hidden_size':12,'layers':2,'dropout':0.,
                       'sequence_length':8,'target_conditioning':'future_window_final_pitch'}
        self.model = MusicLSTM(embedding_size=8,hidden_size=12,num_layers=2,dropout=0)
        self.base = self.root/'base.pt'
        torch.save({'model_state':self.model.state_dict(),'config':self.config},self.base)
        self.inputs = (torch.randint(0,128,(2,8)),torch.randint(0,16,(2,8)),
                       torch.randint(0,8,(2,8)),torch.randint(0,16,(2,8)),
                       torch.tensor([60,64]),torch.tensor([100.,100.]))

    def tearDown(self):
        self.tmp.cleanup()

    def train_adapter(self):
        model = self.model
        attach_lora(model,rank=2,alpha=4)
        frozen = {k:p.clone() for k,p in model.named_parameters() if not p.requires_grad}
        params = adapter_parameters(model)
        self.assertEqual(set(params),{k for k,p in model.named_parameters() if p.requires_grad})
        optimizer = torch.optim.AdamW(params.values(),lr=.01)
        losses = []
        for _ in range(4):
            optimizer.zero_grad()
            out,_ = model(*self.inputs)
            targets = torch.stack(self.inputs[:3],dim=-1)
            loss,_ = event_loss(out,targets,torch.zeros(2,8),torch.ones(2,8,dtype=torch.bool))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        self.assertLess(losses[-1],losses[0])
        self.assertTrue(all(torch.equal(dict(model.named_parameters())[k],v) for k,v in frozen.items()))
        self.assertTrue(any(p.abs().sum()>0 for k,p in params.items() if k.endswith('.B')))
        return model

    def test_zero_initialization_matches_base(self):
        expected,_ = self.model(*self.inputs)
        attach_lora(self.model,2,4)
        actual,_ = self.model(*self.inputs)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)

    def test_learning_freezing_save_merge_and_generator(self):
        model = self.train_adapter(); expected,_ = model(*self.inputs)
        path = self.root/'adapter.pt'; save_adapter(path,model,self.base,2,4)
        payload = torch.load(path,weights_only=True)
        self.assertNotIn('model_state',payload)
        merged,_ = load_base(self.base); apply_adapter(merged,path,self.base)
        actual,_ = merged(*self.inputs)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=1e-5,atol=1e-6)
        self.assertFalse(adapter_parameters(merged))
        generator = LSTMGenerator(self.base,adapter=path)
        self.assertTrue(generator.loaded,generator.load_error)
        self.assertTrue(generator.adapter_status.startswith('active'))
        generator.load_checkpoint(self.base)  # Reload cannot double-apply deltas.
        device_inputs = [x.to(generator.device) for x in self.inputs]
        expected_device,_ = merged.to(generator.device)(*device_inputs)
        actual,_ = generator.model(*device_inputs)
        for key in expected:torch.testing.assert_close(actual[key],expected_device[key],rtol=1e-5,atol=1e-6)
        other=self.root/'other.pt'
        other_model,_=load_base(self.base)
        with torch.no_grad():other_model.pitch_head.bias.add_(1)
        torch.save({'model_state':other_model.state_dict(),'config':self.config},other)
        generator.load_checkpoint(other)
        self.assertFalse(generator.loaded)
        self.assertIn('SHA-256',generator.load_error)
        generator.set_adapter(None)
        self.assertTrue(generator.loaded,generator.load_error)
        self.assertEqual(generator.adapter_status,'disabled')
        torch.testing.assert_close(generator.model.pitch_head.bias.cpu(),other_model.pitch_head.bias)

    def test_reject_wrong_base_and_corrupt_weights_without_mutation(self):
        model = self.train_adapter(); path = self.root/'adapter.pt'
        save_adapter(path,model,self.base,2,4)
        original,_ = load_base(self.base); before = copy.deepcopy(original.state_dict())
        wrong = self.root/'wrong.pt'; wrong.write_bytes(b'wrong checkpoint')
        with self.assertRaisesRegex(ValueError,'SHA-256'):apply_adapter(original,path,wrong)
        payload = torch.load(path,weights_only=True)
        next(iter(payload['adapter_state'].values())).fill_(float('nan'))
        torch.save(payload,path)
        with self.assertRaisesRegex(ValueError,'Non-finite'):apply_adapter(original,path,self.base)
        self.assertTrue(all(torch.equal(v,original.state_dict()[k]) for k,v in before.items()))

    def test_invalid_options_and_no_overwrite(self):
        for rank,alpha in [(0,1),(65,1),(2,float('nan')),(2,-1)]:
            with self.assertRaises(ValueError):attach_lora(self.model,rank,alpha)
        attach_lora(self.model,2,4)
        with self.assertRaises(ValueError):attach_lora(self.model,2,4)
        path=self.root/'adapter.pt'; save_adapter(path,self.model,self.base,2,4)
        with self.assertRaises(FileExistsError):save_adapter(path,self.model,self.base,2,4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not installed')
    def test_cuda_training_process_exits_cleanly(self):
        code = '''
import torch
from ml.lora import load_base, attach_lora, adapter_parameters
import sys
torch.set_num_threads(2)
model,_=load_base(sys.argv[1],'cuda')
attach_lora(model,2,4)
optimizer=torch.optim.AdamW(adapter_parameters(model).values(),lr=.001)
pitch=torch.ones(2,8,dtype=torch.long,device='cuda')
for i in range(3):
    optimizer.zero_grad()
    out,_=model(pitch,pitch,pitch,pitch,pitch[:,0],torch.ones(2,device='cuda')*100)
    sum(v.square().mean() for v in out.values()).backward()
    optimizer.step()
torch.cuda.synchronize()
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(self.base)],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_evaluator_accepts_adapter_and_records_hash(self):
        from ml.evaluate import evaluate
        from ml.lora import sha256
        model=self.train_adapter();adapter=self.root/'adapter.pt'
        save_adapter(adapter,model,self.base,2,4)
        data=self.root/'data';data.mkdir()
        for name in ('song_a','song_b','song_c'):
            (data/f'{name}.jsonl').write_text('\n'.join(json.dumps({'pitch':60+i,'duration':.25,'velocity':80,'delta_time':.25}) for i in range(6)),encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            report=evaluate(data,self.base,self.root/'report.json',2,adapter)
        self.assertEqual(report['adapter_sha256'],sha256(adapter))
        self.assertEqual(report['real_targets']['target_events'],5)


if __name__ == '__main__':
    unittest.main()
