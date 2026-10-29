"""Real Tk optional-LoRA controls with small compatible MusicLSTM fixtures."""
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch
import torch
from ml.model import MusicLSTM
from ml.lora import attach_lora, adapter_parameters, save_adapter, sha256
from ml.inference import LSTMGenerator
from gui import MusicApp


class LoRAGUITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        (self.folder/'models').mkdir();self.adapters=self.folder/'runs/lora';self.adapters.mkdir(parents=True)
        self.mapping={'LSTM 3x512 (baseline)':'baseline.pt','LSTM 4x768 (experimental)':'large.pt'}
        for i,name in enumerate(self.mapping.values()):
            torch.manual_seed(4+i)
            model=MusicLSTM(embedding_size=8,hidden_size=12+i*4,num_layers=2,dropout=0)
            config={'embedding_size':8,'hidden_size':12+i*4,'layers':2,'dropout':0,
                    'sequence_length':8,'target_conditioning':'future_window_final_pitch'}
            base=self.folder/'models'/name
            torch.save({'model_state':model.state_dict(),'config':config},base)
            attach_lora(model,2,4)
            with torch.no_grad():
                for key,parameter in adapter_parameters(model).items():
                    if key.endswith('.B'):parameter.fill_(.01)
            save_adapter(self.adapters/f'adapter_{i}.pt',model,base,2,4,{'epoch':1})
        self.hashes={p:sha256(p) for p in (self.folder/'models').glob('*.pt')}
        self.root=tk.Tk();self.root.withdraw()
        mapping=self.mapping
        class FixtureGenerator(LSTMGenerator):
            CHECKPOINTS=mapping
            def __init__(instance,checkpoint,*args,**kwargs):
                checkpoint=Path(checkpoint)
                if checkpoint.name=='best_model_3x512.pt':checkpoint=checkpoint.with_name('baseline.pt')
                super().__init__(checkpoint,*args,**kwargs)
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch('gui.tk.Tk',return_value=self.root))
        self.stack.enter_context(patch('gui.APP_DIR',self.folder))
        self.stack.enter_context(patch('gui.LOG_DIR',self.folder/'logs'))
        self.stack.enter_context(patch('gui.MidiPlayer',return_value=MagicMock()))
        self.stack.enter_context(patch('gui.LSTMGenerator',FixtureGenerator))
        self.stack.enter_context(patch.dict(os.environ,{'MUSIC_LORA_ADAPTER':str(self.adapters/'adapter_0.pt')}))
        with patch('ml.lora.read_adapter',side_effect=AssertionError('OFF must not read adapters')):
            self.app=MusicApp(adapter_hint=self.adapters/'adapter_0.pt')

    def tearDown(self):
        self.app._closed=True;self.app.lora.close_window()
        for token in self.root.tk.call('after','info'):self.root.after_cancel(token)
        self.root.destroy();self.stack.close()
        for path,value in self.hashes.items():self.assertEqual(sha256(path),value)
        del self.app;gc.collect();self.tmp.cleanup()

    def enable_apply(self):
        controls=self.app.lora;controls.enabled.set(True);controls.changed()
        self.assertEqual(len(controls.paths),1)
        controls.selection.set(next(iter(controls.paths)));controls.selected();controls.apply()
        self.assertTrue(controls.ready,controls.message.get())
        return controls

    def test_starts_off_even_with_environment_and_hint(self):
        app=self.app
        self.assertFalse(app.lora.enabled.get());self.assertIsNone(app.lora.window)
        self.assertEqual(app.lora.toggle.cget('text'),'Enable LoRA')
        self.assertIsNone(app.ml_generator.adapter_path)
        self.assertEqual(app.ml_generator.adapter_status,'disabled')
        with patch('ml.lora.read_adapter',side_effect=AssertionError('OFF must not scan')):
            app.lora.refresh()
            app.model_choice.set('LSTM 4x768 (experimental)');app._select_model()
        self.assertTrue(app.ml_generator.loaded,app.ml_generator.load_error)
        self.assertEqual(app.ml_generator.adapter_status,'disabled')

    def test_enable_disable_restores_exact_base_weights_for_both_models(self):
        for choice in self.mapping:
            with self.subTest(model=choice):
                self.app.model_choice.set(choice);self.app._select_model()
                before={k:v.detach().cpu().clone() for k,v in self.app.ml_generator.model.state_dict().items()}
                controls=self.enable_apply()
                self.assertIn('Rank: 2',controls.details.get())
                self.assertTrue(any(not torch.equal(v,self.app.ml_generator.model.state_dict()[k].cpu()) for k,v in before.items()))
                controls.enabled.set(False);controls.changed()
                self.assertIsNone(controls.window)
                self.assertTrue(controls.ready)
                self.assertEqual(self.app.ml_generator.adapter_status,'disabled')
                for k,v in before.items():torch.testing.assert_close(v,self.app.ml_generator.model.state_dict()[k].cpu(),rtol=0,atol=0)

    def test_switch_requires_new_compatible_selection_and_blocks_generation(self):
        controls=self.enable_apply()
        self.app.model_choice.set('LSTM 4x768 (experimental)');self.app._select_model()
        self.assertFalse(controls.ready);self.assertEqual(controls.selection.get(),'')
        self.assertTrue(all(p.name=='adapter_1.pt' for p in controls.paths.values()))
        self.app._choose_note('C');self.assertEqual(self.app.composition.events,[])
        self.assertIn('not ready',self.app.status.get())
        controls.selection.set(next(iter(controls.paths)));controls.selected();controls.apply()
        self.assertTrue(controls.ready,controls.message.get())

    def test_missing_and_corrupt_adapters_never_silently_fall_back(self):
        controls=self.app.lora
        controls.adapter_root=self.folder/'empty';controls.hint=None
        controls.enabled.set(True);controls.changed()
        self.assertIn('No compatible adapters',controls.message.get())
        self.assertFalse(controls.ready)
        controls.adapter_root=self.adapters;controls.refresh()
        label=next(iter(controls.paths));controls.selection.set(label);controls.selected()
        path=controls.paths[label]
        payload=torch.load(path,weights_only=True)
        next(iter(payload['adapter_state'].values())).fill_(float('nan'));torch.save(payload,path)
        controls.apply()
        self.assertFalse(controls.ready);self.assertIn('Non-finite',controls.message.get())
        self.app._choose_note('C');self.assertEqual(self.app.composition.events,[])
        controls.disable();self.assertTrue(self.app.ml_generator.loaded)

    def test_missing_selected_file_and_incompatible_browse(self):
        controls=self.enable_apply()
        with patch('lora_controls.filedialog.askopenfilename',return_value=str(self.adapters/'adapter_1.pt')):
            controls.browse()
        self.assertIn('incompatible',controls.message.get())
        self.assertIn('previous adapter remains active',controls.message.get())
        self.assertTrue(controls.ready)
        path=controls.paths[controls.selection.get()];path.unlink()  # Owned temporary fixture only.
        controls.apply();self.assertFalse(controls.ready)
        self.assertIn('LoRA not applied',controls.message.get())
        controls.disable();self.assertIsNone(self.app.ml_generator.adapter_path)

    def test_lora_inference_error_does_not_use_symbolic_fallback(self):
        self.enable_apply()
        with patch.object(self.app.ml_generator,'generate_continuation',side_effect=RuntimeError('fixture failure')),patch.object(self.app.generator,'generate_continuation') as fallback:
            with self.assertRaisesRegex(RuntimeError,'no fallback'):
                self.app._generate_worker('C',1,100,.75,'2026-10-29T00:00:00+00:00',[],None)
            fallback.assert_not_called()

    def test_close_settings_turns_off_and_busy_controls_are_disabled(self):
        controls=self.enable_apply()
        self.app._set_busy(True)
        self.assertEqual(str(controls.toggle.cget('state')),'disabled')
        self.assertEqual(str(controls.box.cget('state')),'disabled')
        self.app._set_busy(False)
        self.assertEqual(str(controls.box.cget('state')),'readonly')
        controls.disable();self.assertFalse(controls.enabled.get())

    def test_generation_appends_and_plays_with_lora_enabled(self):
        self.enable_apply()
        with contextlib.redirect_stdout(io.StringIO()):
            self.app._generate_worker('C',.5,100,.75,'2026-10-29T00:00:00+00:00',[],None)
            self.root.update()
        self.assertGreater(len(self.app.composition.events),1)
        self.app.player.play.assert_called_once()

    def test_launch_argument_is_only_a_hint(self):
        import main
        with patch('sys.argv',['main.py','--adapter','missing.pt']),patch('gui.MusicApp') as constructor:
            main.main()
        constructor.assert_called_once_with(adapter_hint=Path('missing.pt'))
        constructor.return_value.run.assert_called_once()


class RealModelToggleTests(unittest.TestCase):
    def test_real_checkpoints_off_on_off_with_seeded_inference(self):
        from composition import Composition
        root=Path(__file__).resolve().parents[1]
        cases=[('3x512','baseline'),('4x768','large')]
        if not all((root/f'models/best_model_{size}.pt').exists() and
                   (root/f'runs/lora/verified_{label}_20261029/adapter_epoch_001.pt').exists() for size,label in cases):
            self.skipTest('Real checkpoints and trained smoke adapters are local-only')
        for size,label in cases:
            with self.subTest(model=size),contextlib.redirect_stdout(io.StringIO()):
                base=root/f'models/best_model_{size}.pt';before=sha256(base)
                adapter=root/f'runs/lora/verified_{label}_20261029/adapter_epoch_001.pt'
                with patch.dict(os.environ,{'MUSIC_LORA_ADAPTER':str(adapter)}),patch('ml.lora.read_adapter',side_effect=AssertionError('Implicit adapter load')):
                    generator=LSTMGenerator(base)
                self.assertTrue(generator.loaded,generator.load_error)
                torch.manual_seed(1234);expected=generator.generate_continuation(Composition(),'C',.5,100)
                generator.set_adapter(adapter)
                self.assertTrue(generator.loaded,generator.load_error)
                self.assertTrue(generator.adapter_status.startswith('active:'))
                self.assertTrue(generator.generate_continuation(Composition(),'C',.5,100))
                with patch('ml.lora.read_adapter',side_effect=AssertionError('OFF must not load adapters')):
                    generator.set_adapter(None)
                torch.manual_seed(1234);actual=generator.generate_continuation(Composition(),'C',.5,100)
                self.assertEqual(actual,expected)
                self.assertEqual(sha256(base),before)

    def test_real_models_evaluate_with_explicit_adapter_only(self):
        from ml.evaluate import evaluate
        root=Path(__file__).resolve().parents[1]
        for size,label in [('3x512','baseline'),('4x768','large')]:
            base=root/f'models/best_model_{size}.pt'
            adapter=root/f'runs/lora/verified_{label}_20261029/adapter_epoch_001.pt'
            if not base.exists() or not adapter.exists():self.skipTest('Local checkpoints/adapters required')
            with self.subTest(model=size),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                folder=Path(tmp);data=folder/'data';data.mkdir()
                for name in ('song_a','song_b','song_c'):
                    (data/f'{name}.jsonl').write_text('\n'.join(json.dumps(dict(pitch=60+i,duration=.25,velocity=80,delta_time=.25)) for i in range(6)),encoding='utf-8')
                with patch('ml.lora.read_adapter',side_effect=AssertionError('OFF evaluation loaded an adapter')):
                    off=evaluate(data,base,folder/'off.json',1)
                on=evaluate(data,base,folder/'on.json',1,adapter)
                self.assertIsNone(off['adapter'])
                self.assertEqual(on['adapter_sha256'],sha256(adapter))
                self.assertEqual(on['real_targets']['target_events'],5)
                self.assertEqual(off['real_targets']['target_events'],5)


if __name__=='__main__':unittest.main()
