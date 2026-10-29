"""Opt-in LoRA controls for the original studio; no ML imports while disabled."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class LoRAControls:
    def __init__(self, app, parent, adapter_root, hint=None):
        self.app = app
        self.adapter_root = Path(adapter_root)
        self.hint = Path(hint) if hint else None
        self.enabled = tk.BooleanVar(master=app.root, value=False)
        self.selection = tk.StringVar(master=app.root)
        self.details = tk.StringVar(master=app.root)
        self.message = tk.StringVar(master=app.root, value='LoRA OFF')
        self.paths = {}
        self.active_path = None
        self.window = None
        self.controls = []
        self.toggle = tk.Checkbutton(parent, text='Enable LoRA', variable=self.enabled,
                                    command=self.changed, bg='#11151d', fg='#edf2f7',
                                    activebackground='#11151d', activeforeground='#edf2f7',
                                    selectcolor='#1b2230', font=('Segoe UI',9))
        self.toggle.pack(side='left', padx=4)
        app._busy_controls.append(self.toggle)
        self.settings_button = app._button(parent, 'LoRA settings', self.show)
        app._busy_controls.append(self.settings_button)

    @property
    def ready(self):
        return (not self.enabled.get() or
                (self.active_path is not None and self.app.ml_generator.loaded and
                 self.app.ml_generator.adapter_path == self.active_path and
                 self.app.ml_generator.adapter_status.startswith('active:')))

    def changed(self):
        if self.app._generating:
            return
        self.app.player.stop()
        if self.enabled.get():
            self.active_path = None
            self.settings_button.pack(side='left', padx=4)
            self.show()
            self.refresh()
        else:
            self.disable()

    def disable(self):
        if self.app._generating:
            return
        self.enabled.set(False)
        self.active_path = None
        self.selection.set(''); self.details.set('')
        self.app.player.stop()
        self.app._reload_selected_generator(None)
        self.settings_button.pack_forget()
        self.close_window()
        self.message.set('LoRA OFF — original checkpoint restored')
        self.app.status.set(self.message.get() if self.app.ml_generator.loaded else
                            self.app.ml_generator.load_error or 'Original model unavailable')
        self.app._refresh_model_label()

    def close_window(self):
        for control in self.controls:
            if control in self.app._busy_controls:
                self.app._busy_controls.remove(control)
        self.controls.clear()
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def show(self):
        if not self.enabled.get():
            return
        if self.window is not None:
            self.window.lift()
            return
        self.window = tk.Toplevel(self.app.root)
        self.window.title('LoRA adapters')
        self.window.configure(bg='#11151d')
        self.window.transient(self.app.root)
        self.window.protocol('WM_DELETE_WINDOW', self.disable)
        panel = tk.Frame(self.window, bg='#11151d', padx=18, pady=16)
        panel.pack(fill='both', expand=True)
        tk.Label(panel, text='LoRA for the selected LSTM', bg='#11151d', fg='#b89cff',
                 font=('Segoe UI',14,'bold')).pack(anchor='w')
        tk.Label(panel, text='Choose an adapter, then Apply. Closing this window turns LoRA OFF.',
                 bg='#11151d', fg='#edf2f7', wraplength=600).pack(anchor='w', pady=8)
        self.box = ttk.Combobox(panel, textvariable=self.selection, state='readonly', width=70)
        self.box.pack(fill='x'); self.box.bind('<<ComboboxSelected>>', self.selected)
        self.controls.append(self.box)
        row = tk.Frame(panel, bg='#11151d'); row.pack(fill='x', pady=8)
        for label, fn in [('Apply adapter', self.apply), ('Browse…', self.browse),
                          ('Refresh', self.refresh), ('Training / evaluation help', self.help)]:
            button = self.app._button(row, label, fn); button.pack(side='left', padx=3)
            self.controls.append(button)
        for variable, colour in [(self.details,'#edf2f7'),(self.message,'#55c2ff')]:
            tk.Label(panel, textvariable=variable, bg='#11151d', fg=colour,
                     wraplength=620, justify='left', anchor='w').pack(fill='x', pady=8)
        self.app._busy_controls.extend(self.controls)

    def _describe(self, payload, path):
        metadata = payload.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('Invalid adapter metadata')
        training = 'Smoke-test adapter only; musical quality is unverified.' if metadata.get('smoke_only') else 'Musical quality is unverified.'
        return (f'{self.app.model_choice.get()}\n{path}\n'
                f"Rank: {payload['rank']}   Alpha: {payload['alpha']}   "
                f"Training epoch: {metadata.get('epoch', 'not recorded')}\n{training}")

    def _read_compatible(self, path):
        from ml.lora import read_adapter, sha256
        payload = read_adapter(path)
        if payload.get('base_sha256') != sha256(self.app._selected_checkpoint()):
            raise ValueError('Adapter is incompatible with the selected LSTM checkpoint')
        self._describe(payload, path)
        return payload

    def refresh(self):
        if not self.enabled.get() or self.app._generating:
            return
        # Refresh deliberately unloads a previous adapter: never display one
        # selection while generating with a different adapter.
        if self.active_path is not None:
            self.app._reload_selected_generator(None)
        self.active_path = None; self.selection.set(''); self.details.set('')
        self.paths = {}; invalid = 0
        try:
            from ml.lora import read_adapter, sha256
            base_hash = sha256(self.app._selected_checkpoint())
            files = sorted(self.adapter_root.rglob('adapter*.pt')) if self.adapter_root.exists() else []
            if self.hint and self.hint not in files:
                files.insert(0, self.hint)
            for path in files:
                try:
                    payload = read_adapter(path)
                    if payload.get('base_sha256') != base_hash:
                        continue
                    self._describe(payload, path)
                    label = str(path.relative_to(self.adapter_root)) if path.is_relative_to(self.adapter_root) else str(path)
                    if payload.get('metadata', {}).get('smoke_only'):
                        label += ' [smoke test]'
                    self.paths[label] = path
                except Exception:
                    invalid += 1
            self.box.configure(values=tuple(self.paths))
            msg = ('Select a compatible adapter and click Apply; generation is paused.' if self.paths else
                   'No compatible adapters found for this model. Browse for an adapter or train one separately. LoRA has not been applied.')
            if invalid:
                msg += f' Skipped {invalid} missing/invalid file(s).'
            self.notify(msg)
        except Exception as exc:
            self.box.configure(values=())
            self.notify(f'Cannot find adapters: {exc}. Turn LoRA OFF to use the original workflow.')

    def selected(self, _event=None):
        if self.app._generating or not self.enabled.get():
            return
        if self.active_path is not None:
            self.app._reload_selected_generator(None)
        self.active_path = None
        try:
            path = self.paths[self.selection.get()]
            self.details.set(self._describe(self._read_compatible(path), path))
            self.notify('Adapter selected. Click Apply before generating.')
        except Exception as exc:
            self.details.set(''); self.notify(f'Cannot select adapter: {exc}')

    def browse(self):
        if not self.enabled.get() or self.app._generating:
            return
        filename = filedialog.askopenfilename(parent=self.window, title='Select LoRA adapter',
                                             initialdir=str(self.adapter_root), filetypes=[('LoRA adapter','*.pt')])
        if not filename:
            return
        path = Path(filename)
        # Keep an existing active selection if browsing an invalid file fails.
        try:
            payload = self._read_compatible(path)
            label = str(path)
            self.paths[label] = path; self.box.configure(values=tuple(self.paths))
            self.selection.set(label); self.selected()
        except Exception as exc:
            suffix = ' The previous adapter remains active.' if self.ready and self.active_path else ' No adapter is active.'
            self.notify(f'Cannot use adapter: {exc}.{suffix}')

    def apply(self):
        if not self.enabled.get() or self.app._generating:
            return
        self.active_path = None
        try:
            path = self.paths[self.selection.get()]
            payload = self._read_compatible(path)
            self.app.player.stop()
            self.app._reload_selected_generator(path)
            if not self.app.ml_generator.loaded:
                raise ValueError(self.app.ml_generator.load_error)
            self.active_path = path
            self.details.set(self._describe(payload, path))
            self.notify('LoRA active for the selected model. Original checkpoint is unchanged.')
        except Exception as exc:
            self.notify(f'LoRA not applied: {exc}. Select a valid adapter or turn LoRA OFF.')

    def model_changed(self):
        self.active_path = None
        if self.enabled.get():
            self.refresh()

    def notify(self, text):
        self.message.set(text); self.app.status.set(text)
        self.app._refresh_model_label()

    def help(self):
        base = self.app._selected_checkpoint()
        adapter = self.paths.get(self.selection.get())
        text = (f'Optional training (run separately from the project folder):\n\n'
                f'python -B -m ml.train_lora --base "{base}" --output runs/lora/my_adapter\n\n'
                'Choose a new output folder for each run. Base checkpoints are never overwritten. '
                'Refresh this list after training. Training never starts automatically.\n\n'
                'Evaluation uses the original base unless --adapter is explicitly supplied.\n')
        if adapter:
            text += (f'\npython -B -m ml.evaluate --data data/processed --checkpoint "{base}" '
                     f'--adapter "{adapter}" --output runs/lora/my_evaluation\n')
        text += '\nSee docs/LORA.md for details.'
        messagebox.showinfo('Optional LoRA training / evaluation', text, parent=self.window)
