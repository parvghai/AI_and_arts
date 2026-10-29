"""Local turn-taking composition studio with reference guidance and editable candidates."""
from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from composition import Composition, NoteEvent
from config import APP_DIR, NOTES
from music_io import CompositionDocument, midi_file, read_midi, validate_events
from guided_generation import GuidedComposer, GenerationSettings, OBJECTIVES, NAMES
from midi_player import MidiPlayer
from reference_audio import ReferenceAnalyzer, AudioPreview

BG, PANEL, TEXT, HUMAN, AI = "#11151d", "#1b2230", "#edf2f7", "#55c2ff", "#b89cff"

NOTE_KEYS={n.lower():n for n in NOTES}
NOTE_KEYS['h']='B'
CHECKPOINTS={'LSTM 3x512 (baseline)':'best_model_3x512.pt',
             'LSTM 4x768 (experimental)':'best_model_4x768.pt','Symbolic fallback':None}

class MusicApp:
    def __init__(self,root=None,autoload=True):
        self.root=root or tk.Tk()
        self.root.title('AI Music Co-Creator • Music Studio')
        self.root.geometry(f"{min(980,self.root.winfo_screenwidth()-80)}x{min(720,self.root.winfo_screenheight()-120)}")
        self.root.minsize(820,620)
        self.root.configure(bg=BG)
        self.document=CompositionDocument()
        self.ml_generator=None;self.reference=None;self.reference_path=None
        self.analyzer=ReferenceAnalyzer();self.preview=AudioPreview()
        self.candidates=[];self.replace_range=None;self._closed=False;self._busy=False
        self._queue=queue.Queue();self._controls=[];self._session_path=None
        self.status=tk.StringVar(value='Ready')
        self.model_choice=tk.StringVar(value='LSTM 3x512 (baseline)')
        self.model_status=tk.StringVar(value='Symbolic composer ready; model not loaded')
        self.objective=tk.StringVar(value=OBJECTIVES[0]);self.target=tk.StringVar(value='C')
        self.bpm=tk.DoubleVar(value=100);self.seconds=tk.DoubleVar(value=1)
        self.temperature=tk.DoubleVar(value=.75);self.influence=tk.DoubleVar(value=.6)
        self.count=tk.IntVar(value=3);self.seed=tk.IntVar(value=42)
        self.follow_tempo=tk.BooleanVar(value=False)
        self.follow={name:tk.BooleanVar(value=True) for name in ('harmony','rhythm','energy','contour')}
        self.edit_target=tk.StringVar(value='Composition')
        self.player=MidiPlayer(lambda value:self._queue.put(('status',value)))
        self._build();self.root.bind('<KeyPress>',self._note_key)
        self.root.protocol('WM_DELETE_WINDOW',self._on_close)
        self.root.after(60,self._poll)
        if autoload:self.root.after(100,self._load_model)

    @property
    def composition(self):return self.document.composition

    def _button(self,parent,text,command,controlled=True):
        b=ttk.Button(parent,text=text,command=command)
        b.pack(side='left',padx=3,pady=3)
        if controlled:self._controls.append(b)
        return b

    def _build(self):
        style=ttk.Style(self.root);style.theme_use('clam')
        style.configure('.',background=BG,foreground=TEXT)
        style.configure('TButton',background='#293446',foreground=TEXT,padding=(10,7))
        style.map('TButton',background=[('active','#354761')],foreground=[('disabled','#9ba8ba')])
        for name in ('TCombobox','TSpinbox','TEntry','Treeview'):
            style.configure(name,fieldbackground=PANEL,background=PANEL,foreground=TEXT,insertcolor=TEXT)
            style.map(name,fieldbackground=[('readonly',PANEL),('disabled',PANEL)],foreground=[('readonly',TEXT),('disabled','#9ba8ba')])
        style.configure('Treeview.Heading',background='#293446',foreground=TEXT)
        style.map('Treeview',background=[('selected','#354761')],foreground=[('selected',TEXT)])
        style.configure('TNotebook.Tab',background=PANEL,foreground=TEXT,padding=(12,6))
        style.map('TNotebook.Tab',background=[('selected','#293446')])
        style.configure('TLabelframe',background=PANEL)
        style.configure('TLabelframe.Label',background=PANEL,foreground=HUMAN)
        self.root.option_add('*TCombobox*Listbox.background',PANEL)
        self.root.option_add('*TCombobox*Listbox.foreground',TEXT)
        header=ttk.Frame(self.root,padding=(14,10));header.pack(fill='x')
        ttk.Label(header,text='AI Music Co-Creator',font=('Segoe UI',21,'bold')).pack(side='left')
        ttk.Label(header,textvariable=self.model_status,wraplength=600).pack(side='right')
        bar=ttk.Frame(self.root,padding=(10,0));bar.pack(fill='x')
        for text,fn in [('Open session',self._open),('Save session',self._save),('Import MIDI',self._import_midi),
                        ('Export MIDI',self._export_midi),('Export interaction log',self._save)]:
            self._button(bar,text,fn)
        ttk.Label(self.root,textvariable=self.status,anchor='w',padding=(14,8),wraplength=1050).pack(side='bottom',fill='x')
        self.tabs=ttk.Notebook(self.root);self.tabs.pack(fill='both',expand=True,padx=12,pady=8)
        compose=ttk.Frame(self.tabs,padding=12);reference=ttk.Frame(self.tabs,padding=12);editor=ttk.Frame(self.tabs,padding=12)
        compare=ttk.Frame(self.tabs,padding=12)
        self.tabs.add(compose,text='Compose');self.tabs.add(compare,text='Compare candidates');self.tabs.add(reference,text='Reference');self.tabs.add(editor,text='Note editor')
        self.compare_tab=compare
        self.compose_tab=compose;self.editor_tab=editor
        cues=ttk.LabelFrame(compose,text='YOUR NOTE — choose a note to append and play the next phrase',padding=6);cues.pack(fill='x')
        keys=tk.Frame(cues,bg=PANEL);keys.pack(fill='x')
        for n in NOTES:
            button=tk.Button(keys,text=n,command=lambda note=n:self._choose_note(note),
                             font=('Segoe UI',17,'bold'),fg=TEXT,bg='#293446',activebackground='#354761',
                             activeforeground=TEXT,relief='flat',cursor='hand2',height=2)
            button.pack(side='left',expand=True,fill='x',padx=4);self._controls.append(button)
        tk.Label(cues,text='Keyboard: C D E F G A B · H also plays B',bg=PANEL,fg='#9ba8ba').pack(anchor='w',pady=(8,0))
        toolbar=ttk.Frame(compose);toolbar.pack(fill='x',pady=8)
        for label,fn in [('Play',lambda:self.player.play(self.composition.events)),('Pause / Resume',self._pause),
                         ('Stop',self._stop),('Undo',self._undo),('Clear',self._clear)]:
            self._button(toolbar,label,fn,controlled=label in ('Undo','Clear'))
        controls=ttk.Frame(compose);controls.pack(fill='x',pady=8)
        self.model_box=self._combo(controls,'Model',self.model_choice,list(CHECKPOINTS),0,28)
        self.model_box.bind('<<ComboboxSelected>>',lambda _:self._load_model())
        self._combo(controls,'Continuation (seconds)',self.seconds,(.5,1,2),1,13)
        numbers=ttk.Frame(compose);numbers.pack(fill='x')
        for label,var,lo,hi,step in [('Tempo',self.bpm,60,160,1),('Temperature',self.temperature,.2,1.5,.05)]:
            ttk.Label(numbers,text=label).pack(side='left',padx=8)
            slider=tk.Scale(numbers,from_=lo,to=hi,resolution=step,variable=var,orient='horizontal',length=170,
                            bg=BG,fg=TEXT,troughcolor=PANEL,highlightthickness=0)
            slider.pack(side='left');self._controls.append(slider)
        ttk.Label(compose,text='COMPOSITION',foreground='#9ba8ba').pack(anchor='w',pady=(12,4))
        self.sequence=ttk.Label(compose,text='No notes yet',font=('Segoe UI',12,'bold'),wraplength=880)
        self.sequence.pack(fill='x',pady=4)
        options=ttk.Frame(compare);options.pack(fill='x')
        self._combo(options,'Objective',self.objective,OBJECTIVES,0,25)
        for col,(label,var,lo,hi,step) in enumerate([('Candidates',self.count,1,5,1),('Seed',self.seed,0,999999,1)],1):
            cell=ttk.Frame(options);cell.grid(row=0,column=col,padx=10)
            ttk.Label(cell,text=label).pack(anchor='w')
            spin=ttk.Spinbox(cell,textvariable=var,from_=lo,to=hi,increment=step,width=10);spin.pack();self._controls.append(spin)
        actions=ttk.Frame(compare);actions.pack(fill='x',pady=8)
        self._button(actions,'Generate candidates',self._generate)
        self._button(actions,'Regenerate selected time range',lambda:self._generate(replace=True))
        self.candidate_list=ttk.Treeview(compare,columns=('engine','notes','span','diversity'),show='tree headings',height=4)
        self.candidate_list.heading('#0',text='Candidate')
        for c,label,width in [('engine','Generator',290),('notes','Notes',60),('span','AI span',90),('diversity','Unique pitches',100)]:
            self.candidate_list.heading(c,text=label);self.candidate_list.column(c,width=width,stretch=c=='engine')
        self.candidate_list.column('#0',width=150);self.candidate_list.pack(fill='x',pady=5)
        self.candidate_list.bind('<<TreeviewSelect>>',self._candidate_selected)
        actions=ttk.Frame(compare);actions.pack(fill='x')
        self._button(actions,'Audition candidate',self._audition)
        self._button(actions,'Accept candidate',self._accept)
        self._button(actions,'Reject all',self._reject)
        self._button(actions,'Edit candidate',self._edit_candidate)
        self.candidate_info=tk.StringVar(value='Suggestions remain separate until you accept one.')
        ttk.Label(compare,textvariable=self.candidate_info,wraplength=1000).pack(anchor='w',pady=3)
        self.canvas=tk.Canvas(compose,height=180,bg='#151b25',highlightthickness=0)
        scroll=ttk.Scrollbar(compose,orient='horizontal',command=self.canvas.xview);scroll.pack(side='bottom',fill='x')
        self.canvas.configure(xscrollcommand=scroll.set)
        self.canvas.bind('<Configure>',lambda _:self._draw())
        ttk.Label(compose,text='Blue: human notes   Purple: accepted AI   Gold: candidate preview').pack(side='bottom',anchor='w')
        self.canvas.pack(fill='both',expand=True,pady=(6,0))

        actions=ttk.Frame(reference);actions.pack(fill='x')
        self._button(actions,'Import MP3 / WAV / MIDI',self._import_reference)
        self._button(actions,'Preview original',self._preview_reference)
        self._button(actions,'Stop preview',self.preview.stop,False)
        self._button(actions,'Remove guidance',self._remove_reference)
        self.reference_label=tk.StringVar(value='No reference loaded. Audio remains on this computer.')
        ttk.Label(reference,textvariable=self.reference_label,font=('Segoe UI',12,'bold'),wraplength=950).pack(anchor='w',pady=10)
        influence=ttk.Frame(reference);influence.pack(fill='x')
        ttk.Label(influence,text='Reference influence: weak').pack(side='left')
        slider=ttk.Scale(influence,variable=self.influence,from_=0,to=1,length=320);slider.pack(side='left',padx=8)
        self._controls.append(slider);ttk.Label(influence,text='strong').pack(side='left')
        ttk.Label(reference,text='Stronger influence follows the selected broad features more closely. Zero ignores all reference guidance.',
                  wraplength=950).pack(anchor='w',pady=6)
        features=ttk.Frame(reference);features.pack(fill='x')
        for name,label in [('harmony','Pitch classes / harmonic colour'),('rhythm','Onset density'),('energy','Energy / dynamics'),('contour','Broad pitch direction')]:
            w=ttk.Checkbutton(features,text=label,variable=self.follow[name]);w.pack(side='left',padx=5);self._controls.append(w)
        w=ttk.Checkbutton(reference,text='Use estimated reference tempo (when available)',variable=self.follow_tempo)
        w.pack(anchor='w',pady=10);self._controls.append(w)
        self.reference_text=tk.Text(reference,wrap='word',height=18,font=('Segoe UI',10),bg=PANEL,fg=TEXT,insertbackground=TEXT,relief='flat')
        self.reference_text.pack(fill='both',expand=True);self.reference_text.configure(state='disabled')
        ttk.Label(reference,text='This is feature-guided symbolic composition, not transcription or a newly trained audio-conditioned model.',
                  wraplength=950).pack(anchor='w',pady=6)
        top=ttk.Frame(editor);top.pack(fill='x')
        self.editor_box=ttk.Combobox(top,textvariable=self.edit_target,values=('Composition','Candidate'),state='readonly',width=18)
        self.editor_box.pack(side='left');self.editor_box.bind('<<ComboboxSelected>>',lambda _:self._refresh_notes())
        ttk.Label(top,text='Select notes to edit/delete. Regeneration replaces all notes starting inside the selected time range.',
                  wraplength=800).pack(side='left',padx=8)
        columns=('pitch','start','duration','velocity','channel','program','source')
        self.notes_tree=ttk.Treeview(editor,columns=columns,show='headings',height=13,selectmode='extended')
        for col in columns:self.notes_tree.heading(col,text=col.title());self.notes_tree.column(col,width=100)
        self.notes_tree.pack(fill='both',expand=True,pady=10)
        self.notes_tree.bind('<<TreeviewSelect>>',self._note_selected)
        inputs=ttk.Frame(editor);inputs.pack(fill='x')
        self.note_vars={}
        for i,(name,value) in enumerate([('pitch','60'),('start','0'),('duration','0.25'),('velocity','85'),('channel','0'),('program','0')]):
            ttk.Label(inputs,text=name.title()).grid(row=0,column=i,sticky='w')
            var=tk.StringVar(value=value);self.note_vars[name]=var
            ttk.Entry(inputs,textvariable=var,width=12).grid(row=1,column=i,padx=(0,8))
        actions=ttk.Frame(editor);actions.pack(fill='x')
        self._button(actions,'Add human note',lambda:self._change_note(add=True))
        self._button(actions,'Apply edit',self._change_note)
        self._button(actions,'Delete selected',self._delete_notes)
        self._button(actions,'Audition selected',self._play_selected)
        ttk.Label(editor,text='Times are seconds; pitch/velocity 0–127 (velocity > 0), channels 0–15. Channel 9 is percussion.',
                  wraplength=950).pack(anchor='w',pady=8)
        self._draw()

    def _combo(self,parent,label,var,values,column,width):
        cell=ttk.Frame(parent);cell.grid(row=0,column=column,padx=(0,20),sticky='w')
        ttk.Label(cell,text=label).pack(anchor='w')
        box=ttk.Combobox(cell,textvariable=var,values=values,state='readonly',width=width)
        box.pack();self._controls.append(box);return box

    def _set_busy(self,value):
        self._busy=value
        for widget in self._controls:
            widget.configure(state='disabled' if value else ('readonly' if isinstance(widget,ttk.Combobox) else 'normal'))

    def _submit(self,label,worker,done):
        if self._busy:return
        self._set_busy(True);self.status.set(label)
        def run():
            try:self._queue.put(('done',done,worker()))
            except Exception as exc:self._queue.put(('error',str(exc)))
        threading.Thread(target=run,daemon=True).start()

    def _poll(self):
        if self._closed:return
        try:
            while True:
                item=self._queue.get_nowait()
                if item[0]=='done':
                    self._set_busy(False)
                    try:item[1](item[2])
                    except Exception as exc:self._error(exc)
                elif item[0]=='error':
                    self._set_busy(False);self._error(item[1])
                elif item[0]=='status':
                    if not self._busy:self.status.set(item[1])
                elif item[0]=='progress':self.status.set(item[1])
        except queue.Empty:pass
        self.root.after(60,self._poll)

    def _error(self,error):self.status.set(f'Unable to complete action: {error}')

    def _load_model(self):
        choice=self.model_choice.get()
        def work():
            if not CHECKPOINTS[choice]:return None,'Symbolic composer selected'
            try:
                from ml.inference import LSTMGenerator
                generator=LSTMGenerator(APP_DIR/'models'/CHECKPOINTS[choice])
                if not generator.loaded:return None,generator.load_error or 'Checkpoint unavailable'
                return generator,f'{choice} · {generator.device_name}'
            except (ImportError,OSError,RuntimeError) as exc:return None,f'Symbolic fallback · {exc}'
        def done(result):
            self.ml_generator,self._model_description=result;self.model_status.set(self._model_description)
            self.status.set('Ready — local generation')
        self._submit('Loading selected model…',work,done)

    def _note_key(self,event):
        if self._busy:return
        focus=self.root.focus_get()
        if isinstance(focus,(tk.Entry,ttk.Entry,tk.Text,ttk.Combobox,ttk.Spinbox,tk.Spinbox)):return
        note=NOTE_KEYS.get(event.char.lower())
        if note:self._choose_note(note);return 'break'

    def _choose_note(self,note):
        if self._busy:return
        self.target.set(note)
        try:
            settings=self._settings();settings.objective='Continue melody';settings.candidates=1
            composition=copy.deepcopy(self.composition);reference=copy.deepcopy(self.reference)
            generator=GuidedComposer(self.ml_generator);self._stop()
            def done(candidates):
                phrase=candidates[0].events
                self.document.set_events(self.composition.events+phrase)
                self.document.settings={'generation':asdict(settings),
                                        'reference_summary':reference.to_dict() if reference else None}
                self._reject();self._refresh_notes();self._draw()
                self.tabs.select(self.compose_tab)
                self.status.set('Playing new continuation')
                self.player.play(phrase)
            self._submit('AI generating…',lambda:generator.generate(composition,settings,reference),done)
        except Exception as exc:self._error(exc)

    def _settings(self):
        influence=float(self.influence.get());bpm=float(self.bpm.get())
        if self.follow_tempo.get() and self.reference and self.reference.tempo and influence>0:
            bpm=(1-influence)*bpm+influence*self.reference.tempo
        settings=GenerationSettings(self.objective.get(),self.target.get(),bpm,float(self.seconds.get()),
                                    influence,tuple(k for k,v in self.follow.items() if v.get()),
                                    float(self.temperature.get()),int(self.seed.get()),int(self.count.get()))
        settings.validate();return settings

    def _generate(self,replace=False):
        if self._busy:return
        try:
            settings=self._settings();composition=copy.deepcopy(self.composition)
            selected_range=None
            if replace:
                if self.edit_target.get()!='Composition':raise ValueError('Select a range in the Composition note editor first')
                selected=self._selected_notes()
                if not selected:raise ValueError('Select composition notes in the Note editor to regenerate their time range')
                a=min(e.start for e in selected);b=max(e.start+e.duration for e in selected)
                if b-a<.25:raise ValueError('Select a time range of at least 0.25 seconds')
                selected_range=(a,b);settings.seconds=b-a
                composition.events=[e for e in composition.events if e.start<a]
            # Complementary parts overlay the composition from its beginning.
            start=(selected_range[0] if selected_range else (0. if settings.objective=='Complementary part' and composition.events else None))
            include_cue=start is None
            reference=copy.deepcopy(self.reference);generator=GuidedComposer(self.ml_generator)
            self._stop()
            def done(candidates):
                self.candidates=candidates;self.replace_range=selected_range
                self._generation_settings=asdict(settings);self._refresh_candidates()
                self.status.set('Candidates ready. Audition, edit, then accept one.')
                self.tabs.select(self.compare_tab)
            self._submit('Generating musical candidates…',
                         lambda:generator.generate(composition,settings,reference,start,include_cue),done)
        except Exception as exc:self._error(exc)

    def _current_candidate(self):
        selection=self.candidate_list.selection()
        return self.candidates[int(selection[0])] if selection else None

    def _refresh_candidates(self):
        selected=self.candidate_list.selection()
        selection=selected[0] if selected and int(selected[0])<len(self.candidates) else '0'
        self.candidate_list.delete(*self.candidate_list.get_children())
        for i,c in enumerate(self.candidates):
            m=c.metrics
            self.candidate_list.insert('', 'end',iid=str(i),text=f'{i+1} · seed {c.seed}',
                                       values=(c.engine,m.get('notes',len(c.events)),f"{m.get('span_seconds',0):.2f} s",m.get('unique_pitches','')))
        if self.candidates:self.candidate_list.selection_set(selection)
        self._draw();self._refresh_notes()

    def _candidate_selected(self,_=None):
        c=self._current_candidate()
        if c:
            self.candidate_info.set(f'{c.engine}. Suggestions are uncommitted; edits can change their measured properties.')
        self._draw();self._refresh_notes()

    def _audition(self):
        c=self._current_candidate()
        if c:self.preview.stop();self.player.play(c.events)

    def _accept(self):
        c=self._current_candidate()
        if not c:return
        old=self.composition.events
        if self.replace_range:
            a,b=self.replace_range;old=[e for e in old if not a<=e.start<b]
        try:
            self.document.set_events(old+c.events)
            self.document.settings={'generation':getattr(self,'_generation_settings',{}),
                                    'reference_summary':self.reference.to_dict() if self.reference else None}
            self._reject();self._refresh_notes();self._draw();self.status.set('Candidate accepted. Save the session to keep your work.')
        except Exception as exc:self._error(exc)

    def _reject(self):
        self._stop();self.candidates=[];self.replace_range=None;self._refresh_candidates()
        self.candidate_info.set('Suggestions remain separate until you accept one.')

    def _edit_candidate(self):
        if self._current_candidate():
            self.edit_target.set('Candidate');self._refresh_notes();self.tabs.select(self.editor_tab)

    def _editor_events(self):
        c=self._current_candidate()
        return c.events if self.edit_target.get()=='Candidate' and c else (self.composition.events if self.edit_target.get()=='Composition' else [])

    def _refresh_notes(self):
        self.notes_tree.delete(*self.notes_tree.get_children())
        for i,e in enumerate(self._editor_events()):
            self.notes_tree.insert('', 'end',iid=str(i),values=(e.pitch,f'{e.start:.4f}',f'{e.duration:.4f}',e.velocity,e.channel,e.program,e.source))

    def _selected_notes(self):
        events=self._editor_events()
        return [events[int(i)] for i in self.notes_tree.selection() if int(i)<len(events)]

    def _note_selected(self,_=None):
        notes=self._selected_notes()
        if notes:
            for name,var in self.note_vars.items():var.set(str(getattr(notes[0],name)))

    def _commit_editor(self,events):
        validate_events(events)
        if self.edit_target.get()=='Candidate':
            c=self._current_candidate()
            if c is None:raise ValueError('Choose a candidate first')
            c.events=sorted(events,key=lambda e:(e.start,e.pitch))
            from guided_generation import musical_metrics
            boundary=min((e.start for e in events if e.source=='ai'),default=0)
            c.metrics=musical_metrics(events,self._settings().bpm,self.reference,boundary)
        else:
            self.document.set_events(events)
            self.candidates=[];self.replace_range=None
        self._refresh_candidates();self._refresh_notes();self._draw()

    def _change_note(self,add=False):
        if self._busy:return
        try:
            events=copy.deepcopy(self._editor_events())
            kw={name:(float(var.get()) if name in ('start','duration') else int(var.get())) for name,var in self.note_vars.items()}
            note=NoteEvent(**kw,source='human',label=str(kw['pitch']))
            if add:events.append(note)
            else:
                selected=self.notes_tree.selection()
                if len(selected)!=1:raise ValueError('Select exactly one note to edit')
                i=int(selected[0]);note.source=events[i].source;events[i]=note
            self._commit_editor(events);self.status.set('Note changes applied')
        except Exception as exc:self._error(exc)

    def _delete_notes(self):
        if self._busy:return
        indices={int(i) for i in self.notes_tree.selection()}
        if indices:
            try:self._commit_editor([e for i,e in enumerate(self._editor_events()) if i not in indices])
            except Exception as exc:self._error(exc)

    def _play_selected(self):
        notes=self._selected_notes()
        if notes:self.player.play(notes)

    def _draw(self):
        c=self.canvas;c.delete('all');events=self.composition.events
        self.sequence.configure(text='  '.join(f"{'HUMAN' if e.source=='human' else 'AI'}: {e.label or e.pitch}" for e in events[-16:]) if events else 'No notes yet')
        candidate=self._current_candidate();proposed=candidate.events if candidate else []
        end=max((e.start+e.duration for e in events+proposed),default=1.)
        width=max(c.winfo_width(),800);height=max(c.winfo_height(),160);scale=max(45,(width-30)/max(end,1))
        for rows,is_candidate in ((events,False),(proposed,True)):
            for e in rows:
                x=12+e.start*scale;y=height-22-(e.pitch-32)*(height-35)/64
                colour='#f0c75e' if is_candidate else ('#55c2ff' if e.source=='human' else '#b89cff')
                c.create_rectangle(x,y,x+max(2,e.duration*scale),y+9,fill=colour,outline='')
        if not events and not proposed:c.create_text(width/2,height/2,text='Choose a note or import MIDI to begin',fill='#bfcadb')
        c.configure(scrollregion=(0,0,max(width,end*scale+30),height))

    def _import_reference(self):
        path=filedialog.askopenfilename(filetypes=[('Audio or MIDI','*.mp3 *.wav *.mid *.midi')])
        if not path:return
        def done(features):
            self.reference=features;self.reference_path=Path(path)
            self.reference_label.set(f'{features.filename} · {features.duration:.1f} seconds · analyzed {features.analyzed_seconds:.1f} seconds')
            tempo=f'{features.tempo:.1f} BPM (confidence {features.tempo_confidence:.2f})' if features.tempo else 'uncertain'
            text=(f'Measured: duration {features.duration:.2f} s; detected onset density (estimated) {features.onset_density:.2f}/s; '
                  f'RMS/velocity proxy {features.rms:.3f}; dynamic ratio {features.dynamic_range:.2f}.\n\n'
                  f'Estimated pulse: {tempo}\nEstimated key: {features.key or "uncertain"} (confidence {features.key_confidence:.2f})\n'
                  f'Broad spectral direction: {features.contour:+.2f}\n'
                  f'Confident-enough chord-template windows: {len(features.chords)} (diagnostic only; not a copied progression).\n\n'
                  +'\n'.join(features.warnings))
            self.reference_text.configure(state='normal');self.reference_text.delete('1.0','end')
            self.reference_text.insert('1.0',text);self.reference_text.configure(state='disabled')
            self.status.set('Reference ready. Choose features and influence, then generate candidates.')
        self._submit('Analyzing reference locally…',lambda:self.analyzer.analyze(path,lambda msg:self._queue.put(('progress',msg))),done)

    def _preview_reference(self):
        try:
            if not self.reference_path:raise ValueError('Import a reference first')
            self.player.stop()
            if self.reference_path.suffix.lower() in ('.mid','.midi'):self.player.play(read_midi(self.reference_path))
            else:self.preview.play(self.reference_path)
        except Exception as exc:self._error(exc)

    def _remove_reference(self):
        self.preview.stop();self.reference=None;self.reference_path=None
        self.reference_label.set('No reference loaded');self.reference_text.configure(state='normal')
        self.reference_text.delete('1.0','end');self.reference_text.configure(state='disabled')

    def _confirm_discard(self):
        return not self.document.dirty or messagebox.askyesno('Unsaved composition','Discard unsaved changes?')

    def _open(self):
        if not self._confirm_discard():return
        path=filedialog.askopenfilename(filetypes=[('Composition session','*.json')])
        if path:
            try:
                self.document.load(path);self._session_path=path;self._reject()
                generation=self.document.settings.get('generation',{})
                for key,var in [('bpm',self.bpm),('seconds',self.seconds),('temperature',self.temperature),('influence',self.influence),
                                ('seed',self.seed),('candidates',self.count),('objective',self.objective),('target',self.target)]:
                    if key in generation:var.set(generation[key])
                if 'features' in generation:
                    for key,var in self.follow.items():var.set(key in generation['features'])
                self._remove_reference()  # Never silently re-open a path stored in an untrusted session.
                self._refresh_notes();self._draw();self.status.set('Session opened. Reimport reference audio to enable fresh guidance.')
            except Exception as exc:self._error(exc)

    def _save(self):
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile=Path(self._session_path).name if self._session_path else 'composition.json',
                                        filetypes=[('Composition session','*.json')])
        if path:
            try:
                self.document.settings['generation']=asdict(self._settings())
                self.document.save(path);self._session_path=path;self.status.set(f'Session saved: {path}')
            except Exception as exc:self._error(exc)

    def _import_midi(self):
        path=filedialog.askopenfilename(filetypes=[('MIDI','*.mid *.midi')])
        if not path:return
        def done(events):
            offset=self.composition.end_time
            for e in events:e.start+=offset
            self.document.set_events(self.composition.events+events);self._reject();self._refresh_notes();self._draw()
            self.status.set('MIDI appended; original timing and polyphony retained')
        self._submit('Importing MIDI…',lambda:read_midi(path),done)

    def _export_midi(self):
        if not self.composition.events:self.status.set('Nothing to export');return
        path=filedialog.asksaveasfilename(defaultextension='.mid',filetypes=[('MIDI','*.mid')])
        if path:
            try:midi_file(self.composition.events,float(self.bpm.get())).save(path);self.status.set(f'MIDI exported: {path}')
            except Exception as exc:self._error(exc)

    def _undo(self):
        self._stop();self.document.undo();self._reject();self._refresh_notes();self._draw()

    def _clear(self):
        if self.composition.events:
            self.document.set_events([]);self._reject();self._refresh_notes();self._draw()

    def _pause(self):
        if self.player.is_paused:self.player.resume()
        else:self.player.pause()

    def _stop(self):self.player.stop();self.preview.stop()

    def _on_close(self):
        if not self._confirm_discard():return
        self._closed=True;self.preview.stop();self.player.close();self.root.destroy()

    def run(self):self.root.mainloop()
