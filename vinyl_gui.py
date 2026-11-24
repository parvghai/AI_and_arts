"""Monochrome vinyl studio. The existing MusicApp owns generation and persistence."""
import math
import copy
import os
import random
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from gui import MusicApp
from config import LOG_DIR
from zen_widgets import CollapsibleSettings, landscape, vinyl_angle, load_preferences, save_preferences
from vinyl_view import event_marks, seek_events, restore_composition

BG='#11100F'; SURFACE='#191817'; SECOND='#211F1D'; ELEVATED='#2B2825'
BORDER='#393531'; TEXT='#F0EAE0'; SUB='#AAA197'; MUTED='#77716B'
HUMAN='#F5F1E9'; AI='#85817B'
FONT='Segoe UI'


def label(parent, text='', size=10, color=SUB, **kw):
    return tk.Label(parent, text=text, bg=parent.cget('bg'), fg=color,
                    font=(FONT,size), **kw)


def clock(value):
    return f'{int(value)//60:02d}:{value%60:04.1f}'


class VinylStudio(MusicApp):
    def __init__(self, adapter_hint=None):
        self._selected='C'; self._view='Circular'; self._last_frame=None
        self._preferences_path=LOG_DIR/'vinyl_preferences.json'
        self._preferences=load_preferences(self._preferences_path)
        self._ambient_started=time.monotonic(); self._ambient_phase=0.; self._ambient_counter=0
        super().__init__(adapter_hint)
        self.root.title('VINYL / Human × AI music studio')
        self.root.geometry('1440x920'); self.root.minsize(1100,760)
        self._refresh(); self._tick()

    @staticmethod
    def _button(parent, label, command):
        return tk.Button(parent,text=label,command=command,bg=SECOND,fg=TEXT,
            activebackground=ELEVATED,activeforeground=HUMAN,disabledforeground=MUTED,
            relief='flat',borderwidth=0,padx=12,pady=9,font=(FONT,9),cursor='hand2')

    def action(self,parent,text,fn,busy=True,**pack):
        b=self._button(parent,text,fn); b.pack(**pack)
        if busy: self._busy_controls.append(b)
        return b

    def _build(self):
        self.root.configure(bg=BG)
        self.reduced_motion=tk.BooleanVar(self.root,value=self._preferences.get('reduced_motion',False))
        style=ttk.Style(); style.theme_use('clam')
        style.configure('TCombobox',fieldbackground=SECOND,background=ELEVATED,
            foreground=TEXT,arrowcolor=TEXT,bordercolor=BORDER)
        style.map('TCombobox',fieldbackground=[('readonly',SECOND)],foreground=[('readonly',TEXT)])
        # Persistent transport is packed first to reserve its space at any supported size.
        bottom=tk.Frame(self.root,bg=SURFACE,padx=20,pady=12,highlightbackground=BORDER,highlightthickness=1)
        bottom.pack(side='bottom',fill='x')
        label(bottom,'◉',24,TEXT).pack(side='left',padx=(0,14))
        title=tk.Frame(bottom,bg=SURFACE); title.pack(side='left',padx=(0,24))
        label(title,'CURRENT COMPOSITION',9,TEXT).pack(anchor='w')
        label(title,textvariable=self.status,size=8,wraplength=230,anchor='w').pack(anchor='w')
        self.action(bottom,'▶ / Ⅱ',self._toggle_play,side='left',padx=3)
        self.action(bottom,'■',self._stop,side='left',padx=3)
        self.time_label=label(bottom,'00:00.0 / 00:00.0',9); self.time_label.pack(side='left',padx=12)
        self.progress=tk.Scale(bottom,from_=0,to=1000,showvalue=False,orient='horizontal',
            bg=SURFACE,troughcolor=BORDER,activebackground=TEXT,highlightthickness=0,
            sliderrelief='flat',sliderlength=10,bd=0)
        self.progress.pack(side='left',fill='x',expand=True,padx=10)
        self.progress.bind('<ButtonPress-1>',lambda e:setattr(self,'_scrubbing',True))
        self.progress.bind('<ButtonRelease-1>',self._seek)
        self._scrubbing=False
        label(bottom,textvariable=self.bpm,size=10).pack(side='left',padx=(12,0))
        label(bottom,'BPM',8).pack(side='left',padx=(3,12))
        self.action(bottom,'MIDI ↗',self._export_midi,side='left',padx=3)
        self.action(bottom,'LOG ↗',self._export_log,side='left',padx=3)
        body=tk.Frame(self.root,bg=BG); body.pack(fill='both',expand=True)
        sidebar=tk.Frame(body,bg=SURFACE,width=170,padx=14,pady=24)
        sidebar.pack(side='left',fill='y'); sidebar.pack_propagate(False)
        label(sidebar,'V I N Y L',20,TEXT).pack(anchor='w')
        label(sidebar,'HUMAN × MACHINE',8).pack(anchor='w',pady=(4,30))
        for text,fn in [('◉  Studio',lambda:self.record.focus_set()),('Session logs ↗',self._logs),
                        ('Restore session',self._restore),('Export MIDI',self._export_midi),
                        ('Export interaction log',self._export_log)]:
            self.action(sidebar,text,fn,fill='x',pady=3)
        label(sidebar,'OFFLINE / LOCAL',8,MUTED).pack(side='bottom',anchor='w',pady=8)
        self.model_label=label(sidebar,'',8,wraplength=140,justify='left')
        self.model_label.pack(side='bottom',anchor='w',pady=12)
        self._refresh_model_label()
        tk.Checkbutton(sidebar,text='Reduced motion',variable=self.reduced_motion,
            command=self._motion_changed,bg=SURFACE,fg=SUB,selectcolor=SECOND,
            activebackground=SURFACE,activeforeground=TEXT,font=(FONT,8),bd=0).pack(anchor='w',pady=10)
        self.sidebar_landscape=tk.Canvas(sidebar,bg=SURFACE,highlightthickness=0,height=150)
        self.sidebar_landscape.pack(side='bottom',fill='x',pady=(12,10))
        self.sidebar_landscape.bind('<Configure>',lambda e:self._paint_landscapes())
        create=tk.Frame(body,bg=SURFACE,width=265,padx=18,pady=24,
                        highlightbackground=BORDER,highlightthickness=1)
        create.pack(side='left',fill='y'); create.pack_propagate(False)
        label(create,'01 / CREATE',9,MUTED).pack(anchor='w')
        label(create,'Your next phrase.',18,TEXT).pack(anchor='w',pady=(10,5))
        label(create,'Choose a note. Let the model answer.',9,wraplength=225,justify='left').pack(anchor='w',pady=(0,6))
        keys=tk.Frame(create,bg=SURFACE); keys.pack(fill='x')
        for i,(n,sol) in enumerate(zip('CDEFGAB?',('Do','Re','Mi','Fa','Sol','La','Si','Random'))):
            b=self._button(keys,f'{n}  /  {sol}',lambda n=n:self._choose_note(random.choice('CDEFGAB') if n=='?' else n))
            b.configure(pady=5)
            b.grid(row=i//2,column=i%2,sticky='ew',padx=2,pady=2)
            self._note_buttons.append(b)
        keys.columnconfigure((0,1),weight=1)
        label(create,'C D E F G A B · H = B',8,MUTED).pack(anchor='w',pady=(6,8))
        self.generation_settings=CollapsibleSettings(create,
            expanded=self._preferences.get('settings_expanded',True),on_change=self._save_preferences,
            reduced_motion=lambda:self.reduced_motion.get())
        self.generation_settings.pack(fill='x')
        settings=self.generation_settings.content
        label(settings,'MODEL',8).pack(anchor='w')
        self.model_box=ttk.Combobox(settings,textvariable=self.model_choice,
            values=['LSTM 3x512 (baseline)','LSTM 4x768 (experimental)'],state='readonly')
        self.model_box.pack(fill='x',pady=(5,12)); self.model_box.bind('<<ComboboxSelected>>',self._select_model)
        label(settings,'CONTINUATION',8).pack(anchor='w')
        self.duration_box=ttk.Combobox(settings,textvariable=self.duration,
            values=['0.5 seconds','1 second','2 seconds'],state='readonly')
        self.duration_box.pack(fill='x',pady=(5,10))
        for title,var,lo,hi,res in [('TEMPO / BPM',self.bpm,60,160,1),('TEMPERATURE',self.temperature,.2,1.5,.05)]:
            label(settings,title,8).pack(anchor='w',pady=(6,0))
            scale=tk.Scale(settings,from_=lo,to=hi,resolution=res,variable=var,orient='horizontal',
                bg=SURFACE,fg=TEXT,troughcolor=BORDER,activebackground=SUB,
                highlightthickness=0,bd=0,sliderrelief='flat')
            scale.pack(fill='x'); self._busy_controls.append(scale)
        lr=tk.Frame(settings,bg=SURFACE); lr.pack(fill='x',pady=(12,0))
        self._build_lora_controls(lr)
        self.lora.toggle.configure(bg=SURFACE,fg=TEXT,activebackground=SURFACE,selectcolor=SECOND)
        label(settings,'Adapters are optional. Starts OFF.',8,MUTED).pack(anchor='w',pady=5)
        self._busy_controls.extend([self.model_box,self.duration_box])
        work=tk.Frame(body,bg=BG,padx=24,pady=22); work.pack(side='left',fill='both',expand=True)
        top=tk.Frame(work,bg=BG); top.pack(fill='x')
        label(top,'02 / THE LISTENING ROOM',9,MUTED).pack(side='left')
        label(top,'STEREO  /  静',9,SUB).pack(side='right')
        label(work,'Untitled co-creation',24,TEXT).pack(anchor='w',pady=(10,4))
        self.info=label(work,'',9); self.info.pack(anchor='w')
        self.visual=tk.Canvas(work,bg=BG,highlightthickness=0,height=180)
        self.visual.pack(fill='both',expand=True,pady=(8,0)); self.record=self.visual
        self.visual.bind('<Configure>',lambda e:self._draw_timeline())
        self.visual.bind('<Button-1>',self._visual_click)
        views=tk.Frame(work,bg=BG); views.pack(fill='x',pady=(0,8))
        label(views,'● Human',9,HUMAN).pack(side='left')
        label(views,'   ● AI',9,AI).pack(side='left')
        for name in ('Circular','Timeline'):
            self.action(views,name,lambda name=name:self._switch_view(name),busy=False,side='right',padx=3)
        self.details=label(work,'',8,MUTED); self.details.pack(anchor='w')
        toolbar=tk.Frame(work,bg=BG); toolbar.pack(fill='x',pady=10)
        for name,fn in [('Play',self._play),('Pause / Resume',self._pause),('Stop',self._stop),
                        ('Regenerate',self._regenerate),('Undo',self._undo),('Clear',self._clear),('Export MIDI',self._export_midi)]:
            self.action(toolbar,name,fn,side='left',padx=(0,4))
        label(work,'03 / PHRASE HISTORY — select to replay',9,MUTED).pack(anchor='w',pady=(5,8))
        self.history=tk.Canvas(work,bg=BG,height=84,highlightthickness=0)
        self.history.pack(fill='x')
        self.history.bind('<Button-1>',self._replay_click)
        scroll=ttk.Scrollbar(work,orient='horizontal',command=self.history.xview)
        scroll.pack(fill='x'); self.history.configure(xscrollcommand=scroll.set)
        self.bottom_landscape=tk.Canvas(work,bg=BG,height=64,highlightthickness=0)
        self.bottom_landscape.pack(fill='x',pady=(5,0))
        self.bottom_landscape.bind('<Configure>',lambda e:self._paint_landscapes())
        self.sequence=label(work,'',8,wraplength=800,anchor='w')
        self.sequence.pack(fill='x',pady=(8,0))

    def _save_preferences(self):
        try:
            save_preferences(self._preferences_path,self.generation_settings.expanded,self.reduced_motion.get())
        except OSError:
            self.status.set('Display preference could not be saved; current settings are retained.')

    def _motion_changed(self):
        self._save_preferences()
        self._paint_landscapes()
        self._draw_timeline()

    def _paint_landscapes(self):
        phase=0 if self.reduced_motion.get() else self._ambient_phase
        if hasattr(self,'sidebar_landscape'):
            c=self.sidebar_landscape
            landscape(c,c.winfo_width(),c.winfo_height(),phase,SURFACE)
        if hasattr(self,'bottom_landscape'):
            c=self.bottom_landscape
            landscape(c,c.winfo_width(),c.winfo_height(),phase,BG)

    def _build_lora_controls(self, parent):
        from lora_controls import LoRAControls
        from config import APP_DIR
        class MonochromeLoRA(LoRAControls):
            def show(control):
                super().show()
                if control.window is None: return
                def theme(widget):
                    for option, value in [('background',SURFACE),('foreground',TEXT),
                            ('activebackground',ELEVATED),('activeforeground',TEXT),('selectcolor',SECOND)]:
                        try: widget.configure(**{option:value})
                        except tk.TclError: pass
                    for child in widget.winfo_children(): theme(child)
                theme(control.window)
        self.lora=MonochromeLoRA(self,parent,APP_DIR/'runs'/'lora',self._adapter_hint)

    def _choose_note(self,note):
        self._selected=note
        super()._choose_note(note)

    def _switch_view(self,name):
        self._view=name; self._draw_timeline()

    def _refresh(self):
        if not hasattr(self,'info'): return
        events=self.composition.events
        self.info.config(text=f'{len(events)} notes  /  {len(self.composition.interactions)} phrases  /  {self.bpm.get()} BPM  /  {self.model_choice.get()}')
        self.sequence.config(text=' · '.join(f'{i.human_note} → {" ".join(i.ai_notes)}' for i in self.composition.interactions[-3:]) or 'Select a note to make the first mark.')
        self._history(); self._draw_timeline()

    def _draw_timeline(self):
        if not hasattr(self,'visual'): return
        c=self.visual; c.delete('all')
        w=max(c.winfo_width(),600); h=max(c.winfo_height(),100)
        total=max(self.composition.end_time,.001); pos=self.player.position
        r=min(max(35,(h-70)/2),w*.245); cx=w*.28; cy=h*.45
        angle=0 if self.reduced_motion.get() else vinyl_angle(pos)
        for rad,col in [(r+13,SURFACE),(r+9,BORDER),(r+6,'#0A0908'),(r,SECOND)]:
            c.create_oval(cx-rad,cy-rad,cx+rad,cy+rad,fill=col,outline=col)
        for i in range(14,95,2):
            rr=r*i/100
            c.create_oval(cx-rr,cy-rr,cx+rr,cy+rr,outline='#302D29' if i%4 else '#151310')
        for start in [30,210]:
            c.create_arc(cx-r*.96,cy-r*.96,cx+r*.96,cy+r*.96,start=start-angle,extent=32,style='arc',outline='#4B453D',width=2)
        for e,a,span,p in event_marks(self.composition.events,total):
            rr=r*(.40+.50*p)
            c.create_arc(cx-rr,cy-rr,cx+rr,cy+rr,start=90-a-angle,extent=-min(span,359.9),
                style='arc',outline=HUMAN if e.source=='human' else AI,width=1+3*e.velocity/127)
        c.create_oval(cx-r*.30,cy-r*.30,cx+r*.30,cy+r*.30,fill=SURFACE,outline=BORDER)
        c.create_text(cx,cy-14,text='V I N Y L',fill=TEXT,font=(FONT,11,'bold'))
        c.create_text(cx,cy+6,text=f'SIDE A / {len(self.composition.events):03d}',fill=SUB,font=(FONT,8))
        c.create_oval(cx-3,cy+23,cx+3,cy+29,fill=TEXT,outline='')
        # Mechanical tonearm: engaged on playing/paused transport, parked after stop/end.
        ax=cx+r+14; ay=cy-r+10
        engaged=self.player._playing
        # The arm tracks gently inward with actual elapsed composition time.
        tx=cx+r*(.70-.14*min(1,pos/total)) if engaged else ax+13; ty=cy+r*.40
        c.create_line(ax,ay,ax-8,ay+35,tx,ty,fill=MUTED,width=8,joinstyle='round')
        c.create_line(ax-1,ay,ax-9,ay+35,tx-1,ty,fill=SUB,width=2)
        c.create_oval(ax-10,ay-10,ax+10,ay+10,fill=ELEVATED,outline=SUB)
        c.create_rectangle(tx-5,ty-4,tx+6,ty+18,fill=TEXT,outline=BORDER)
        c.create_text(cx,cy+r+29,text='● PLAYING' if self.player.is_playing else ('Ⅱ PAUSED' if self.player.is_paused else '● IDLE'),fill=TEXT if self.player.is_playing else MUTED,font=(FONT,8))
        sx=w*.77; sr=min(r*.82,w*.20); self._center=(sx,cy,sr*.27)
        marks=event_marks(self.composition.events,total)
        if self._view=='Circular':
            for i in range(5):
                rr=sr*(.40+.14*i)
                c.create_oval(sx-rr,cy-rr,sx+rr,cy+rr,outline=BORDER)
            step=60/max(1,self.bpm.get())/4
            count=max(1,math.ceil(total/step))
            # Bound drawing cost while retaining the true step count in the caption.
            for i in range(0,count,max(1,math.ceil(count/128))):
                a=2*math.pi*i*step/total
                x=sx+sr*math.sin(a); y=cy-sr*math.cos(a)
                c.create_oval(x-1,y-1,x+1,y+1,fill=MUTED,outline='')
            for e,a,span,p in marks:
                rr=sr*(.40+.56*p)
                active=self.player._playing and e.start<=pos<e.start+e.duration
                c.create_arc(sx-rr,cy-rr,sx+rr,cy+rr,start=90-a,extent=-min(span,359.9),style='arc',
                    outline=HUMAN if active or e.source=='human' else AI,width=(2+4*e.velocity/127))
            a=pos/total*2*math.pi
            c.create_line(sx,cy,sx+sr*math.sin(a),cy-sr*math.cos(a),fill=TEXT,width=1)
            c.create_oval(sx-sr*.27,cy-sr*.27,sx+sr*.27,cy+sr*.27,fill=SECOND,outline=BORDER)
            c.create_text(sx,cy,text='Ⅱ' if self.player.is_playing else '▶',fill=TEXT,font=(FONT,18))
            c.create_text(sx,cy+sr+30,text='CIRCULAR / 1⁄16 GRID',fill=SUB,font=(FONT,9))
        else:
            left=sx-sr; right=sx+sr
            for i,title in enumerate(['HUMAN','AI']):
                y=cy-50+i*80
                c.create_text(left,y-18,text=title,anchor='w',fill=SUB,font=(FONT,8))
                c.create_line(left,y+20,right,y+20,fill=BORDER)
            for e,a,span,p in marks:
                x=left+2*sr*e.start/total; y=cy-50+(0 if e.source=='human' else 80)
                c.create_rectangle(x,y,x+max(1,2*sr*e.duration/total),y+14,
                    fill=HUMAN if e.source=='human' else AI,outline='')
            x=left+2*sr*pos/total
            c.create_line(x,cy-80,x,cy+65,fill=TEXT)
            count=max(1,math.ceil(total/(60/max(1,self.bpm.get())/4)))
            c.create_text(sx,cy+sr+30,text='TIMELINE / SECONDS',fill=SUB,font=(FONT,9))
            self._center=None
        pitches=[e.pitch for e in self.composition.events]
        self.details.config(text=f'Pitch {min(pitches,default=60)}–{max(pitches,default=72)} MIDI  /  {count} steps  /  1⁄16 note grid  /  event timing preserved')

    def _history(self):
        c=self.history; c.delete('all')
        self._paint_landscapes()
        for i,item in enumerate(self.composition.interactions):
            x=i*172
            c.create_rectangle(x,0,x+160,84,fill=SURFACE,outline=BORDER)
            c.create_text(x+10,15,text=f'{i+1:02d}  /  HUMAN {item.human_note}    ▶',anchor='w',fill=TEXT,font=(FONT,9))
            origin=min(e.start for e in item.events); end=max(e.start+e.duration for e in item.events)
            for e in item.events:
                px=x+10+140*(e.start-origin)/max(.001,end-origin)
                yy=60-(e.pitch%12)*2
                c.create_line(px,yy,px+max(2,140*e.duration/max(.001,end-origin)),yy,
                    fill=HUMAN if e.source=='human' else AI,width=2)
            c.create_text(x+10,74,text=f'{len(item.events)} notes · {end-origin:.2f}s',anchor='w',fill=MUTED,font=(FONT,8))
        c.configure(scrollregion=(0,0,max(c.winfo_width(),max(1,len(self.composition.interactions))*172),84))

    def _replay_click(self,event):
        index=int(self.history.canvasx(event.x)//172)
        if not self._generating and index<len(self.composition.interactions):
            self.player.play(self.composition.interactions[index].events)

    def _visual_click(self,event):
        if self._center and math.hypot(event.x-self._center[0],event.y-self._center[1])<=self._center[2]:
            self._toggle_play()

    def _toggle_play(self):
        if self.player._playing: self._pause()
        else: self._play()

    def _pause(self):
        if self.player._playing: super()._pause()

    def _stop(self):
        super()._stop()
        self.player._position=0.0
        self._draw_timeline()

    def _clear(self):
        super()._clear(); self.player._position=0.0; self._draw_timeline()

    def _regenerate(self):
        if self._generating or not self.lora.ready: return
        note=self.composition.interactions[-1].human_note if self.composition.interactions else self._selected
        self._regeneration_backup=copy.deepcopy(self.composition)
        if self.composition.interactions:
            self.player.stop(); self.composition.undo(); self._refresh()
        self._choose_note(note)

    def _generation_failed(self,error):
        if getattr(self,'_regeneration_backup',None) is not None:
            self.composition=self._regeneration_backup; self._regeneration_backup=None
            self._refresh()
        super()._generation_failed(error)

    def _finish_generation(self,*args,**kwargs):
        super()._finish_generation(*args,**kwargs)
        self._regeneration_backup=None

    def _seek(self,event=None):
        self._scrubbing=False
        if self._generating: return
        pos=self.composition.end_time*self.progress.get()/1000
        events=seek_events(self.composition.events,pos)
        self.player.stop()
        if events: self.player.play(events)
        else: self.player._position=pos

    def _logs(self):
        folder=LOG_DIR/'sessions'; folder.mkdir(parents=True,exist_ok=True)
        os.startfile(folder)

    def _restore(self):
        filename=filedialog.askopenfilename(title='Restore composition',initialdir=LOG_DIR/'sessions',filetypes=[('Session JSON','*.json')])
        if not filename: return
        try:
            composition=restore_composition(filename)
            self.player.stop(); self.player._position=0
            self.composition=composition
            if hasattr(self,'_autosave_path'): del self._autosave_path
            if composition.interactions: self.bpm.set(composition.interactions[-1].bpm)
            self._refresh(); self.status.set('Session restored; future edits save a new session')
        except (ValueError,KeyError,TypeError,OSError) as exc:
            messagebox.showerror('Cannot restore session',str(exc))

    def _tick(self):
        if self._closed: return
        pos=self.player.position; total=self.composition.end_time
        self.time_label.config(text=f'{clock(pos)} / {clock(total)}')
        if not self._scrubbing: self.progress.set(1000*pos/max(total,.001))
        state=(round(pos,2),self.player._playing,self.player.is_paused,self.bpm.get())
        if state!=self._last_frame:
            self._draw_timeline(); self._last_frame=state
        self._ambient_counter+=1
        if not self.reduced_motion.get() and self._ambient_counter%8==0:
            self._ambient_phase=(time.monotonic()-self._ambient_started)*2*math.pi/90
            self._paint_landscapes()
        self.root.after(50,self._tick)
