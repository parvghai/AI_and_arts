"""Tkinter interface for iterative human–AI composition."""
import json
import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from datetime import datetime, timezone
from time import perf_counter
from ai_generator import RuleBasedGenerator
from composition import (Composition, human_note_duration, sanitize_timeline,
                         validate_timeline)
from config import DEFAULT_BPM, DEFAULT_CONTINUATION, LOG_DIR, NOTES
from midi_player import MidiPlayer
from config import APP_DIR
try:
    from ml.inference import LSTMGenerator
except ImportError:  # PyTorch is an optional training/inference dependency.
    class LSTMGenerator:
        CHECKPOINTS = {"LSTM 3x512 (baseline)": "best_model_3x512.pt",
                       "LSTM 4x768 (experimental)": "best_model_4x768.pt"}
        loaded = False
        config = {}
        temperature = 0.8
        def __init__(self, *args, **kwargs): pass
        def load_checkpoint(self, *args, **kwargs): pass

BG, PANEL, TEXT, HUMAN, AI = "#11151d", "#1b2230", "#edf2f7", "#55c2ff", "#b89cff"
PITCH = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71}
NOTE_KEYS = {note.lower(): note for note in PITCH}
# H is B natural in German note naming; keep B as the button's displayed name.
NOTE_KEYS["h"] = "B"


class MusicApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Music Co-Creator • Music Studio")
        self.root.geometry("980x720")
        self.root.minsize(820, 620)
        self.root.configure(bg=BG)
        self.composition = Composition()
        self.generator = RuleBasedGenerator()
        self.ml_generator = LSTMGenerator(APP_DIR / "models" / "best_model_3x512.pt")
        self.temperature = tk.DoubleVar(value=0.75)
        self.model_choice = tk.StringVar(value="LSTM 3x512 (baseline)")
        if (self.ml_generator.config.get("layers") == 4 and
                self.ml_generator.config.get("hidden_size") == 768):
            self.model_choice.set("LSTM 4x768 (experimental)")
        self.player = MidiPlayer(self._set_status_threadsafe)
        self.status = tk.StringVar(value="Ready")
        self.duration = tk.StringVar(value="1 second")
        self.bpm = tk.IntVar(value=DEFAULT_BPM)
        self._generating = False
        self._closed = False
        self._busy_controls: list[tk.Widget] = []
        self._note_buttons: list[tk.Button] = []
        self._build()
        self.root.bind_all("<KeyPress>", self._handle_note_key)
        self._draw_timeline()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT)
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(22, 12))
        tk.Label(header, text="AI Music Co-Creator", font=("Segoe UI", 24, "bold"), fg=TEXT, bg=BG).pack(side="left")
        right = tk.Frame(header, bg=BG); right.pack(side="right", pady=5)
        tk.Label(right, text="HUMAN  →  AI  →  HUMAN  →  AI", font=("Segoe UI", 10, "bold"),
                 fg=AI, bg=BG).pack(anchor="e")
        self.model_label = tk.Label(right, text=("AI Model: LSTM "
            f"{self.ml_generator.config.get('layers',3)}×{self.ml_generator.config.get('hidden_size',512)} · Model loaded"
            if self.ml_generator.loaded else "AI Model: rule-based · ML model unavailable — using rule-based generator"),
            font=("Segoe UI", 9), fg="#9ba8ba", bg=BG)
        self.model_label.pack(anchor="e", pady=(4,0))
        self._refresh_model_label()
        tk.Label(self.root, text="Choose a note to guide the next musical phrase.",
                 font=("Segoe UI", 11), fg="#9ba8ba", bg=BG).pack(anchor="w", padx=32)
        controls = tk.Frame(self.root, bg=PANEL, padx=18, pady=16)
        controls.pack(fill="x", padx=28, pady=18)
        tk.Label(controls, text="YOUR NOTE", fg=HUMAN, bg=PANEL, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        keys = tk.Frame(controls, bg=PANEL)
        keys.pack(fill="x", pady=(10, 0))
        for note in NOTES:
            button = tk.Button(keys, text=note, command=lambda n=note: self._choose_note(n),
                               font=("Segoe UI", 17, "bold"), fg=TEXT, bg="#293446",
                               activebackground="#354761", activeforeground=TEXT,
                               relief="flat", cursor="hand2", height=2)
            button.pack(side="left", expand=True, fill="x", padx=4)
            self._note_buttons.append(button)
        tk.Label(controls, text="Keyboard: C D E F G A B  ·  H also plays B",
                 fg="#9ba8ba", bg=PANEL, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
        toolbar = tk.Frame(self.root, bg=BG)
        toolbar.pack(fill="x", padx=28)
        for label, callback in (("Play", self._play), ("Pause / Resume", self._pause),
                                ("Stop", self._stop), ("Undo", self._undo), ("Clear", self._clear)):
            button = self._button(toolbar, label, callback)
            button.pack(side="left", padx=4)
            self._busy_controls.append(button)
        settings = tk.Frame(self.root, bg=BG)
        settings.pack(fill="x", padx=30, pady=(15, 7))
        tk.Label(settings, text="MODEL", fg="#9ba8ba", bg=BG, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.model_box = ttk.Combobox(settings, textvariable=self.model_choice,
                                      values=("LSTM 3x512 (baseline)", "LSTM 4x768 (experimental)"),
                                      state="readonly", width=26)
        self.model_box.pack(side="left", padx=(7, 14))
        self.model_box.bind("<<ComboboxSelected>>", self._select_model)
        tk.Label(settings, text="CONTINUATION", fg="#9ba8ba", bg=BG, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.duration_box = ttk.Combobox(settings, textvariable=self.duration,
                                         values=("0.5 seconds", "1 second", "2 seconds"),
                                         state="readonly", width=13)
        self.duration_box.pack(side="left", padx=9)
        tk.Label(settings, text="TEMPO", fg="#9ba8ba", bg=BG, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(18, 5))
        self.tempo_scale = tk.Scale(settings, from_=60, to=160, variable=self.bpm, orient="horizontal", length=170,
                                    bg=BG, fg=TEXT, troughcolor=PANEL, highlightthickness=0)
        self.tempo_scale.pack(side="left")
        self.bpm_label = tk.Label(settings, text="100 BPM", fg=TEXT, bg=BG, width=9)
        self.bpm_label.pack(side="left")
        self.bpm.trace_add("write", lambda *_: self.bpm_label.config(text=f"{self.bpm.get()} BPM"))
        tk.Label(settings, text="TEMP", fg="#9ba8ba", bg=BG, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(9,2))
        self.temperature_scale = tk.Scale(settings, from_=0.2, to=1.5, resolution=0.05, variable=self.temperature, orient="horizontal",
                                          length=110, bg=BG, fg=TEXT, troughcolor=PANEL, highlightthickness=0)
        self.temperature_scale.pack(side="left")
        self._busy_controls.extend((self.model_box, self.duration_box, self.tempo_scale, self.temperature_scale))
        exportbar = tk.Frame(self.root, bg=BG)
        exportbar.pack(fill="x", padx=28, pady=(0, 5))
        self._button(exportbar, "Export MIDI", self._export_midi).pack(side="right", padx=4)
        self._button(exportbar, "Export interaction log", self._export_log).pack(side="right", padx=4)
        statusbar = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        statusbar.pack(fill="x", padx=28, pady=(6, 12))
        tk.Label(statusbar, text="●", fg=AI, bg=PANEL).pack(side="left")
        tk.Label(statusbar, textvariable=self.status, fg=TEXT, bg=PANEL, font=("Segoe UI", 10)).pack(side="left", padx=7)
        tk.Label(self.root, text="COMPOSITION", fg="#9ba8ba", bg=BG, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=30)
        self.sequence = tk.Label(self.root, text="No notes yet", fg=TEXT, bg=BG, font=("Segoe UI", 14, "bold"), anchor="w")
        self.sequence.pack(fill="x", padx=30, pady=(5, 5))
        self.canvas = tk.Canvas(self.root, bg="#151b25", highlightthickness=0, height=180)
        self.canvas.pack(fill="both", expand=True, padx=28, pady=(3, 10))
        self.timeline_scroll = ttk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.timeline_scroll.pack(fill="x", padx=28, pady=(0, 5))
        self.canvas.configure(xscrollcommand=self.timeline_scroll.set)
        tk.Label(self.root, text="Blue = human input     Purple = AI-generated continuation", fg="#9ba8ba",
                 bg=BG, font=("Segoe UI", 9)).pack(anchor="w", padx=30, pady=(0, 16))

    @staticmethod
    def _button(parent: tk.Widget, label: str, command) -> tk.Button:
        return tk.Button(parent, text=label, command=command, fg=TEXT, bg="#293446",
                         activebackground="#354761", activeforeground=TEXT, relief="flat",
                         padx=12, pady=9, cursor="hand2")

    def _choose_note(self, note: str) -> None:
        if self._generating:
            return
        self._generating = True
        duration = {"0.5 seconds": 0.5, "1 second": 1.0, "2 seconds": 2.0}[self.duration.get()]
        bpm, temperature = self.bpm.get(), self.temperature.get()
        chosen_time = datetime.now(timezone.utc)
        chosen_at = chosen_time.isoformat()
        input_interval = None
        if self.composition.interactions:
            try:
                prior_time=datetime.fromisoformat(self.composition.interactions[-1].timestamp)
                input_interval=max(0.0,(chosen_time-prior_time).total_seconds())
            except ValueError:
                pass
        previous = [{"pitch":e.pitch,"start":e.start,"duration":e.duration,"velocity":e.velocity,
                     "source":e.source,"label":e.label} for e in self.composition.events]
        self._set_busy(True)
        self.player.stop()
        self.status.set("AI generating...")
        threading.Thread(target=self._safe_generate_worker,
                         args=(note,duration,bpm,temperature,chosen_at,previous,input_interval),daemon=True).start()

    def _handle_note_key(self, event: tk.Event) -> str | None:
        """Use natural-note keys globally; H is accepted as German B natural."""
        note = NOTE_KEYS.get(str(getattr(event, "char", "")).lower())
        if note is None:
            return None
        self._choose_note(note)
        return "break"

    def _select_model(self, _event=None) -> None:
        stem = LSTMGenerator.CHECKPOINTS.get(self.model_choice.get())
        if not stem:
            return
        self.status.set(f"Loading {self.model_choice.get()}...")
        self.ml_generator.load_checkpoint(APP_DIR / "models" / stem)
        if not self.ml_generator.loaded:
            self.status.set(self.ml_generator.load_error or "Selected checkpoint unavailable")
        else:
            self.status.set(f"Loaded {self.model_choice.get()}")
        self._refresh_model_label()

    def _safe_generate_worker(self, note: str, duration: float, bpm: int, temperature: float,
                              chosen_at: str, previous: list[dict], input_interval: float | None) -> None:
        try:
            self._generate_worker(note,duration,bpm,temperature,chosen_at,previous,input_interval)
        except Exception as exc:
            try:
                self.root.after(0,lambda error=str(exc):self._generation_failed(error))
            except tk.TclError:
                pass

    def _generate_worker(self, note: str, duration: float, bpm: int, temperature: float,
                         chosen_at: str, previous: list[dict], input_interval: float | None) -> None:
        started = perf_counter()
        composition_end = float(self.composition.end_time)
        continuation_boundary = composition_end + human_note_duration(bpm)
        error = None
        active_model = False
        if self.ml_generator.loaded:
            self.ml_generator.temperature = temperature
            try:
                generated = self.ml_generator.generate_continuation(self.composition,note,duration,bpm)
                active_model = True
                model_name = (f"LSTM {self.ml_generator.config.get('layers')}x"
                              f"{self.ml_generator.config.get('hidden_size')}")
            except Exception as exc:
                error = f"LSTM inference failed: {exc}"
                generated = self.generator.generate_continuation(self.composition,note,duration,bpm)
                model_name = "rule-based fallback"
        else:
            error = getattr(self.ml_generator,"load_error",None) or "Trained LSTM checkpoint unavailable"
            generated = self.generator.generate_continuation(self.composition,note,duration,bpm)
            model_name = "rule-based fallback"
        latency = perf_counter()-started
        config = self.ml_generator.config
        device = getattr(self.ml_generator,"device_name","CPU")
        device_kind = str(getattr(self.ml_generator,"device","cpu"))
        generated_before_sanitization = list(generated)
        generated, timeline_diagnostics = sanitize_timeline(generated, continuation_boundary, ("ai",))
        timeline_debug = {
            "composition_end": composition_end,
            "continuation_boundary": continuation_boundary,
            "local_generated_start_end": [
                {"start": round(e.start - continuation_boundary, 6),
                 "end": round(e.start - continuation_boundary + e.duration, 6)}
                for e in generated_before_sanitization],
            "absolute_final_start_end": [
                {"start": round(e.start, 6), "end": round(e.start + e.duration, 6)}
                for e in generated],
            "overlap_detected": timeline_diagnostics["before"]["overlap_detected"],
            "events_before_sanitization": timeline_diagnostics["events_before_sanitization"],
            "events_after_sanitization": timeline_diagnostics["events_after_sanitization"],
            "validation_after": timeline_diagnostics["after"],
        }
        if timeline_diagnostics["before"]["overlap_detected"] or not timeline_diagnostics["after"]["valid"]:
            print("TIMELINE WARNING: continuation sanitization detected/repaired issues "
                  + json.dumps({**timeline_debug, "diagnostics": timeline_diagnostics}))
        else:
            print("TIMELINE GUI VALIDATION " + json.dumps(timeline_debug))
        metadata = {"previous_composition":previous,"temperature":temperature,"model":model_name,
                    "top_k":getattr(self.ml_generator,"top_k",20),
                    "top_p":getattr(self.ml_generator,"top_p",.90),
                    "hidden_size":config.get("hidden_size",512),"num_layers":config.get("layers",3),
                    "embedding_size":config.get("embedding_size",256),
                    "context_length":config.get("sequence_length",128),
                    "human_input_frequency":"one human decision per AI continuation",
                    "human_input_interval_seconds":input_interval,
                    "human_input_frequency_hz":(1.0/input_interval if input_interval and input_interval>0 else None),
                    "ai_autonomy_seconds":duration,"device":device_kind,"device_name":device,
                    "ai_response_time_seconds":round(latency,6),
                    "composition_end":composition_end,
                    "continuation_boundary":continuation_boundary,
                    "timeline_diagnostics":timeline_diagnostics,
                    "inference_timeline_diagnostics":getattr(self.ml_generator,"last_timeline_diagnostics",None),
                    "ai_generated_events":[e.__dict__.copy() for e in generated]}
        try:
            self.root.after(0,lambda:self._finish_generation(note,duration,bpm,chosen_at,generated,
                                                            metadata,active_model,error))
        except tk.TclError:
            pass

    def _generation_failed(self, error: str) -> None:
        if self._closed:
            return
        self._set_busy(False)
        self._generating=False
        self.status.set(f"Generation failed: {error}")

    def _finish_generation(self, note: str, duration: float, bpm: int, chosen_at: str,
                           generated, metadata: dict, active_model: bool, error: str | None) -> None:
        if self._closed:
            return
        interaction=self.composition.add_interaction(note,generated,duration,bpm,metadata,chosen_at)
        post_validation = validate_timeline(interaction.events, monophonic_sources=("ai",))
        if not post_validation["valid"]:
            print("TIMELINE WARNING after append " + json.dumps(post_validation))
        else:
            print("TIMELINE POST-APPEND " + json.dumps(post_validation))
        self._generating=False
        self._refresh(); self._save_log(); self._set_busy(False)
        self._refresh_model_label(active_model=active_model,error=error)
        self.status.set("Playing AI continuation" if self.player.playback_mode != "unavailable" else
                        "Generation saved; audio playback unavailable")
        self.player.play(interaction.events)

    def _refresh_model_label(self, active_model: bool | None = None, error: str | None = None) -> None:
        device=getattr(self.ml_generator,"device_name","CPU")
        device_type=str(getattr(self.ml_generator,"device","cpu"))
        if active_model is None:
            active_model=self.ml_generator.loaded
        if active_model:
            text=(f"AI Model: LSTM {self.ml_generator.config.get('layers',3)}×"
                  f"{self.ml_generator.config.get('hidden_size',512)} · Device: {device_type} / {device}")
        else:
            text=f"ML model unavailable · Using rule-based fallback · Device: {device_type} / {device}"
            details=error or getattr(self.ml_generator,"load_error",None)
            if details:
                text+=f"\n{details}"
        self.model_label.config(text=text)

    def _set_busy(self, busy: bool) -> None:
        state="disabled" if busy else "normal"
        for button in self._note_buttons:
            button.configure(state=state)
        for widget in self._busy_controls:
            try:
                widget.configure(state="disabled" if busy else ("readonly" if widget is self.duration_box else "normal"))
            except tk.TclError:
                pass

    def _refresh(self) -> None:
        parts = []
        for item in self.composition.interactions:
            parts += [f"HUMAN {item.human_note}",
                      "AI " + (" → ".join(item.ai_notes) if item.ai_notes else "(no notes)")]
        self.sequence.config(text="     |     ".join(parts) if parts else "No notes yet")
        self._draw_timeline()

    def _draw_timeline(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        if not self.composition.events:
            c.create_text(w / 2, 85, text="Your co-created timeline will appear here",
                          fill="#78869a", font=("Segoe UI", 11))
            return
        total = max(self.composition.end_time, 0.1)
        scale = max(55, (w - 60) / total)
        c.create_text(18, 15, text="HUMAN INPUT", anchor="w", fill=HUMAN,
                      font=("Segoe UI", 8, "bold"))
        c.create_text(18, 63, text="AI CONTINUATION", anchor="w", fill=AI,
                      font=("Segoe UI", 8, "bold"))
        for interaction in self.composition.interactions:
            human = next((e for e in interaction.events if e.source == "human"), None)
            if human:
                x=20+human.start*scale; width=max(48,human.duration*scale)
                c.create_rectangle(x,25,x+width,50,fill=HUMAN,outline="")
                c.create_text(x+width/2,37,text=f"H {interaction.human_note}",
                              fill="#11151d",font=("Segoe UI",8,"bold"))
            ai_events=[e for e in interaction.events if e.source=="ai"]
            for event in ai_events:
                x=20+event.start*scale; width=max(22,event.duration*scale)
                y=max(78,min(148,120-(event.pitch-60)*4))
                c.create_rectangle(x,y,x+width,y+22,fill=AI,outline="")
                c.create_text(x+width/2,y+11,text=event.label,fill="#11151d",
                              font=("Segoe UI",8,"bold"))
            if human and ai_events:
                x=20+human.start*scale
                phrase=" → ".join(e.label for e in ai_events)
                c.create_text(x,166,text=f"AI: {phrase}",anchor="w",fill="#c5b5ff",
                              font=("Segoe UI",8))
        c.create_line(20,54,max(w-18,20+total*scale),54,fill="#354052",width=1)
        c.configure(scrollregion=(0,0,max(w,int(30+total*scale)),180))

    def _play(self) -> None:
        if self.composition.events:
            self.player.play(self.composition.events)
        else:
            self.status.set("Nothing to play")

    def _pause(self) -> None:
        if self.player.is_paused:
            self.player.resume()
            self.status.set("Playing...")
        else:
            self.player.pause()
            self.status.set("Paused")

    def _stop(self) -> None:
        self.player.stop()
        self.status.set("Ready")

    def _undo(self) -> None:
        self.player.stop()
        self.composition.undo()
        self._refresh()
        self._save_log()
        self.status.set("Waiting for human input" if self.composition.events else "Ready")

    def _clear(self) -> None:
        self.player.stop()
        self.composition.clear()
        self._refresh()
        self._save_log()
        self.status.set("Ready")

    def _save_log(self) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            path = LOG_DIR / f"session_{stamp}.json"
            path.write_text(json.dumps(self.composition.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            self.status.set(f"Log save failed: {exc}")

    def _export_log(self) -> None:
        filename=filedialog.asksaveasfilename(defaultextension=".json",
                                               filetypes=[("JSON interaction log","*.json")])
        if not filename:
            return
        try:
            with open(filename,"w",encoding="utf-8") as stream:
                json.dump(self.composition.to_dict(),stream,indent=2)
            self.status.set(f"Interaction log exported: {filename}")
        except OSError as exc:
            messagebox.showerror("Log export failed",str(exc))

    def _export_midi(self) -> None:
        if not self.composition.events:
            self.status.set("Nothing to export")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
        if not filename:
            return
        try:
            import mido
            midi = mido.MidiFile(); track = mido.MidiTrack(); midi.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo",tempo=round(60_000_000/self.bpm.get()),time=0))
            export_validation = validate_timeline(self.composition.events, monophonic_sources=("ai",))
            if not export_validation["valid"]:
                print("TIMELINE WARNING before MIDI export " + json.dumps(export_validation))
            ticks_per_second = midi.ticks_per_beat * self.bpm.get() / 60.0
            midi_events = []
            for event in sorted(self.composition.events, key=lambda e: e.start):
                start_tick = max(0, round(event.start * ticks_per_second))
                end_tick = start_tick + max(1, round(event.duration * ticks_per_second))
                midi_events.append((start_tick, 1, mido.Message("note_on", note=max(0,min(127,event.pitch)), velocity=max(1,min(127,event.velocity)))))
                midi_events.append((end_tick, 0, mido.Message("note_off", note=max(0,min(127,event.pitch)), velocity=0)))
            previous_tick = 0
            for tick, _, message in sorted(midi_events, key=lambda item: (item[0], item[1])):
                track.append(message.copy(time=tick-previous_tick)); previous_tick = tick
            midi.save(filename); self.status.set(f"MIDI exported: {filename}")
        except Exception as exc:
            messagebox.showerror("MIDI export failed", f"Install mido with python -m pip install mido\\n\\n{exc}")

    def _set_status_threadsafe(self, value: str) -> None:
        if self._closed:
            return
        try:
            self.root.after(0, lambda:self.status.set(value) if not self._closed else None)
        except tk.TclError:
            pass

    def _on_close(self) -> None:
        self._closed=True
        self.player.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
