"""Stateful, filtered autoregressive inference for the symbolic MusicLSTM."""
from __future__ import annotations

from pathlib import Path
import torch
from ml.model import MusicLSTM
from ml.preprocessing import Event, encode_event, decode_duration, decode_timing, decode_velocity
from composition import NoteEvent, human_note_duration, sanitize_timeline, validate_timeline


def _sample(logits: torch.Tensor, temperature: float, top_k: int = 20,
            top_p: float = .90, penalty: torch.Tensor | None = None) -> int:
    """Sample with repetition penalty, top-k and nucleus filtering."""
    logits = torch.nan_to_num(logits.float().clone(), nan=0.0, posinf=50.0, neginf=-50.0)
    if penalty is not None:
        logits -= penalty
    logits /= max(.05, float(temperature))
    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = -torch.inf
        filtered = torch.full_like(logits, -torch.inf).scatter(0, indices, sorted_logits)
        logits = filtered
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


def autoregressive_generate(model: MusicLSTM, events: list[Event], target_note: int,
                           duration: float, bpm: int, temperature: float = .75,
                           device: torch.device | None = None, sequence_length: int = 256,
                           top_k: int = 20, top_p: float = .90,
                           chunk_seconds: float = .5) -> list[Event]:
    """Generate stateful onset times; ``chunk_seconds`` stays for call compatibility.

    Timing tokens represent onset-to-onset gaps. In monophonic GUI output, each
    gap is raised to the previous note duration to avoid overlapping note-ons.
    """
    device = device or next(model.parameters()).device
    rows = [encode_event(e) for e in events[-sequence_length:]] or [(60, 3, 4, 0)]
    generated: list[Event] = []
    elapsed = 0.0  # Local onset time of the next generated note.
    recent_pitches: list[int] = [row[0] for row in rows[-12:]]
    recent_rhythm: list[int] = [row[3] for row in rows[-12:]]
    repeated = 0
    model.eval()
    target = torch.tensor([min(127, max(0, int(target_note)))], device=device)
    tempo = torch.tensor([min(240, max(30, int(bpm)))], dtype=torch.float32, device=device)
    pitch_penalty = torch.zeros(128, device=device)
    with torch.inference_mode():
        # Prefill once from context. At chunk boundaries, keep hidden state.
        row = torch.tensor(rows[-sequence_length:], dtype=torch.long, device=device).unsqueeze(0)
        out, hidden = model(row[:, :, 0], row[:, :, 1], row[:, :, 2], row[:, :, 3], target, tempo)
        logits = {name: values[0, -1] for name, values in out.items()}
        while elapsed < duration and len(generated) < 256:
            pitch_penalty.zero_()
            for pitch in recent_pitches[-8:]:
                pitch_penalty[pitch] += 0.22
            if len(recent_pitches) >= 3 and len(set(recent_pitches[-3:])) == 1:
                pitch_penalty[recent_pitches[-1]] += 2.0
            # Mild target attraction only near the phrase end; intermediate ideas remain free.
            remaining = max(0.0, duration - elapsed)
            target_distance = torch.arange(128, device=device).sub(target_note).abs()
            if remaining < .35:
                pitch_penalty += target_distance.float() * (0.012 * (1.0 - remaining / .35))
            pitch = _sample(logits["pitch"], temperature, top_k, top_p, pitch_penalty)
            dur_token = _sample(logits["duration"], temperature, top_k, top_p)
            vel_token = _sample(logits["velocity"], temperature, top_k, top_p)
            timing_value = torch.nan_to_num(logits["timing"], nan=0.0, posinf=2.0, neginf=0.0)
            timing_token = int(round(min(15.0, max(0.0, float(timing_value.item()) * 7.5))))
            dt = min(.75, max(0.0, decode_timing(timing_token)))
            # This app generates a monophonic AI line. Timing tokens encode
            # onset-to-onset gaps, so never let a new onset precede note-off.
            if generated:
                dt = max(dt, generated[-1].duration)
            elif dt < .025:
                dt = .03
            if elapsed + dt >= duration:
                break
            duration_s = min(2.0, max(.08, decode_duration(dur_token)))
            velocity = min(127, max(1, decode_velocity(vel_token)))
            pitch = min(127, max(0, pitch))
            if recent_pitches and pitch == recent_pitches[-1]:
                repeated += 1
            else:
                repeated = 0
            if repeated >= 3:
                pitch = (pitch + (2 if pitch <= 125 else -2)) % 128
                repeated = 0
            if recent_pitches:
                pitch = min(recent_pitches[-1] + 12, max(recent_pitches[-1] - 12, pitch))
            event = Event(pitch, duration_s, velocity, max(0.0, dt))
            generated.append(event)
            elapsed += dt
            recent_pitches.append(pitch); recent_pitches = recent_pitches[-16:]
            recent_rhythm.append(timing_token); recent_rhythm = recent_rhythm[-16:]
            # Feed only the newly sampled event; hidden state carries through chunks.
            token = torch.tensor([[encode_event(event)]], dtype=torch.long, device=device)
            out, hidden = model(token[:, :, 0], token[:, :, 1], token[:, :, 2], token[:, :, 3],
                                target, tempo, hidden)
            logits = {name: values[0, -1] for name, values in out.items()}
    return generated or [Event(min(127, max(0, target_note)), .12, 80, 0.0)]


class LSTMGenerator:
    """Load-once checkpoint adapter; supports switching between model sizes."""
    CHECKPOINTS = {"LSTM 3x512 (baseline)": "best_model_3x512.pt",
                   "LSTM 4x768 (experimental)": "best_model_4x768.pt"}

    def __init__(self, checkpoint: str | Path = "models/best_model_3x512.pt",
                 temperature: float = .75) -> None:
        self.temperature = temperature
        self.top_k = 20
        self.top_p = .90
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        self.model: MusicLSTM | None = None
        self.config: dict = {}
        self.load_error: str | None = None
        self.checkpoint = Path(checkpoint)
        self.load_checkpoint(self.checkpoint)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        self.checkpoint = path
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location=self.device, weights_only=False)
            config = state.get("config", {})
            expected = {"best_model_3x512.pt": (3, 512, 256, .25),
                        "best_model_4x768.pt": (4, 768, 384, .30)}
            expected_spec = expected.get(self.checkpoint.name)
            actual_spec = (int(config.get("layers", -1)), int(config.get("hidden_size", -1)),
                           int(config.get("embedding_size", -1)), float(config.get("dropout", -1)))
            if expected_spec and actual_spec != expected_spec:
                raise ValueError(f"{path.name} has architecture {actual_spec}; expected {expected_spec}")
            if config.get("target_conditioning") != "future_window_final_pitch":
                raise ValueError("Checkpoint has incompatible target conditioning")
            model = MusicLSTM(embedding_size=config.get("embedding_size", 256),
                              hidden_size=config.get("hidden_size", 512),
                              num_layers=config.get("layers", 3),
                              dropout=config.get("dropout", .25)).to(self.device)
            model.load_state_dict(state["model_state"]); model.eval()
            self.config, self.model, self.load_error = config, model, None
            self.top_k = int(config.get("top_k", 20)); self.top_p = float(config.get("top_p", .90))
        except Exception as exc:
            self.model = None; self.config = {}; self.load_error = f"Checkpoint unavailable: {exc}"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def generate_continuation(self, composition, target_note: str, duration: float, bpm: int):
        if self.model is None:
            raise RuntimeError("Trained LSTM checkpoint unavailable")
        names = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71}
        context_length = self.config.get("sequence_length", 128)
        composition_end = float(composition.end_time)
        continuation_boundary = composition_end + human_note_duration(bpm)
        context = composition.events[-context_length:]
        prior = []
        previous_start = context[0].start if context else 0.0
        for index, event in enumerate(context):
            delta = 0.0 if index == 0 else max(0.0, event.start - previous_start)
            prior.append(Event(event.pitch, max(.08,event.duration), event.velocity, delta))
            previous_start = event.start
        started = __import__("time").perf_counter()
        notes = autoregressive_generate(self.model, prior, names[target_note], duration, bpm,
                                        self.temperature, self.device, context_length,
                                        self.top_k, self.top_p)
        self.last_latency_seconds = __import__("time").perf_counter() - started
        local_events = []
        cursor = 0.0
        for e in notes:
            cursor += e.delta_time
            label = next((n for n,p in names.items() if p == e.pitch), f"{e.pitch}")
            local_events.append(NoteEvent(e.pitch, cursor, max(.08, e.duration), e.velocity, "ai", label))
        sanitized_local, diagnostics = sanitize_timeline(local_events, 0.0, ("ai",))
        absolute_events = [NoteEvent(e.pitch, continuation_boundary + e.start, e.duration,
                                     e.velocity, e.source, e.label) for e in sanitized_local]
        final_validation = validate_timeline(absolute_events, continuation_boundary, ("ai",))
        self.last_timeline_diagnostics = {
            "composition_end": composition_end,
            "human_input_end": continuation_boundary,
            "continuation_boundary": continuation_boundary,
            "duration_seconds": duration,
            "local_events_before_sanitization": [
                {"start": round(e.start, 6), "end": round(e.start + e.duration, 6)} for e in local_events],
            "absolute_events_after_sanitization": [
                {"start": round(e.start, 6), "end": round(e.start + e.duration, 6)} for e in absolute_events],
            "overlap_detected": diagnostics["before"]["overlap_detected"],
            "events_before_sanitization": diagnostics["events_before_sanitization"],
            "events_after_sanitization": diagnostics["events_after_sanitization"],
            "local_validation_before": diagnostics["before"],
            "absolute_validation_after": final_validation,
        }
        print("TIMELINE DEBUG " + __import__("json").dumps(self.last_timeline_diagnostics))
        if diagnostics["before"]["overlap_detected"] or not final_validation["valid"]:
            print("TIMELINE WARNING: generated timing required repair or still contains an issue")
        return absolute_events
