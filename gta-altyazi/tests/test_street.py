import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from altyazi import street
from altyazi.street import (Segment, Segmenter, SileroVad, StreetListener, VadSegmenter, accept, is_prompt_echo,
                            name_prompt, silero_model_path)

SR, B = 16000, 800  # 50 ms bloklar


def _feed(seg, signal, t0=100.0, keep_early=False):
    out = []
    for i in range(0, len(signal) - B + 1, B):
        s = seg.feed(t0 + (i + B) / SR, signal[i:i + B])
        if s is not None and (keep_early or s.final):
            out.append((t0 + (i + B) / SR, s) if keep_early else s)
    return out


def _noise(sec, level, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(sec * SR)) * level).astype(np.float32)


def _speech(sec, seed=1):
    t = np.arange(int(sec * SR)) / SR
    syll = (np.sin(2 * np.pi * 3.5 * t) > -0.3).astype(np.float32)  # heceler
    return (0.15 * np.sin(2 * np.pi * 140 * t) * syll + _noise(sec, 0.002, seed)).astype(np.float32)


def test_one_spoken_sentence_is_one_segment():
    seg = Segmenter()
    audio = np.concatenate([_noise(1.0, 0.002), _speech(1.5), _noise(1.0, 0.002)])
    out = _feed(seg, audio)
    assert len(out) == 1
    assert 1.3 <= out[0].end - out[0].start <= 2.4


def test_short_bang_is_not_speech():
    seg = Segmenter()
    bang = np.random.default_rng(3).standard_normal(int(0.2 * SR)).astype(np.float32) * 0.3
    assert _feed(seg, np.concatenate([_noise(1.0, 0.002), bang, _noise(1.0, 0.002)])) == []


def test_endless_loud_audio_is_cut():
    seg = Segmenter()
    out = _feed(seg, np.concatenate([_noise(0.5, 0.002), _speech(20.0)]))
    assert out and all(s.end - s.start <= Segmenter.MAX_S + 0.5 for s in out)


def test_whisper_results_are_filtered():
    assert accept(" Hey, watch where you're going! ", 0.05, -0.3) == "Hey, watch where you're going!"
    assert accept("♪ some radio song ♪", 0.05, -0.3) == ""   # radyo / müzik
    assert accept("[Music]", 0.1, -0.2) == ""
    assert accept("Thank you.", 0.2, -0.5) == ""             # Whisper'ın sessizlikte uydurduğu
    assert accept("Hey, watch out!", 0.8, -0.3) == ""         # konuşma yok olasılığı yüksek
    assert accept("Hey, watch out!", 0.1, -1.4) == ""         # düşük güven
    assert accept("go go go go go go go go go go", 0.1, -0.3, compression_ratio=3.1) == ""  # tekrar döngüsü


# --- Silero parçalayıcı (sahte VAD: 250 Hz ton "konuşma", başka her şey değil) ----------------------

TONE_BIN = 8  # 512 örneklik çerçevede 250 Hz


class ToneVad:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, frame):
        self.calls += 1
        spec = np.abs(np.fft.rfft(frame)) ** 2
        return 0.9 if spec[TONE_BIN] > 0.5 * spec.sum() else 0.05


def _tone(sec, level=0.1):
    t = np.arange(int(sec * SR)) / SR
    return (level * np.sin(2 * np.pi * 250 * t)).astype(np.float32)


def _quiet(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def test_vad_segment_ends_half_a_second_after_speech_and_early_matches_final():
    seg = VadSegmenter(ToneVad())
    audio = np.concatenate([_quiet(1.0), _tone(1.5), _quiet(1.5)])
    events = _feed(seg, audio, t0=0.0, keep_early=True)
    assert [s.final for _, s in events] == [False, True]
    (early_at, early), (final_at, final) = events
    assert early.key == final.key and np.array_equal(early.samples, final.samples)
    assert abs(final.end - 2.5) < 0.05                  # konuşmanın bittiği an
    assert 0.15 <= early_at - 2.5 <= 0.3                 # erken tanıma ~0,2 sn sonra
    assert 0.45 <= final_at - 2.5 <= 0.6                 # parça ~0,5 sn sonra kapanır
    assert 0.75 <= final.start <= 0.85                   # 0,2 sn baş payı (ilk hece kaçmasın)
    assert len(final.samples) / SR < 1.5 + VadSegmenter.PAD_S + VadSegmenter.TAIL_S + 0.1


def test_loud_non_speech_does_not_keep_segment_open():
    seg = VadSegmenter(ToneVad())
    engine = _noise(4.0, 0.3, seed=5)  # motor/silah/müzik: yüksek ama konuşma değil
    audio = np.concatenate([_noise(1.0, 0.3, seed=4), _tone(1.5, 0.5) + _noise(1.5, 0.01), engine])
    events = _feed(seg, audio, t0=0.0, keep_early=True)
    finals = [(at, s) for at, s in events if s.final]
    assert len(finals) == 1
    at, final = finals[0]
    assert 0.45 <= at - 2.5 <= 0.6 and abs(final.end - 2.5) < 0.05


def test_resumed_speech_invalidates_early_result_and_continues_segment():
    seg = VadSegmenter(ToneVad())
    audio = np.concatenate([_quiet(0.5), _tone(1.0), _quiet(0.3), _tone(1.0), _quiet(1.0)])
    events = _feed(seg, audio, t0=0.0, keep_early=True)
    early = [s for _, s in events if not s.final]
    finals = [s for _, s in events if s.final]
    assert len(finals) == 1 and len(early) == 2
    assert early[0].key != finals[0].key == early[1].key  # ilk erken sonuç atılır
    assert finals[0].end - finals[0].start > 2.2           # iki parça tek cümle


def test_pending_key_clears_when_speech_resumes():
    seg = VadSegmenter(ToneVad())
    _feed(seg, np.concatenate([_quiet(0.5), _tone(1.0), _quiet(0.3)]), t0=0.0)
    assert seg.pending_key is not None
    seg.feed(10.0, _tone(0.1)[:B])
    assert seg.pending_key is None


def test_vad_short_blip_and_silence_make_no_segment():
    vad = ToneVad()
    seg = VadSegmenter(vad)
    audio = np.concatenate([_quiet(1.0), _tone(0.2), _quiet(1.0)])
    assert _feed(seg, audio, keep_early=True) == []
    assert vad.calls <= 10  # sessiz çerçeveler modele gitmez (yalnızca ~0,2 sn ton)
    assert vad.resets >= 1  # sessizlikten sonra durum sıfırlanır


def test_vad_endless_speech_is_cut():
    seg = VadSegmenter(ToneVad())
    out = _feed(seg, np.concatenate([_quiet(0.5), _tone(20.0)]))
    assert len(out) >= 2 and all(s.end - s.start <= VadSegmenter.MAX_S + 0.1 for s in out)


def test_resample_to_16k():
    block = np.sin(2 * np.pi * 300 * np.arange(4800) / 48000).astype(np.float32)
    out = street.to_16k(block, 48000)
    assert len(out) == 1600 and out.dtype == np.float32
    assert street.to_16k(block, 16000) is block


# --- gerçek Silero modeli (faster-whisper kuruluysa) ---------------------------------------------

def _resonate(x, freq, bw):
    r, th = np.exp(-np.pi * bw / SR), 2 * np.pi * freq / SR
    a1, a2 = -2 * r * np.cos(th), r * r
    y, y1, y2 = np.zeros_like(x), 0.0, 0.0
    for i, v in enumerate(x):
        y0 = v - a1 * y1 - a2 * y2
        y[i], y2, y1 = y0, y1, y0
    return y


def _formant_speech(sec, seed=1):
    """Ses telleri darbeleri + ünlü formantları + hece zarfı: Silero'nun konuşma saydığı yapay ses."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    t = np.arange(n) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t) + 8 * np.sin(2 * np.pi * 3.1 * t)
    src = np.diff(np.floor(np.cumsum(f0 / SR)), prepend=0.0) + 0.02 * rng.standard_normal(n)
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240)]
    out, syl = np.zeros(n), int(0.22 * SR)
    for a in range(0, n, syl):
        part = src[a:a + syl]
        formants = vowels[rng.integers(len(vowels))]
        y = sum(_resonate(part, f, bw) * g for f, bw, g in zip(formants, (80, 100, 120), (1, 0.5, 0.25)))
        out[a:a + syl] = y * np.sin(np.pi * np.arange(len(part)) / len(part)) ** 0.6
    return (0.2 * out / np.max(np.abs(out))).astype(np.float32)


def _engine(sec, seed=7):
    t = np.arange(int(sec * SR)) / SR
    phase = 2 * np.pi * np.cumsum(45 + 5 * np.sin(2 * np.pi * 0.3 * t)) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * (1 + 0.3 * np.sin(2 * np.pi * 8 * t))
    return (0.15 * x / np.max(np.abs(x)) + _noise(sec, 0.01, seed)).astype(np.float32)


@pytest.fixture(scope="module")
def silero():
    path = silero_model_path()
    if path is None:
        pytest.skip("faster-whisper (Silero modeli) kurulu değil")
    pytest.importorskip("onnxruntime")
    return SileroVad(path)


def test_real_silero_separates_speech_from_engine_noise(silero):
    seg = VadSegmenter(silero)
    audio = np.concatenate([_engine(1.0, 1), _formant_speech(2.0) + _engine(2.0, 2) * 0.3, _engine(3.0, 3)])
    events = _feed(seg, audio, t0=0.0, keep_early=True)
    finals = [(at, s) for at, s in events if s.final]
    assert len(finals) == 1
    at, final = finals[0]
    # Silero konuşma bitişini ~0,1-0,3 sn geç görür; motor sesi sürse de parça kapanır.
    assert abs(final.end - 3.0) < 0.35 and at - 3.0 < 0.9
    assert any(not s.final for _, s in events)


def test_real_silero_ignores_noise_only(silero):
    seg = VadSegmenter(silero)
    audio = np.concatenate([_engine(3.0, 4), _noise(2.0, 0.1, seed=9)])
    assert _feed(seg, audio, keep_early=True) == []


def test_make_segmenter_falls_back_without_model(monkeypatch):
    monkeypatch.setattr(street, "silero_model_path", lambda: None)
    logs = []
    assert isinstance(street.make_segmenter(logs.append), Segmenter)
    assert logs


# --- ad istemi --------------------------------------------------------------------------------------

def test_name_prompt_keeps_only_proper_nouns_and_is_short():
    from altyazi import glossary

    _general, bundled = glossary.load_bundled()
    user = {"Big Mike": "Big Mike", "cop": "polis", "Wasted": "Öldün", "Jason": "Jason"}
    prompt = name_prompt(user, bundled)
    assert prompt.startswith("Big Mike, ")
    assert len(prompt) <= street.PROMPT_CHARS
    names = prompt.split(", ")
    assert "Vice City" in names and "Jason Duval" in names
    assert "Jason" not in names and "cop" not in names and "Wasted" not in names  # alt ad / çeviri
    assert name_prompt({}, {}) == ""


def test_prompt_echo_is_detected():
    prompt = "Jason Duval, Lucia Caminos, Vice City, Leonida Keys"
    assert is_prompt_echo("Jason Duval, Lucia Caminos.", prompt)
    assert is_prompt_echo("Vice City, Leonida Keys, Jason Duval, Lucia", prompt)
    assert not is_prompt_echo("Lucia, get in the car!", prompt)
    assert not is_prompt_echo("Hey Jason, you owe me money.", prompt)
    assert not is_prompt_echo("Jason Duval", "")


# --- dinleyici: erken tanıma ve tek iş ----------------------------------------------------------------

class FakeModel:
    def __init__(self, texts, gate=None):
        self.texts = texts  # örnek sayısı -> metin
        self.gate = gate
        self.calls = []

    def transcribe(self, samples, **kw):
        self.calls.append((len(samples), kw))
        if self.gate is not None:
            assert self.gate.wait(5)
        part = SimpleNamespace(text=self.texts.get(len(samples), "Hello there, friend."), no_speech_prob=0.1,
                               avg_logprob=-0.2, compression_ratio=1.2)
        return iter([part]), None


class FakeTranslator:
    def __init__(self):
        self.lines = []
        self.sources = []

    def submit(self, text, context, is_current, on_result, on_error, source="ocr"):
        self.lines.append(text)
        self.sources.append(source)


def _listener(model, blocked=lambda a, b: False):
    cfg = SimpleNamespace(street_enabled=True, street_model="base.en", performance_mode="normal")
    tr = FakeTranslator()
    lst = StreetListener(cfg, "sokak", lambda: None, tr, __import__("queue").Queue(), blocked, lambda m: None)
    lst._model = model
    lst.prompt = "Jason Duval, Vice City"
    lst._ensure_worker()
    return lst, tr


def _seg(n, final, key=(1, 0), start=0.0):
    return Segment(start, start + n / SR, np.zeros(n, np.float32), final, key)


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.005)
    return cond()


def test_early_result_is_used_when_segment_closes():
    gate = threading.Event()
    model = FakeModel({16000: "Watch where you're going!"}, gate)
    lst, tr = _listener(model)
    lst.on_segment(_seg(16000, False), (1, 0))
    lst.on_segment(None, (1, 0))
    gate.set()
    assert _wait(lambda: model.calls)
    lst.on_segment(_seg(16000, True), None)
    assert _wait(lambda: tr.lines)
    assert tr.lines == ["Watch where you're going!"] and len(model.calls) == 1 and lst.early_used == 1
    assert tr.sources == ["asr"]  # çeviriye konuşma tanıma satırı olarak gider
    kw = model.calls[0][1]
    assert kw["temperature"] == 0.0 and kw["initial_prompt"] == "Jason Duval, Vice City"


def test_early_result_is_discarded_when_speech_resumes():
    model = FakeModel({16000: "Watch where", 32000: "Watch where you're going, man!"})
    lst, tr = _listener(model)
    lst.on_segment(_seg(16000, False), (1, 16000))
    lst.on_segment(None, None)  # konuşma sürdü
    assert _wait(lambda: not lst._jobs)
    lst.on_segment(_seg(32000, True, key=(1, 32000)), None)
    assert _wait(lambda: tr.lines)
    time.sleep(0.05)
    assert tr.lines == ["Watch where you're going, man!"]


def test_only_one_recognition_runs_and_early_waits_for_idle():
    gate = threading.Event()
    model = FakeModel({}, gate)
    lst, tr = _listener(model)
    lst.on_segment(_seg(8000, True, key=(1, 1)), None)
    assert _wait(lambda: len(model.calls) == 1)
    lst.on_segment(_seg(9000, False, key=(2, 1)), (2, 1))  # işçi meşgul: erken tanıma başlamaz
    assert lst._early is None
    lst.on_segment(_seg(9000, True, key=(2, 1)), None)
    time.sleep(0.05)
    assert len(model.calls) == 1  # ikinci iş ilki bitene kadar bekler
    gate.set()
    assert _wait(lambda: len(tr.lines) == 2)
    assert [n for n, _ in model.calls] == [8000, 9000]


def test_blocked_final_drops_early_result():
    model = FakeModel({})
    lst, tr = _listener(model, blocked=lambda a, b: True)
    lst.on_segment(_seg(16000, False), (1, 0))  # altyazı varken erken tanıma da başlamaz
    lst.on_segment(_seg(16000, True), None)
    time.sleep(0.05)
    assert tr.lines == [] and model.calls == [] and lst.skipped == 1


def test_prompt_echo_is_not_shown():
    model = FakeModel({16000: "Jason Duval, Vice City."})
    lst, tr = _listener(model)
    lst.on_segment(_seg(16000, True), None)
    assert _wait(lambda: model.calls and not lst._jobs)
    time.sleep(0.05)
    assert tr.lines == []
