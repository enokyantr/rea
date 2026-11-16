import numpy as np

from altyazi.street import Segmenter, accept

SR, B = 16000, 800  # 50 ms bloklar


def _feed(seg, signal, t0=100.0):
    out = []
    for i in range(0, len(signal) - B + 1, B):
        s = seg.feed(t0 + (i + B) / SR, signal[i:i + B])
        if s is not None:
            out.append(s)
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
