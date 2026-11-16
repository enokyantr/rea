"""Konuşanı sesinden tanıma: sentetik seslerle (gırtlak darbeleri + formant süzgeçleri + müzik)."""

import numpy as np

from altyazi import voiceid as v

SR = 16000


def _resonate(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    a1, a2 = -2 * r * np.cos(2 * np.pi * f / SR), r * r
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for n in range(len(x)):
        y0 = x[n] - a1 * y1 - a2 * y2
        y[n], y2, y1 = y0, y1, y0
    return y


def _speak(f0, formants, seed, dur=0.7, music=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(1.5, 3) * t + rng.uniform(0, 6)))
    src = (np.diff(np.floor(np.cumsum(pitch / SR)), prepend=0) > 0).astype(float)
    y = src
    for f, bw in formants:
        y = _resonate(y, f * rng.uniform(0.95, 1.05), bw)
    y *= (np.sin(2 * np.pi * 4 * t + seed) > -0.6) * np.clip(np.sin(np.pi * t / dur) * 3, 0, 1)
    y /= np.abs(y).max() + 1e-9
    backing = music * (0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 330 * t + 1))
    return (y + backing + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


MAN_A = (110, [(700, 90), (1200, 110), (2500, 160)])
MAN_B = (125, [(400, 80), (2000, 120), (2800, 170)])  # güçlü 400 Hz formant: perde sanılmamalı
WOMAN = (210, [(800, 100), (1500, 130), (2900, 180)])


def test_gender_and_same_character_gets_same_voice(tmp_path):
    book = v.CharacterBook(tmp_path / "karakterler.json")
    seen: dict[str, set[int]] = {}
    for i, who in enumerate(["A", "W", "B", "A", "B", "W", "A"]):
        f0, formants = {"A": MAN_A, "B": MAN_B, "W": WOMAN}[who]
        feats = v.features(_speak(f0, formants, seed=i))
        assert feats is not None
        assert feats.gender == ("kadin" if who == "W" else "erkek"), (who, feats.f0)
        seen.setdefault(who, set()).add(book.identify(feats).id)
    assert all(len(ids) == 1 for ids in seen.values()), seen  # her karakter hep aynı
    assert len({next(iter(ids)) for ids in seen.values()}) == 3  # birbirine karışmadı
    # Hafıza kalıcı; iki erkek farklı ses ayarı aldı
    again = v.CharacterBook(tmp_path / "karakterler.json").snapshot()
    men = [c for c in again if c.gender == "erkek"]
    assert len(again) == 3 and (men[0].pitch_pct, men[0].rate_pct) != (men[1].pitch_pct, men[1].rate_pct)


def test_silence_and_noise_are_not_voices():
    assert v.features(np.zeros(SR, dtype=np.float32)) is None
    assert v.features(np.random.default_rng(1).standard_normal(SR).astype(np.float32) * 0.1) is None


def test_manual_gender_is_kept(tmp_path):
    book = v.CharacterBook(tmp_path / "k.json")
    ch = book.identify(v.features(_speak(*MAN_A, seed=1)))
    book.set_gender(ch.id, "kadin")
    again = book.identify(v.features(_speak(*MAN_A, seed=2)))
    assert again.id == ch.id and again.gender == "kadin"  # kullanıcının seçimi korunur
