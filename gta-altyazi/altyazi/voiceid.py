"""Konuşanı sesinden tanıma (platformdan bağımsız, yalnızca numpy).

Altyazı belirdiği andan itibaren oyun sesinin kısa bir parçasından:
- temel frekans (perde, F0): kadın/erkek ayrımı (konuşmada erkek ~85-165 Hz, kadın ~165-255 Hz),
- ses rengi: log-mel bant enerjilerinden kepstrum katsayıları (MFCC benzeri) + perde dağılımı.
Bu "ses parmak izi" hafızadaki karakterlerle karşılaştırılır; yakın olan aynı karakter sayılır ve hep
aynı Türkçe sesle (ses, perde, hız) okunur. Oyun müziği/efektleri karıştığı için yaklaşıktır: menüden
karakterin cinsiyeti düzeltilebilir, hafıza silinebilir."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FEMALE_F0 = 165.0  # Hz: bu perdeden yüksek konuşma kadın sayılır
_F0_MIN, _F0_MAX = 70.0, 400.0


@dataclass
class VoiceFeatures:
    f0: float            # konuşma perdesi (Hz, medyan)
    f0_spread: float     # perde dağılımı (oktav, çeyrekler arası)
    timbre: list[float]  # ses rengi (normalize kepstrum)
    voiced_frames: int

    @property
    def gender(self) -> str:
        return "kadin" if self.f0 >= FEMALE_F0 else "erkek"


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - size) // hop)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _pitch(frame: np.ndarray, sr: int) -> float | None:
    """Normalize öz-ilinti ile perde; sesli değilse None.

    Önce merkez kırpma (Sondhi): genliğin %30'unun altı sıfırlanır. Güçlü bir ses yolu rezonansı
    (formant, ör. 400 Hz) aksi hâlde perde sanılıyor ve erkek sesi kadın çıkıyordu."""
    frame = frame - frame.mean()
    peak = float(np.abs(frame).max())
    if peak <= 1e-6:
        return None
    clip = 0.3 * peak
    frame = np.where(frame > clip, frame - clip, np.where(frame < -clip, frame + clip, 0.0))
    energy = float(np.dot(frame, frame))
    if energy <= 1e-9:
        return None
    n = len(frame)
    spec = np.fft.rfft(frame, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac /= ac[0]
    lo, hi = int(sr / _F0_MAX), min(n - 1, int(sr / _F0_MIN))
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.45:  # periyodik değil (gürültü, müzik vuruşu, sessiz harf)
        return None
    # Alt harmoniği seçme hatası (oktav aşağı): yarım gecikmede de güçlü tepe varsa onu al.
    half = lag // 2
    if half >= lo and ac[half] > 0.85 * ac[lag]:
        lag = half
    return sr / lag


def _mel_filters(sr: int, n_fft: int, n_mels: int = 24) -> np.ndarray:
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(60), hz_to_mel(sr / 2 * 0.95), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            fb[m - 1, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[m - 1, k] = (c - k) / max(1, c - b)
    return fb


_MEL_CACHE: dict[tuple[int, int], np.ndarray] = {}


def features(samples: np.ndarray, sr: int = SAMPLE_RATE) -> VoiceFeatures | None:
    """Konuşma parçasının özellikleri. Yeterli sesli (perdeli) bölüm yoksa None."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    size, hop = int(0.04 * sr), int(0.01 * sr)
    if len(x) < size * 3:
        return None
    frames = _frames(x, size, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    if energy.max() <= 1e-4:
        return None
    # Yalnızca yüksek enerjili kareler (konuşma); sessizlik ve uzak efektler atılır.
    loud = energy >= max(energy.max() * 0.25, np.percentile(energy, 40))
    f0s, voiced = [], []
    for i in np.flatnonzero(loud):
        f0 = _pitch(frames[i], sr)
        if f0 is not None:
            f0s.append(f0)
            voiced.append(i)
    if len(f0s) < 6:
        return None
    f0s = np.array(f0s)
    log_f0 = np.log2(f0s)
    f0 = float(2 ** np.median(log_f0))
    spread = float(np.subtract(*np.percentile(log_f0, [75, 25])))

    window = np.hanning(size).astype(np.float32)
    n_fft = 1 << (size - 1).bit_length()
    key = (sr, n_fft)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = _mel_filters(sr, n_fft)
    spec = np.abs(np.fft.rfft(frames[voiced] * window, n_fft)) ** 2
    logmel = np.log(spec @ _MEL_CACHE[key].T + 1e-10)
    n = logmel.shape[1]
    dct = np.cos(np.pi / n * (np.arange(n)[:, None] + 0.5) * np.arange(1, 13)[None, :])
    ceps = (logmel @ dct).mean(axis=0)  # c1..c12 (c0 = ses düzeyi, atılır)
    norm = float(np.linalg.norm(ceps))
    timbre = (ceps / norm).tolist() if norm > 0 else ceps.tolist()
    return VoiceFeatures(f0, spread, timbre, len(f0s))


def distance(a: VoiceFeatures | "Character", b: VoiceFeatures | "Character") -> float:
    """0 = aynı ses. Perde farkı (oktav) ağırlıklı + ses rengi (kosinüs) farkı."""
    pitch = abs(math.log2(a.f0 / b.f0))
    ta, tb = np.asarray(a.timbre), np.asarray(b.timbre)
    timbre = 1.0 - float(np.dot(ta, tb) / (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9))
    return 2.0 * pitch + timbre


# Aynı cinsiyetteki karakterleri birbirinden ayırmak için sırayla verilen ses ayarları.
_VARIANTS = [(0, 0), (-8, -4), (8, 4), (-14, 6), (14, -6), (-4, 10), (4, -10)]  # (perde %, hız %)


@dataclass
class Character:
    id: int
    gender: str               # "kadin" / "erkek" (menüden değiştirilebilir)
    f0: float
    timbre: list[float]
    lines: int = 1
    pitch_pct: int = 0        # bu karakterin sesine eklenen perde (%)
    rate_pct: int = 0         # ve hız (%)
    name: str = ""
    gender_locked: bool = False  # kullanıcı cinsiyeti elle seçti: otomatik değişmez

    @property
    def label(self) -> str:
        return self.name or f"Karakter {self.id}"


@dataclass
class _Book:
    characters: list[Character] = field(default_factory=list)


class CharacterBook:
    """Karakter hafızası: karakter_sesleri.json. İş parçacığı güvenli."""

    MATCH = 0.55       # bu uzaklıktan yakın ses aynı karakter
    LEARN = 0.15       # eşleşince karakterin parmak izi bu oranda güncellenir
    MAX_CHARACTERS = 60

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self.characters: list[Character] = []
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.characters = [Character(**c) for c in data.get("characters", [])]
            except (ValueError, TypeError, OSError):
                self.characters = []

    def identify(self, feats: VoiceFeatures) -> Character:
        with self._lock:
            best, best_d = None, math.inf
            for ch in self.characters:
                d = distance(feats, ch)
                if d < best_d:
                    best, best_d = ch, d
            if best is not None and best_d <= self.MATCH:
                best.lines += 1
                w = self.LEARN
                best.f0 = float(2 ** ((1 - w) * math.log2(best.f0) + w * math.log2(feats.f0)))
                best.timbre = [(1 - w) * a + w * b for a, b in zip(best.timbre, feats.timbre)]
                if not best.gender_locked:
                    best.gender = "kadin" if best.f0 >= FEMALE_F0 else "erkek"
                self._save()
                return best
            gender = feats.gender
            same = sum(1 for c in self.characters if c.gender == gender)
            pitch, rate = _VARIANTS[same % len(_VARIANTS)]
            ch = Character(id=max((c.id for c in self.characters), default=0) + 1, gender=gender, f0=feats.f0,
                           timbre=list(feats.timbre), pitch_pct=pitch, rate_pct=rate)
            self.characters.append(ch)
            if len(self.characters) > self.MAX_CHARACTERS:  # en az konuşanı unut
                self.characters.remove(min(self.characters, key=lambda c: c.lines))
            self._save()
            return ch

    def set_gender(self, char_id: int, gender: str) -> None:
        with self._lock:
            for ch in self.characters:
                if ch.id == char_id:
                    ch.gender, ch.gender_locked = gender, True
            self._save()

    def remove(self, char_id: int) -> None:
        with self._lock:
            self.characters = [c for c in self.characters if c.id != char_id]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self.characters = []
            self._save()

    def snapshot(self) -> list[Character]:
        with self._lock:
            return [Character(**asdict(c)) for c in self.characters]

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"characters": [asdict(c) for c in self.characters]}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass
