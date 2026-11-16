"""Sokak konuşmaları: altyazısı olmayan NPC konuşmalarını oyun sesinden yazıya çevirir.

Akış (arka plan iş parçacığı):
1. Segmenter: oyun sesinde konuşma parçalarını ses gücüyle bulur (ortam gürültüsü sürekli ölçülür;
   konuşma onun belirgin üstündeyse parça başlar, ~0,5 sn sessizlikte biter).
2. Engel: parça sırasında ekranda altyazı vardıysa (o diyalog zaten OCR ile çevriliyor) ya da kendi
   dublajımız çalıyorduysa parça atılır.
3. faster-whisper (yerel, işlemcide, ücretsiz) ile İngilizce yazıya çevrilir. "Konuşma yok" olasılığı
   yüksek, düşük güvenli ya da müzik işaretli (♪, [Music]) sonuçlar atılır: radyo şarkıları çevrilmesin.
4. Çıkan cümle normal çeviri yoluna girer ve ekranın üstünde "Sokak:" diye kısa süre görünür."""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .voiceid import SAMPLE_RATE

BLOCK_S = 0.05


@dataclass
class Segment:
    start: float  # time.monotonic
    end: float
    samples: np.ndarray


class Segmenter:
    """Ses gücüne dayalı konuşma parçalayıcı (blok başına < 0,1 ms)."""

    START_BLOCKS = 3      # konuşma başlaması için art arda yüksek blok (150 ms)
    END_SILENCE_S = 0.5   # bu kadar sessizlik parçayı bitirir
    MIN_S, MAX_S = 0.6, 8.0
    PAD_BLOCKS = 4        # parçanın başına eklenen önceki bloklar (ilk heceyi kaçırmamak için)

    def __init__(self):
        self._floor = 1e-3
        self._recent: list[np.ndarray] = []
        self._current: list[np.ndarray] = []
        self._start = 0.0
        self._loud_run = 0
        self._quiet_s = 0.0
        self._active = False

    def feed(self, stamp: float, block: np.ndarray) -> Segment | None:
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if len(block) else 0.0
        dur = len(block) / SAMPLE_RATE
        loud = rms > max(3.0 * self._floor, 0.004)
        if not self._active:
            # Ortam gürültüsü: yalnızca konuşma dışındayken, yavaşça (yukarı daha yavaş) izlenir.
            rate = 0.02 if rms > self._floor else 0.1
            self._floor = max(1e-4, (1 - rate) * self._floor + rate * rms)
            self._recent.append(block)
            self._recent = self._recent[-(self.PAD_BLOCKS + self.START_BLOCKS):]
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.START_BLOCKS:
                self._active = True
                self._current = list(self._recent)
                self._start = stamp - dur * len(self._current)
                self._quiet_s = 0.0
            return None
        self._current.append(block)
        self._quiet_s = 0.0 if loud else self._quiet_s + dur
        length = stamp - self._start
        if self._quiet_s >= self.END_SILENCE_S or length >= self.MAX_S:
            samples = np.concatenate(self._current)
            start, self._active, self._current, self._loud_run, self._recent = self._start, False, [], 0, []
            if length - self._quiet_s < self.MIN_S:
                return None
            return Segment(start, stamp, samples)
        return None


_MUSIC = re.compile(r"[♪♫]|\[(?:music|singing|song|applause|laughter|inaudible|.*?)\]|\((?:music|singing).*?\)",
                    re.IGNORECASE)
_HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "so", "okay."}


def accept(text: str, no_speech_prob: float, avg_logprob: float) -> str:
    """Whisper sonucunu süz: kabul edilen temiz metin ya da "" (atılır)."""
    text = (text or "").strip()
    if not text or _MUSIC.search(text):
        return ""
    if no_speech_prob > 0.6 or avg_logprob < -1.0:
        return ""
    if text.lower() in _HALLUCINATIONS or len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
        return ""
    return text


class StreetListener(threading.Thread):
    """Pipeline ile aynı arayüz (seq, paused, stopped, reset, base...): app aynı şekilde yönetir."""

    def __init__(self, cfg, name: str, get_recorder: Callable[[], object], translator, ui: queue.Queue,
                 blocked: Callable[[float, float], bool], log: Callable[[str], None]):
        super().__init__(daemon=True, name="sokak")
        self.cfg = cfg
        self.region_name = name
        self.get_recorder = get_recorder
        self.translator = translator
        self.ui = ui
        self.blocked = blocked
        self._log = log
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.reset = threading.Event()
        self.seq = 0
        self.base = None
        self.last_read_at = 0.0
        self.engine_name = ""
        self.dump_dir = None
        self.heard = 0       # yazıya çevrilen parça sayısı
        self.skipped = 0     # altyazı/dublaj yüzünden atlanan
        self._model = None
        self._model_name = ""

    def _load_model(self):
        name = self.cfg.street_model
        if self._model is not None and self._model_name == name:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.engine_name = "faster-whisper kurulu değil (guncelle.bat)"
            return None
        self._log(f"Sokak konuşmaları: konuşma tanıma modeli yükleniyor ({name}; ilk seferde indirilir)...")
        try:
            self._model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2)
        except Exception as exc:
            self.engine_name = f"model yüklenemedi: {exc}"
            self._log(f"Sokak konuşmaları: {self.engine_name}")
            return None
        self._model_name = name
        self.engine_name = f"Whisper {name} (işlemci)"
        self._log(f"Sokak konuşmaları hazır: {self.engine_name}")
        return self._model

    def run(self) -> None:
        sub, rec = None, None
        seg = Segmenter()
        while not self.stopped.is_set():
            if not self.cfg.street_enabled or self.paused.is_set():
                if sub is not None and rec is not None:
                    rec.unsubscribe(sub)
                    sub, rec = None, None
                time.sleep(0.2)
                continue
            if sub is None:
                rec = self.get_recorder()
                if rec is None or rec.error:
                    time.sleep(1.0)
                    continue
                sub = rec.subscribe()
                seg = Segmenter()
                if self._load_model() is None:
                    self.cfg.street_enabled = False
                    continue
            try:
                stamp, block = sub.get(timeout=0.5)
            except queue.Empty:
                continue
            self.last_read_at = time.monotonic()
            segment = seg.feed(stamp, block)
            if segment is None:
                continue
            if self.blocked(segment.start, segment.end):
                self.skipped += 1
                continue
            self._transcribe(segment)
        if sub is not None and rec is not None:
            rec.unsubscribe(sub)

    def _transcribe(self, segment: Segment) -> None:
        t0 = time.perf_counter()
        try:
            parts, _info = self._model.transcribe(
                segment.samples.astype(np.float32), language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False, without_timestamps=True)
            texts = [accept(p.text, p.no_speech_prob, p.avg_logprob) for p in parts]
        except Exception as exc:
            self._log(f"Sokak konuşması tanınamadı: {exc}")
            return
        text = " ".join(t for t in texts if t).strip()
        if not text:
            return
        self.heard += 1
        took = time.perf_counter() - t0
        self.seq += 1
        seq = self.seq
        first_seen = segment.end - 0.0  # gecikme konuşmanın bittiği andan ölçülür
        self._log(f"[Sokak] EN ({1000 * took:.0f} ms): {text}")

        def on_result(stage: str, translated: str, _took: float) -> None:
            self.ui.put(("show", self.region_name, seq, stage, translated, text, first_seen, None))

        def on_error(exc: Exception, stage: str, draft_shown: bool) -> None:
            if draft_shown:
                self.ui.put(("draft-final", self.region_name, seq))

        self.translator.submit(text, None, lambda: seq == self.seq, on_result, on_error)
