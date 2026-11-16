"""Oyun sesini dinleme: varsayılan hoparlöre giden sesin kopyası (WASAPI loopback, `soundcard`).

Son birkaç saniye halka arabellekte, zaman damgasıyla (time.monotonic) tutulur; altyazı belirdiği
andan itibaren konuşanın sesi buradan alınır. Hiçbir şey diske yazılmaz ya da gönderilmez."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

import numpy as np

from .voiceid import SAMPLE_RATE

BLOCK = 800  # 50 ms


class LoopbackRecorder(threading.Thread):
    def __init__(self, seconds: float = 6.0, log=print):
        super().__init__(daemon=True, name="loopback")
        self._blocks: deque[tuple[float, np.ndarray]] = deque(maxlen=int(seconds * SAMPLE_RATE / BLOCK))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._log = log
        self.error: str | None = None
        self.device = ""
        self.ready = threading.Event()
        self._subscribers: list[queue.Queue] = []

    def subscribe(self, max_blocks: int = 200) -> "queue.Queue[tuple[float, np.ndarray]]":
        """Her yeni blok (zaman damgası, ses) bu kuyruğa da gelir (sokak konuşmaları için).
        Kuyruk dolarsa en eskiler atılır: dinleyici geride kalsa da kaydedici beklemez."""
        q: queue.Queue = queue.Queue(maxsize=max_blocks)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            self.device = speaker.name
        except Exception as exc:
            self.error = f"Oyun sesi dinlenemiyor ({exc}); konuşan tanıma kapalı."
            self.ready.set()
            return
        self.ready.set()
        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK)
                    now = time.monotonic()
                    block = np.asarray(data, dtype=np.float32).reshape(-1)
                    with self._lock:
                        self._blocks.append((now, block))
                        subscribers = list(self._subscribers)
                    for q in subscribers:
                        if q.full():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                        q.put_nowait((now, block))
        except Exception as exc:  # cihaz değişti/çıkarıldı
            self.error = f"Oyun sesi dinleme durdu: {exc}"
            self._log(self.error)

    def window(self, start: float, end: float) -> np.ndarray:
        """[start, end] (time.monotonic) aralığındaki ses."""
        with self._lock:
            blocks = list(self._blocks)
        parts = []
        for stamp, block in blocks:
            block_start = stamp - len(block) / SAMPLE_RATE
            if stamp < start or block_start > end:
                continue
            parts.append(block)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
