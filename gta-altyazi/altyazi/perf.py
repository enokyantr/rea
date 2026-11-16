"""Yük izleme ve kendini kısma: oyun + Remote Play + Discord yayını açıkken bilgisayar zorlanırsa
altyazı programı daha az kaynak kullanır, rahatlayınca tam hıza döner.

Kısma seviyeleri (Pipeline ve sokak dinleyicisi her döngüde okur):
  0: tam hız   — ekran 33 ms'de bir, OCR yalnızca yazı değişince (beklemesiz)
  1: hafif     — ekran 50 ms'de bir, iki OCR arası en az 80 ms
  2: yoğun     — ekran 80 ms'de bir, iki OCR arası en az 150 ms, sokak dinleme duraklar
Mod: "auto" (ölçüme göre), "yayin" (en az 1; ayrıca taslak çeviri 2 çekirdek, sokakta hafif model),
"normal" (hep 0)."""

from __future__ import annotations

import os
import threading
import time
from collections import deque

# seviye -> (ekran yakalama aralığı ms, iki OCR arası en az ms)
LEVELS = {0: (33, 0), 1: (50, 80), 2: (80, 150)}


def decide(level: int, history: list[float], mode: str) -> int:
    """Son ölçümlere (sistem işlemci %, 1 sn aralıklı) göre yeni seviye. Yükselmek hızlı
    (3 sn), düşmek yavaş (6 sn): seviye gidip gelmesin."""
    floor = 1 if mode == "yayin" else 0
    if mode == "normal":
        return 0
    recent3, recent6 = history[-3:], history[-6:]
    if len(recent3) == 3 and min(recent3) >= 92:
        new = 2
    elif len(recent3) == 3 and min(recent3) >= 80:
        new = max(level, 1)
    elif len(recent6) == 6 and max(recent6) < 65:
        new = level - 1
    elif len(recent6) == 6 and max(recent6) < 85 and level == 2:
        new = 1
    else:
        new = level
    return max(floor, min(2, new))


class LoadMonitor(threading.Thread):
    def __init__(self, cfg, log=print):
        super().__init__(daemon=True, name="yuk")
        self.cfg = cfg
        self._log = log
        self.level = 1 if cfg.performance_mode == "yayin" else 0
        self.system_cpu: float | None = None
        self.app_cpu: float | None = None
        self.stopped = threading.Event()
        self._history: deque[float] = deque(maxlen=12)

    @property
    def capture_interval_s(self) -> float:
        return max(self.cfg.capture_interval_ms, LEVELS[self.level][0]) / 1000

    @property
    def min_ocr_gap_s(self) -> float:
        return LEVELS[self.level][1] / 1000

    def describe(self) -> str:
        names = {0: "tam hız", 1: "hafif kısma", 2: "yoğun kısma (sokak dinleme duraklatıldı)"}
        load = "" if self.system_cpu is None else \
            f"sistem işlemci %{self.system_cpu:.0f}, program %{self.app_cpu or 0:.0f} · "
        return f"{load}{names[self.level]} ({self.cfg.performance_mode})"

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        proc = psutil.Process(os.getpid()) if psutil else None
        ncpu = os.cpu_count() or 1
        if psutil:
            psutil.cpu_percent(None)
            proc.cpu_percent(None)
        while not self.stopped.wait(1.0):
            if psutil:
                self.system_cpu = float(psutil.cpu_percent(None))
                self.app_cpu = float(proc.cpu_percent(None)) / ncpu
                self._history.append(self.system_cpu)
            new = decide(self.level, list(self._history), self.cfg.performance_mode)
            if new != self.level:
                self._log(f"Yük: {'kısılıyor' if new > self.level else 'rahatladı'} -> "
                          f"{['tam hız', 'hafif kısma', 'yoğun kısma'][new]}"
                          + (f" (sistem işlemci %{self.system_cpu:.0f})" if self.system_cpu is not None else ""))
                self.level = new

    def stop(self) -> None:
        self.stopped.set()


class _Static:
    """Yük izleyici yokken (testler, --ocr-test): her zaman tam hız."""
    level = 0
    capture_interval_s = 0.033
    min_ocr_gap_s = 0.0


STATIC = _Static()


def now() -> float:
    return time.monotonic()
