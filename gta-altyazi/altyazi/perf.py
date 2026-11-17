"""Yük izleme ve kendini kısma: oyun + Remote Play + Discord yayını açıkken bilgisayar zorlanırsa
altyazı programı daha az kaynak kullanır, rahatlayınca tam hıza döner.

Kısma seviyeleri (Pipeline ve sokak dinleyicisi her döngüde okur):
  0: tam hız   — ekran 33 ms'de bir, OCR yalnızca yazı değişince (beklemesiz)
  1: hafif     — ekran 50 ms'de bir, iki OCR arası en az 80 ms
  2: yoğun     — ekran 80 ms'de bir, iki OCR arası en az 150 ms, sokak dinleme duraklar
Mod: "auto" (ölçüme göre), "yayin" (en az 1; ayrıca taslak çeviri 2 çekirdek, sokakta hafif model),
"oyun" (oyun aynı bilgisayarda: en az 1, ekran kartı doluysa 2; seyrek okuma, bkz. GAME_LEVELS),
"normal" (hep 0).

Ekran kartı (NVIDIA, nvidia-ml-py kuruluysa) kullanımı ve belleği de ölçülür; menüde gösterilir.
Kısma kararına yalnızca "oyun" modunda girer: Remote Play'de ekran kartını dolduran bizim dil
modelimizdir, onu OCR'ı kısarak rahatlatamayız."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

# seviye -> (ekran yakalama aralığı ms, iki OCR arası en az ms)
LEVELS = {0: (33, 0), 1: (50, 80), 2: (80, 150)}
# "oyun" modu: oyun ekran kartını/işlemciyi dolduruyor. Altyazı ekranda en az 1-2 sn kalır; 66-100 ms'de
# bir bakmak yeter (satır en geç ~0,1 sn sonra fark edilir). OCR arası: doğrulama okuması bu kadar
# gecikebilir (emin okumalarda doğrulama beklenmez).
GAME_LEVELS = {0: (33, 0), 1: (66, 120), 2: (100, 250)}
# Ekran kartı 3 sn bu kadar doluysa (oyun) ya da boş belleği bu kadar MB'ın altındaysa yoğun kısma.
GPU_BUSY = 90
GPU_CALM = 80
VRAM_LOW_MB = 700
# Yerel dil modelinin (aya-expanse:8b, num_ctx 2048) ekran kartında kapladığı yaklaşık bellek.
LLM_VRAM_MB = 6000

MODES = ("auto", "yayin", "oyun", "normal")


def decide(level: int, history: list[float], mode: str, gpu_history: list[float] | None = None,
           vram_free_mb: float | None = None) -> int:
    """Son ölçümlere (sistem işlemci %, 1 sn aralıklı) göre yeni seviye. Yükselmek hızlı
    (3 sn), düşmek yavaş (6 sn): seviye gidip gelmesin. "oyun" modunda ekran kartı kullanımı
    (`gpu_history`, %) ve boş ekran kartı belleği de hesaba katılır."""
    if mode == "normal":
        return 0
    floor = 1 if mode in ("yayin", "oyun") else 0
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
    if mode == "oyun" and (gpu_history or vram_free_mb is not None):
        gpu = gpu_history or []
        vram_low = vram_free_mb is not None and vram_free_mb < VRAM_LOW_MB
        if vram_low or (len(gpu) >= 3 and min(gpu[-3:]) >= GPU_BUSY):
            new = 2
        elif not (len(gpu) >= 6 and max(gpu[-6:]) < GPU_CALM):
            new = max(new, level)  # ekran kartı henüz rahatlamadı: düşme
    return max(floor, min(2, new))


class GpuSampler:
    """NVIDIA ekran kartı kullanımı (%) ve belleği (MB), nvidia-ml-py ile. Kurulu değilse ya da
    NVIDIA kartı yoksa `available` False olur ve yük izleme yalnızca işlemciye bakar."""

    def __init__(self):
        self._nvml = None
        self._handle = None
        self.name = ""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def sample(self) -> tuple[float, float, float] | None:
        """(kullanım %, kullanılan MB, toplam MB) ya da ölçülemezse None."""
        if self._nvml is None:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return float(util), mem.used / 2**20, mem.total / 2**20
        except Exception:
            return None


def vram_free_mb() -> float | None:
    """Başlangıçta tek seferlik ölçüm (boş ekran kartı belleği, MB). Ölçülemezse None."""
    sample = GpuSampler().sample()
    return None if sample is None else sample[2] - sample[1]


@dataclass(frozen=True)
class GamePolicy:
    """Aynı bilgisayarda oyun varken ağır motorların yeri (başlangıçta bir kez karar verilir)."""

    llm: str                   # "gpu" (bugünkü gibi) | "cpu" (num_gpu=0) | "kapali" (taslak son çeviri)
    num_gpu: int | None        # OllamaBackend'e: ekran kartındaki katman sayısı (None: Ollama seçer)
    ocr_device: str            # RapidOCR: "auto" (DirectML) | "cpu"
    note: str = ""             # kayda yazılacak açıklama


def game_policy(cfg, free_mb: float | None = None) -> GamePolicy:
    """`performance_mode` "oyun" değilse hiçbir şey değişmez (yalnızca belleği az görürse uyarı).
    "oyun": `game_llm` ve `game_ocr_device` ayarları uygulanır. `free_mb`: dil modeli hariç boş
    ekran kartı belleği (bilinmiyorsa None)."""
    if cfg.performance_mode != "oyun":
        note = ""
        if (free_mb is not None and cfg.final_translator == "ollama"
                and cfg.translation_mode in ("hybrid", "final") and free_mb < LLM_VRAM_MB):
            note = (f"Ekran kartında {free_mb / 1024:.1f} GB boş; yerel dil modeli ~{LLM_VRAM_MB / 1024:.0f} GB "
                    "ister. Oyun bu bilgisayarda açıksa menü > OCR ve hız > Performans: oyun seçin.")
        return GamePolicy("gpu", None, cfg.ocr_device, note)
    llm = cfg.game_llm if cfg.game_llm in ("gpu", "cpu", "kapali") else "kapali"
    ocr_device = "cpu" if cfg.game_ocr_device == "cpu" else cfg.ocr_device
    ocr_note = " OCR işlemcide." if ocr_device == "cpu" else ""
    if cfg.final_translator != "ollama":
        return GamePolicy("gpu", None, ocr_device, ("Oyun modu:" + ocr_note) if ocr_note else "")
    notes = {
        "kapali": "Oyun modu: yerel dil modeli yüklenmiyor (ekran kartı belleği oyuna kalır); "
                  "taslak çeviri son çeviri olarak gösterilir.",
        "cpu": "Oyun modu: yerel dil modeli işlemcide (ekran kartı belleği oyuna kalır; satır başına "
               "birkaç saniye sürebilir, işlemciyi yorar).",
        "gpu": "Oyun modu: yerel dil modeli ekran kartında (oyunla paylaşır; oyun takılırsa "
               "\"kapali\" seçin).",
    }
    return GamePolicy(llm, 0 if llm == "cpu" else None, ocr_device, notes[llm] + ocr_note)


class LoadMonitor(threading.Thread):
    def __init__(self, cfg, log=print, gpu: GpuSampler | None = None):
        super().__init__(daemon=True, name="yuk")
        self.cfg = cfg
        self._log = log
        self._gpu = gpu
        self.level = 1 if cfg.performance_mode in ("yayin", "oyun") else 0
        self.system_cpu: float | None = None
        self.app_cpu: float | None = None
        self.gpu_util: float | None = None
        self.vram_used_mb: float | None = None
        self.vram_total_mb: float | None = None
        self.stopped = threading.Event()
        self._history: deque[float] = deque(maxlen=12)
        self._gpu_history: deque[float] = deque(maxlen=12)

    @property
    def _table(self) -> dict[int, tuple[int, int]]:
        return GAME_LEVELS if self.cfg.performance_mode == "oyun" else LEVELS

    @property
    def capture_interval_s(self) -> float:
        return max(self.cfg.capture_interval_ms, self._table[self.level][0]) / 1000

    @property
    def min_ocr_gap_s(self) -> float:
        return self._table[self.level][1] / 1000

    @property
    def vram_free_mb(self) -> float | None:
        if self.vram_used_mb is None or self.vram_total_mb is None:
            return None
        return self.vram_total_mb - self.vram_used_mb

    def describe(self) -> str:
        names = {0: "tam hız", 1: "hafif kısma", 2: "yoğun kısma (sokak dinleme duraklatıldı)"}
        load = "" if self.system_cpu is None else \
            f"sistem işlemci %{self.system_cpu:.0f}, program %{self.app_cpu or 0:.0f} · "
        if self.gpu_util is not None:
            load += (f"ekran kartı %{self.gpu_util:.0f}, bellek {self.vram_used_mb / 1024:.1f}/"
                     f"{self.vram_total_mb / 1024:.1f} GB · ")
        return f"{load}{names[self.level]} ({self.cfg.performance_mode})"

    def sample_once(self, psutil=None, proc=None, ncpu: int = 1) -> None:
        if psutil:
            self.system_cpu = float(psutil.cpu_percent(None))
            self.app_cpu = float(proc.cpu_percent(None)) / ncpu
            self._history.append(self.system_cpu)
        gpu = self._gpu.sample() if self._gpu is not None else None
        if gpu is not None:
            self.gpu_util, self.vram_used_mb, self.vram_total_mb = gpu
            self._gpu_history.append(self.gpu_util)
        new = decide(self.level, list(self._history), self.cfg.performance_mode,
                     list(self._gpu_history), self.vram_free_mb)
        if new != self.level:
            why = []
            if self.system_cpu is not None:
                why.append(f"sistem işlemci %{self.system_cpu:.0f}")
            if self.gpu_util is not None:
                why.append(f"ekran kartı %{self.gpu_util:.0f}, boş bellek {self.vram_free_mb:.0f} MB")
            self._log(f"Yük: {'kısılıyor' if new > self.level else 'rahatladı'} -> "
                      f"{['tam hız', 'hafif kısma', 'yoğun kısma'][new]}"
                      + (f" ({'; '.join(why)})" if why else ""))
            self.level = new

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        if self._gpu is None:
            self._gpu = GpuSampler()
        proc = psutil.Process(os.getpid()) if psutil else None
        ncpu = os.cpu_count() or 1
        if psutil:
            psutil.cpu_percent(None)
            proc.cpu_percent(None)
        while not self.stopped.wait(1.0):
            self.sample_once(psutil, proc, ncpu)

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
