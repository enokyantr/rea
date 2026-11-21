"""Sokak konuşmaları: altyazısı olmayan NPC konuşmalarını oyun sesinden yazıya çevirir.

Akış (arka plan iş parçacığı):
1. Parçalama: ses gücü ucuz ilk eleme; geçen 32 ms'lik çerçevelerde Silero VAD (faster-whisper'ın
   içinde gelen ONNX modeli, durumu çerçeveden çerçeveye taşınır) konuşma olup olmadığına karar verir.
   Konuşma ~0,5 sn kesilince parça biter; motor/silah/müzik gibi yüksek ama konuşma olmayan ses
   parçayı uzatmaz. Model yoksa eski ses gücü parçalayıcısı kullanılır.
2. Erken tanıma: konuşma ~0,2 sn kesilince parça yazıya çevrilmeye başlar; konuşma 0,5 sn dolmadan
   sürerse o sonuç atılır, sürmezse kullanılır (~0,3 sn kazanç). Aynı anda en çok bir tanıma çalışır.
3. Engel: parça sırasında ekranda altyazı vardıysa (o diyalog zaten OCR ile çevriliyor) ya da kendi
   dublajımız çalıyorduysa parça atılır.
4. faster-whisper (yerel, işlemcide, ücretsiz) ile İngilizce yazıya çevrilir; oyundaki adlar kısa bir
   istemle önerilir. "Konuşma yok" olasılığı yüksek, düşük güvenli, tekrara düşmüş, müzik işaretli
   (♪, [Music]) ya da yalnızca istemi tekrarlayan sonuçlar atılır: radyo şarkıları çevrilmesin.
5. Çıkan cümle normal çeviri yoluna girer ve ekranın üstünde "Sokak:" diye kısa süre görünür."""

from __future__ import annotations

import importlib.util
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .voiceid import SAMPLE_RATE

BLOCK_S = 0.05
RATE = 16000  # Whisper ve Silero 16 kHz bekler


def to_16k(block: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Kaydedici 16 kHz dışında çalışırsa doğrusal yeniden örnekleme (konuşma tanımaya yeterli)."""
    if rate == RATE or not len(block):
        return block
    n = max(1, round(len(block) * RATE / rate))
    return np.interp(np.arange(n) * (rate / RATE), np.arange(len(block)), block).astype(np.float32)


@dataclass
class Segment:
    start: float  # time.monotonic
    end: float    # konuşmanın bittiği an
    samples: np.ndarray
    final: bool = True  # False: erken (konuşma ~0,2 sn kesildi; parça henüz kapanmadı)
    key: tuple = ()     # erken ve son parça aynı sesse anahtarları eşittir


class Segmenter:
    """Ses gücüne dayalı konuşma parçalayıcı (blok başına < 0,1 ms). Silero yoksa kullanılır."""

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
        self.pending_key = None  # erken parça üretmez

    def feed(self, stamp: float, block: np.ndarray) -> Segment | None:
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if len(block) else 0.0
        dur = len(block) / RATE
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
            return Segment(start, stamp, samples, key=(start,))
        return None


# --- Silero VAD -----------------------------------------------------------------------------------

def silero_model_path() -> Path | None:
    """faster-whisper paketinin içinde gelen Silero VAD modeli (paketi içe aktarmadan bulunur)."""
    try:
        spec = importlib.util.find_spec("faster_whisper")
    except (ImportError, ValueError):
        return None
    for base in (spec.submodule_search_locations or []) if spec else []:
        found = sorted((Path(base) / "assets").glob("silero_vad*.onnx"), reverse=True)  # yeni sürüm önce
        if found:
            return found[0]
    return None


class SileroVad:
    """Silero VAD'in akış hâli: her çağrı bir 512 örneklik (32 ms) çerçeve alır, LSTM durumu ve önceki
    çerçevenin son 64 örneği (bağlam) sonraki çağrıya taşınır. faster-whisper'ın kendi sarmalayıcısı
    her çağrıda durumu sıfırladığı için akışta kullanılamaz. Çerçeve başına ~0,2 ms (tek iş parçacığı)."""

    FRAME, CONTEXT = 512, 64

    def __init__(self, path: Path):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        from .ocr import ORT_INIT_LOCK

        with ORT_INIT_LOCK:  # OCR oturumlarıyla eşzamanlı kurulmasın (bkz. ocr.ORT_INIT_LOCK)
            self._session = onnxruntime.InferenceSession(str(path), sess_options=opts,
                                                         providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        self._audio_name = inputs[0].name
        # Eski modellerde ayrıca örnekleme hızı girdisi var; durum girdileri (h/c ya da state) çıktılarla
        # aynı sırada döner.
        self._rate_name = next((i.name for i in inputs if i.name == "sr"), None)
        self._state_shapes = {i.name: tuple(d if isinstance(d, int) else 1 for d in i.shape)
                              for i in inputs[1:] if i.name != "sr"}
        self._input = np.zeros((1, self.CONTEXT + self.FRAME), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._input[:] = 0.0
        self._state = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()}

    def __call__(self, frame: np.ndarray) -> float:
        self._input[0, self.CONTEXT:] = frame
        feeds = {self._audio_name: self._input, **self._state}
        if self._rate_name:
            feeds[self._rate_name] = np.array(RATE, dtype=np.int64)
        out = self._session.run(None, feeds)
        self._state = dict(zip(self._state_shapes, out[1:]))
        self._input[0, :self.CONTEXT] = frame[-self.CONTEXT:]
        return float(np.ravel(out[0])[0])


class VadSegmenter:
    """Silero ile konuşma parçalayıcı. Ses gücü ucuz ilk elemedir: çok sessiz çerçeve modele gitmez.

    Konuşma: olasılık >= ON ile başlar, süren konuşmada >= OFF yeter (kararsız çerçevede titremesin).
    Parça konuşma kesildikten END_SILENCE_S sonra biter; arada ses ne kadar yüksek olursa olsun
    (motor, silah, müzik) konuşma değilse süre işler. EARLY_S'de aynı ses erken parça olarak verilir."""

    FRAME = SileroVad.FRAME
    GATE_RMS = 0.002       # bunun altı konuşma sayılmaz (model çalışmaz)
    ON, OFF = 0.5, 0.35
    START_FRAMES = 3       # konuşma başlaması için art arda konuşma çerçevesi (~100 ms)
    PAD_S = 0.2            # parçanın başına eklenen önceki ses (ilk heceyi kaçırmamak için)
    TAIL_S = 0.15          # son konuşma çerçevesinden sonra eklenen ses
    EARLY_S = 0.2          # konuşma bu kadar kesilince erken tanıma
    END_SILENCE_S = 0.5    # bu kadar kesilince parça biter
    MIN_SPEECH_S = 0.4     # daha kısa konuşma (tek ünlem, patlama) atılır
    MAX_S = 8.0
    HISTORY_S = 1.0        # parça dışında saklanan ses (baş payı için)

    def __init__(self, vad: Callable[[np.ndarray], float]):
        self._vad = vad
        self._stale = True        # model durumu son çerçeveyi görmedi: yeniden başlarken sıfırlanır
        self._chunks: list[np.ndarray] = []
        self._chunks_at = 0       # ilk parçanın mutlak örnek konumu
        self._pos = 0             # şimdiye kadar gelen örnek sayısı
        self._framed = 0          # çerçevelenmiş örnek sayısı (sonrası bir sonraki bloğu bekler)
        self._pending = np.zeros(0, dtype=np.float32)
        self._clock = (0.0, 0)    # (zaman damgası, o andaki örnek konumu): konum -> zaman
        self._speech = False
        self._run, self._run_at = 0, 0
        self._active = False
        self._seg_at = self._first_speech = self._last_speech = 0
        self._early_sent = False
        self._seg_id = 0
        self.pending_key: tuple | None = None  # geçerli erken parçanın anahtarı (konuşma sürerse None)

    def _time(self, pos: int) -> float:
        stamp, at = self._clock
        return stamp - (at - pos) / RATE

    def _audio(self, a: int, b: int) -> np.ndarray:
        a = max(a, self._chunks_at)
        joined = np.concatenate(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
        return joined[a - self._chunks_at:b - self._chunks_at]

    def _trim(self) -> None:
        keep = self._seg_at if self._active else self._pos - int(self.HISTORY_S * RATE)
        while len(self._chunks) > 1 and self._chunks_at + len(self._chunks[0]) <= keep:
            self._chunks_at += len(self._chunks.pop(0))

    def _segment(self, end: int, final: bool) -> Segment:
        key = (self._seg_id, end)
        samples = self._audio(self._seg_at, end)
        return Segment(self._time(self._seg_at), self._time(min(end, self._last_speech)), samples, final, key)

    def feed(self, stamp: float, block: np.ndarray) -> Segment | None:
        block = np.asarray(block, dtype=np.float32)
        self._chunks.append(block)
        self._pos += len(block)
        self._clock = (stamp, self._pos)
        frames = np.concatenate([self._pending, block]) if len(self._pending) else block
        out = None
        i = 0
        while i + self.FRAME <= len(frames):
            event = self._frame(frames[i:i + self.FRAME], self._framed + self.FRAME)
            self._framed += self.FRAME
            i += self.FRAME
            if event is not None and (out is None or not out.final):
                out = event  # aynı blokta son parça erkenin önüne geçer
        self._pending = frames[i:].copy()
        self._trim()
        return out

    def _frame(self, frame: np.ndarray, end: int) -> Segment | None:
        if float(np.sqrt(np.mean(frame * frame))) < self.GATE_RMS:
            prob, self._stale = 0.0, True
        else:
            if self._stale:
                reset = getattr(self._vad, "reset", None)
                if reset is not None:
                    reset()
                self._stale = False
            prob = self._vad(frame)
        self._speech = prob >= (self.OFF if self._speech else self.ON)
        if self._speech:
            if self._run == 0:
                self._run_at = end - self.FRAME
            self._run += 1
            self._last_speech = end
            self._early_sent = False
            self.pending_key = None  # konuşma sürdü: erken sonuç geçersiz
            if not self._active and self._run >= self.START_FRAMES:
                self._active = True
                self._seg_id += 1
                self._seg_at = max(self._chunks_at, self._run_at - int(self.PAD_S * RATE))
                self._first_speech = self._run_at
        else:
            self._run = 0
        if not self._active:
            return None
        quiet = (end - self._last_speech) / RATE
        tail = min(end, self._last_speech + int(self.TAIL_S * RATE))
        enough = (self._last_speech - self._first_speech) / RATE >= self.MIN_SPEECH_S
        if quiet >= self.END_SILENCE_S:
            self._active, self.pending_key = False, None
            return self._segment(tail, True) if enough else None
        if end - self._seg_at >= self.MAX_S * RATE:  # bitmeyen konuşma/şarkı: kesilir
            seg = self._segment(end, True)
            self._active, self._run, self.pending_key = False, 0, None
            return seg if enough else None
        if quiet >= self.EARLY_S and not self._early_sent and enough:
            self._early_sent = True
            seg = self._segment(tail, False)
            self.pending_key = seg.key
            return seg
        return None


def make_segmenter(log: Callable[[str], None] | None = None):
    """Silero modeli bulunursa VadSegmenter, yoksa ses gücü parçalayıcısı."""
    path = silero_model_path()
    if path is not None:
        try:
            return VadSegmenter(SileroVad(path))
        except Exception as exc:  # onnxruntime yok ya da model açılamadı
            if log:
                log(f"Sokak konuşmaları: Silero VAD açılamadı ({exc}); ses gücüyle parçalanıyor.")
    elif log:
        log("Sokak konuşmaları: Silero VAD modeli bulunamadı; ses gücüyle parçalanıyor.")
    return Segmenter()


# --- Whisper sonucu ---------------------------------------------------------------------------------

_MUSIC = re.compile(r"[♪♫]|\[(?:music|singing|song|applause|laughter|inaudible|.*?)\]|\((?:music|singing).*?\)",
                    re.IGNORECASE)
_HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "so", "okay."}
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def accept(text: str, no_speech_prob: float, avg_logprob: float, compression_ratio: float = 0.0) -> str:
    """Whisper sonucunu süz: kabul edilen temiz metin ya da "" (atılır). Sıkıştırma oranı > 2,4:
    Whisper aynı sözü tekrar tekrar yazmış (sıcaklık yedeği kapalı olduğundan burada elenir)."""
    text = (text or "").strip()
    if not text or _MUSIC.search(text):
        return ""
    if no_speech_prob > 0.6 or avg_logprob < -1.0 or compression_ratio > 2.4:
        return ""
    if text.lower() in _HALLUCINATIONS or len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
        return ""
    return text


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold().replace("’", "'"))


PROMPT_CHARS = 200


def name_prompt(user: dict[str, str], bundled: dict[str, str], limit: int = PROMPT_CHARS) -> str:
    """Whisper'a oyundaki adları öneren kısa istem: "Jason Duval, Lucia Caminos, Vice City, ...".

    Yalnızca özel adlar: çevrilmeden kalan (kaynak = karşılık) ve büyük harfle başlayan terimler.
    Kullanıcının adları önce gelir. Başka bir adın içinde kalan ad (Jason / Jason Duval) eklenmez."""
    names: list[str] = []
    seen: set[str] = set()
    for entries in (user, bundled):
        for src, dst in entries.items():
            src = " ".join(src.split())
            if src[:1].isupper() and src.casefold() == dst.strip().casefold() and src.casefold() not in seen:
                seen.add(src.casefold())
                names.append(src)
    word_sets = [set(_words(n)) for n in names]
    kept = [n for n, ws in zip(names, word_sets)
            if ws and not any(ws < other for other in word_sets)]
    prompt = ""
    for name in kept:
        candidate = f"{prompt}, {name}" if prompt else name
        if len(candidate) > limit:
            break
        prompt = candidate
    return prompt


def is_prompt_echo(text: str, prompt: str) -> bool:
    """Whisper gürültüde istemi tekrarlar: metnin (neredeyse) tüm kelimeleri istemdeyse atılır."""
    words = _words(text)
    if not words or not prompt:
        return False
    vocab = set(_words(prompt))
    inside = sum(w in vocab for w in words)
    return inside == len(words) or (inside >= 3 and inside >= 0.8 * len(words))


def _app_dir() -> Path:
    # __main__.APP_DIR ile aynı (kullanıcı dosyaları programın klasöründe).
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


def load_prompt(cfg, glossary_path: Path | None, log: Callable[[str], None]) -> str:
    """Programla gelen oyun adları + kullanıcının sozluk.txt'sindeki adlardan istem."""
    from . import glossary

    _general, bundled = glossary.load_bundled()
    user: dict[str, str] = {}
    path = glossary_path
    if path is None and getattr(cfg, "glossary_file", ""):
        path = Path(cfg.glossary_file)
        path = path if path.is_absolute() else _app_dir() / path
    if path is not None and path.is_file():
        try:
            user = glossary.parse(path.read_text(encoding="utf-8-sig"), source=path.name)
        except (OSError, glossary.GlossaryError) as exc:
            log(f"Sokak konuşmaları: sözlükteki adlar okunamadı ({exc})")
    return name_prompt(user, bundled)


# --- dinleyici ----------------------------------------------------------------------------------

class _Job:
    """Bir parçanın tanınması. Erken iş, son parça gelip onaylayana kadar gösterilmez."""

    def __init__(self, segment: Segment, confirmed: bool):
        self.segment = segment
        self.confirmed = confirmed
        self.cancelled = False
        self.done = False
        self.emitted = False
        self.text = ""
        self.took = 0.0


class StreetListener(threading.Thread):
    """Pipeline ile aynı arayüz (seq, paused, stopped, reset, base...): app aynı şekilde yönetir."""

    MAX_WAITING = 2  # sırada bekleyen son parça (fazlası en eskiden atılır: gecikme birikmesin)

    def __init__(self, cfg, name: str, get_recorder: Callable[[], object], translator, ui: queue.Queue,
                 blocked: Callable[[float, float], bool], log: Callable[[str], None],
                 glossary_path: Path | None = None):
        super().__init__(daemon=True, name="sokak")
        self.cfg = cfg
        self.region_name = name
        self.get_recorder = get_recorder
        self.translator = translator
        self.ui = ui
        self.blocked = blocked
        self._log = log
        self.glossary_path = glossary_path
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.reset = threading.Event()
        self.seq = 0
        self.base = None
        self.last_read_at = 0.0
        self.engine_name = ""
        self.dump_dir = None
        self.heard = 0       # yazıya çevrilen parça sayısı
        self.skipped = 0     # altyazı/dublaj/yük yüzünden atlanan
        self.early_used = 0  # erken tanıması kullanılan parça
        self._model = None
        self._model_name = ""
        self._vad = False    # Silero ile mi parçalanıyor (Whisper'ın kendi VAD'i gereksiz)
        self.prompt = ""
        self._lock = threading.Lock()
        self._jobs: list[_Job] = []   # sıradaki/çalışan işler (ilk eleman çalışıyor olabilir)
        self._wake = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None
        self._early: _Job | None = None
        from .perf import STATIC

        self.load = STATIC  # app yük izleyiciyi bağlar

    def _load_model(self):
        # Yayın modunda hafif model: Discord/OBS kodlaması ve oyun işlemciye kalsın.
        name = "base.en" if self.cfg.performance_mode == "yayin" else self.cfg.street_model
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
        self.prompt = load_prompt(self.cfg, self.glossary_path, self._log)
        self.engine_name = f"Whisper {name} (işlemci)"
        self._log(f"Sokak konuşmaları hazır: {self.engine_name}"
                  + (f" · ad istemi: {self.prompt}" if self.prompt else ""))
        return self._model

    def _new_segmenter(self):
        seg = make_segmenter(self._log)
        self._vad = isinstance(seg, VadSegmenter)
        return seg

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True, name="sokak-tanima")
            self._worker.start()

    def run(self) -> None:
        self._ensure_worker()
        sub, rec = None, None
        seg = None
        while not self.stopped.is_set():
            if not self.cfg.street_enabled or self.paused.is_set():
                if sub is not None and rec is not None:
                    rec.unsubscribe(sub)
                    sub, rec = None, None
                    self._drop_early()
                time.sleep(0.2)
                continue
            if sub is None:
                rec = self.get_recorder()
                if rec is None or rec.error:
                    time.sleep(1.0)
                    continue
                sub = rec.subscribe()
                seg = self._new_segmenter()
                if self._load_model() is None:
                    self.cfg.street_enabled = False
                    continue
            try:
                stamp, block = sub.get(timeout=0.5)
            except queue.Empty:
                continue
            self.last_read_at = time.monotonic()
            self.on_segment(seg.feed(stamp, to_16k(block)), seg.pending_key)
        if sub is not None and rec is not None:
            rec.unsubscribe(sub)
        with self._lock:
            for job in self._jobs:
                job.cancelled = True
            self._jobs.append(None)  # type: ignore[arg-type]  # işçiyi durdur
            self._wake.notify()

    # --- parça -> iş ---------------------------------------------------------------------------

    def _gated(self, segment: Segment) -> bool:
        return self.blocked(segment.start, segment.end) or self.load.level >= 2

    def _drop_early(self) -> None:
        with self._lock:
            if self._early is not None and not self._early.confirmed:
                self._early.cancelled = True
            self._early = None

    def on_segment(self, segment: Segment | None, pending_key) -> None:
        """Parçalayıcının çıktısını işler (her blokta çağrılır). Erken iş, konuşma sürünce iptal."""
        early = self._early
        if early is not None and segment is None and pending_key != early.segment.key:
            self._drop_early()  # konuşma sürdü: erken sonuç kullanılmayacak
            return
        if segment is None:
            return
        if not segment.final:
            self._drop_early()
            with self._lock:
                idle = not self._jobs
            # Erken tanıma yalnızca boştayken: sıradaki son parçaları geciktirmesin.
            if idle and not self._gated(segment):
                self._early = self._enqueue(_Job(segment, confirmed=False))
            return
        if self._gated(segment):
            self._drop_early()
            self.skipped += 1  # altyazı/dublaj sırasında ya da bilgisayar çok yüklüyken
            return
        emit = None
        with self._lock:
            if early is not None and early.segment.key == segment.key and not early.cancelled:
                early.confirmed = True
                self.early_used += 1
                if early.done and not early.emitted:
                    early.emitted = emit = True
                self._early = None
        if emit:
            self._emit(early)
            return
        if early is not None and early.confirmed:
            return  # işçi bitirince gösterir
        self._drop_early()
        self._enqueue(_Job(segment, confirmed=True))

    def _enqueue(self, job: _Job) -> _Job:
        with self._lock:
            waiting = [j for j in self._jobs[1:] if j is not None and not j.cancelled]
            if job.confirmed and len(waiting) >= self.MAX_WAITING:
                waiting[0].cancelled = True  # en eski bekleyen: sıra birikiyor
                self.skipped += 1
            self._jobs.append(job)
            self._wake.notify()
        return job

    def _work(self) -> None:
        while True:
            with self._lock:
                while not self._jobs:
                    self._wake.wait()
                job = self._jobs[0]
            if job is None:
                return
            if not job.cancelled:
                t0 = time.perf_counter()
                job.text = self._recognize(job)
                job.took = time.perf_counter() - t0
            with self._lock:
                self._jobs.pop(0)
                job.done = True
                emit = job.confirmed and not job.cancelled and not job.emitted
                job.emitted = job.emitted or emit
            if emit:
                self._emit(job)

    def _recognize(self, job: _Job) -> str:
        """Whisper. Sıcaklık yedeği kapalı (0,0): yedek, avg_logprob < -1 ya da tekrar olunca parçayı 0,2..1,0
        ile 5 kez daha çözer; accept() aynı eşikle zaten atıyor, işlemcide her yeniden çözme sonraki satırları
        geciktirir. Kaçan nadir satır, gecikmiş her satırdan ucuz."""
        model = self._model
        if model is None:
            return ""
        try:
            parts, _info = model.transcribe(
                job.segment.samples.astype(np.float32), language="en", beam_size=1, temperature=0.0,
                vad_filter=not self._vad, condition_on_previous_text=False, without_timestamps=True,
                initial_prompt=self.prompt or None)
            texts = []
            for p in parts:  # üretici: iptal edilen iş sonraki 30 sn'lik pencereyi çözmez
                if job.cancelled:
                    return ""
                texts.append(accept(p.text, p.no_speech_prob, p.avg_logprob, p.compression_ratio))
        except Exception as exc:
            self._log(f"Sokak konuşması tanınamadı: {exc}")
            return ""
        text = " ".join(t for t in texts if t).strip()
        if text and is_prompt_echo(text, self.prompt):
            return ""
        return text

    def _emit(self, job: _Job) -> None:
        text = job.text
        if not text:
            return
        segment = job.segment
        with self._lock:
            self.heard += 1
            self.seq += 1
            seq = self.seq
        first_seen = segment.end  # gecikme konuşmanın bittiği andan ölçülür
        lag = time.monotonic() - segment.end
        self._log(f"[Sokak] EN ({1000 * job.took:.0f} ms{', erken' if job.segment.final is False else ''}"
                  f"; konuşma bitince {1000 * lag:.0f} ms): {text}")

        def on_result(stage: str, translated: str, _took: float) -> None:
            self.ui.put(("show", self.region_name, seq, stage, translated, text, first_seen, None))

        def on_error(exc: Exception, stage: str, draft_shown: bool) -> None:
            if draft_shown:
                self.ui.put(("draft-final", self.region_name, seq))

        self.translator.submit(text, None, lambda: seq == self.seq, on_result, on_error, source="asr")
