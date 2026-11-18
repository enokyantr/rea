"""Telefon okuma modu: oyundaki telefonun ekranı (sosyal medya gönderileri, haberler, mesajlar).

Altyazı Pipeline'ından farkı: ekranda aynı anda birçok bağımsız metin vardır ve kullanıcı kaydırır.
  1. Yakalama: bölge yalnızca içeriği değişip ~300 ms durağan kalınca okunur (kaydırma bitsin);
     yük seviyesinin izin verdiğinden sık değil, duraklatılmışken hiç.
  2. Bloklar: RapidOCR kutuları satırlara, satırlar dikey boşluğa / sol kenara / yazı boyutuna göre
     bloklara ayrılır; her blok bir gönderi, yorum, başlık ya da mesaj balonudur. Arayüz parçaları
     (saat, beğeni sayısı, "3h") atılır.
  3. Çeviri: blok başına, önbellekli (geri kaydırınca anında). Yalnızca çevrilmemiş görünür bloklar
     yukarıdan aşağı, tek tek son çevirmene (yerel LLM) gider; ekrandan çıkan bloğun sırası iptal.

Bu modül platformdan bağımsızdır (ekran yakalama ve OCR çağırana enjekte edilebilir)."""

from __future__ import annotations

import queue
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import imageproc, textproc
from .geometry import Rect, intersect
from .ocr import Box, Line, merge_lines

SOURCE = "phone"  # llm.SOURCES: istemde <line source="phone">
# Satırlar arası boşluk bu kadar satır yüksekliğini aşarsa yeni blok.
GAP_FACTOR = 0.8
# Sol kenar bu kadar satır yüksekliği kayarsa yeni blok (sağa yaslı mesaj balonu, girintili yanıt).
INDENT_FACTOR = 1.2
# Yazı boyutu bu oranda değişirse yeni blok (haber başlığı -> metin). Kutu yüksekliği harflere göre
# oynak (aynı yazıda 20-29 px); harf başına genişlik daha kararlı, ikisi de bakılır.
HEIGHT_RATIO = 1.35
CHAR_WIDTH_RATIO = 1.4
# Bu kadar sn bölgede yazı yoksa panel gizlenir (panel kendi sayar; bkz. phone_panel).
HIDE_AFTER_S = 3.0
# Yakalama aralığı en az bu (telefon yazısı altyazı gibi kısa süreli değil; ~10 kare/sn yeter).
MIN_CAPTURE_S = 0.10
# Benzer okuma birleştirme: OCR aynı gönderiyi iki taramada bir harf farklı okursa ("2.1K"/"2.lK")
# yeni blok sayılmasın (yeniden çeviri ve kart titremesi olmasın).
SAME_BLOCK_SIMILARITY = 0.92
KNOWN_MAX = 256
# Değişim tespiti: arka plandan en az bu kadar farklı pikseller içerik sayılır (açık ve koyu tema).
CONTENT_DELTA = 60


@dataclass(frozen=True)
class Block:
    """Bir gönderi/yorum/başlık/mesaj. `key`: önbellek ve panel kartı kimliği (ekran sırasından bağımsız)."""

    key: str
    text: str
    top: float
    bottom: float


def blocks_from_lines(lines: list[Line], ignore: list[str] | tuple = (), min_chars: int = 3) -> list[Block]:
    """Satırlar -> bloklar (yukarıdan aşağı), arayüz parçaları atılmış ve metinleri temizlenmiş."""
    lines = [ln for ln in lines if ln.text.strip()]
    if not lines:
        return []
    median_h = statistics.median(ln.height for ln in lines) or 1.0
    groups: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: ln.y0):
        if groups:
            prev = groups[-1][-1]
            gap = line.y0 - prev.y1
            if not (gap > GAP_FACTOR * median_h or abs(line.x0 - prev.x0) > INDENT_FACTOR * median_h
                    or _size_changed(prev, line)):
                groups[-1].append(line)
                continue
        groups.append([line])

    # Yalnızca arayüz parçası olan satırlar (beğeni/saat satırı) bloktan çıkar.
    groups = [[ln for ln in g if not textproc.is_ui_chrome(ln.text)] for g in groups]
    groups = [g for g in groups if g]
    # Gönderi başlığı ("Lenny Avery @lennyavery · 3h") boşlukla ayrıldıysa gönderisine eklenir.
    merged: list[list[Line]] = []
    carry: list[Line] = []
    for i, g in enumerate(groups):
        if carry:
            g = carry + g
            carry = []
        if _is_header(g) and i + 1 < len(groups):
            nxt = groups[i + 1][0]
            if nxt.y0 - g[-1].y1 <= 2.5 * median_h:
                carry = g
                continue
        merged.append(g)
    if carry:
        merged.append(carry)

    out: list[Block] = []
    seen: dict[str, int] = {}
    for g in merged:
        text = textproc.normalize_post([ln.text for ln in g])
        if not _keep(text, ignore, min_chars):
            continue
        base = " ".join(text.split()).casefold()
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(Block(base if n == 0 else f"{base}\x1f{n}", text, g[0].y0, g[-1].y1))
    return out


def _ratio(a: float, b: float) -> float:
    return max(a, b) / max(1e-6, min(a, b))


def _size_changed(a: Line, b: Line) -> bool:
    if _ratio(a.height, b.height) > HEIGHT_RATIO:
        return True
    na, nb = len(a.text.replace(" ", "")), len(b.text.replace(" ", ""))
    if na < 8 or nb < 8:
        return False
    return _ratio((a.x1 - a.x0) / na, (b.x1 - b.x0) / nb) > CHAR_WIDTH_RATIO


def _is_header(group: list[Line]) -> bool:
    """Tek satırlık gönderi başlığı: kullanıcı adı (+ görünen ad, en fazla 4 kelime, + saat)."""
    if len(group) != 1:
        return False
    text = group[0].text
    if textproc.is_handle_only(text):
        return True
    tokens = text.split()
    if not any(t.startswith("@") for t in tokens):
        return False
    words = [t for t in tokens if not t.startswith("@") and not textproc.is_ui_chrome(t)]
    return len(words) <= 4


def _keep(text: str, ignore, min_chars: int) -> bool:
    if len(text.strip()) < max(3, min_chars) or textproc.is_ui_chrome(text) or textproc.is_handle_only(text):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if ignore and not textproc.ignore_phrases(text, ignore, min_chars):
        return False
    return not textproc.looks_turkish(text)  # konsolun Türkçe arayüzü


def segment(boxes: list[Box], ignore=(), min_chars: int = 3) -> list[Block]:
    """RapidOCR kutuları -> bloklar."""
    return blocks_from_lines(merge_lines(boxes), ignore, min_chars)


def segment_lines(texts: list[str], ignore=(), min_chars: int = 3) -> list[Block]:
    """Kutu bilgisi vermeyen motor (Windows OCR): her satır ayrı blok."""
    out: list[Block] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(texts):
        text = textproc.normalize_post([raw])
        if textproc.is_ui_chrome(raw) or not _keep(text, ignore, min_chars):
            continue
        base = " ".join(text.split()).casefold()
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(Block(base if n == 0 else f"{base}\x1f{n}", text, float(i), float(i) + 1))
    return out


def read_blocks(ocr, gray: np.ndarray, ignore=(), min_chars: int = 3) -> list[Block]:
    """Motor kutu verebiliyorsa (RapidOCR.boxes) gerçek bloklar; veremiyorsa satır başına blok."""
    boxes = getattr(ocr, "boxes", None)
    if boxes is not None:
        return segment(boxes(gray), ignore, min_chars)
    return segment_lines(ocr.recognize(gray), ignore, min_chars)


class KnownBlocks:
    """Son görülen blok metinleri: OCR'ın aynı gönderiyi küçük farkla okuması aynı bloğa bağlanır.
    Sınırlı (KNOWN_MAX); benzerlik yalnızca en son görülen RECENT blokla karşılaştırılır (tarama
    başına en fazla blok x RECENT karşılaştırma, hızlı ön elemeyle)."""

    RECENT = 32

    def __init__(self, limit: int = KNOWN_MAX):
        self._limit = limit
        self._texts: OrderedDict[str, str] = OrderedDict()  # anahtar -> metin

    def _similar_key(self, text: str, used: set[str]) -> str | None:
        from difflib import SequenceMatcher

        low = text.lower()
        for i, (key, known) in enumerate(reversed(self._texts.items())):
            if i >= self.RECENT:
                break
            if key in used or abs(len(known) - len(text)) > max(2, len(known) // 12):
                continue
            m = SequenceMatcher(None, known.lower(), low, autojunk=False)
            if (m.real_quick_ratio() >= SAME_BLOCK_SIMILARITY and m.quick_ratio() >= SAME_BLOCK_SIMILARITY
                    and m.ratio() >= SAME_BLOCK_SIMILARITY):
                return key
        return None

    def canonical(self, blocks: list[Block]) -> list[Block]:
        out: list[Block] = []
        used: set[str] = set()
        for b in blocks:
            if b.key not in self._texts or b.key in used:
                key = self._similar_key(b.text, used | {b.key})
                if key is not None:
                    b = Block(key, self._texts[key], b.top, b.bottom)
            used.add(b.key)
            out.append(b)
        for b in out:
            self._texts[b.key] = b.text
            self._texts.move_to_end(b.key)
        while len(self._texts) > self._limit:
            self._texts.popitem(last=False)
        return out


# --- çeviri sırası ---------------------------------------------------------------------------------


class PhoneQueue:
    """Görünür blokların çevirisi: önbellekte olan anında; olmayanlar yukarıdan aşağı, aynı anda
    yalnızca biri çevirmende. Ekrandan çıkan bloğun sırası iptal edilir (sürmekte olan istek
    bitirilir, sonucu önbelleğe girer: geri kaydırınca anında).

    `translator`: HybridTranslator. Son çevirmen varsa yalnızca o (taslak yok: yazı iki kez değişmez);
    yoksa (ör. oyun modunda dil modeli kapalı) yerel taslak son çeviri sayılır.
    `on_result(key, text, translated)` ve `on_error(key, message)` çevirmen iş parçacığından çağrılır."""

    RETRY_AFTER_S = 5.0

    def __init__(self, translator, on_result: Callable[[str, str, str], None],
                 on_error: Callable[[str, str], None] = lambda key, message: None,
                 log: Callable[[str], None] = lambda msg: None):
        self.translator = translator
        self._on_result = on_result
        self._on_error = on_error
        self._log = log
        self._lock = threading.Lock()
        self._visible: dict[str, Block] = {}
        self._wanted: list[str] = []
        self._busy: str | None = None
        self._failed: dict[str, float] = {}  # anahtar -> tekrar denenebileceği an
        self.debug = False

    def cached(self, text: str) -> str | None:
        return self.translator.cached(text, source=SOURCE, draft=False)

    def update(self, blocks: list[Block]) -> list[tuple[Block, str | None]]:
        """Ekrandaki bloklar (sırayla). Dönüş: (blok, önbellekteki çeviri ya da None)."""
        now = time.monotonic()
        rows = [(b, self.cached(b.text)) for b in blocks]
        with self._lock:
            self._visible = {b.key: b for b in blocks}
            self._wanted = [b.key for b, hit in rows if hit is None and b.key != self._busy
                            and self._failed.get(b.key, 0.0) <= now]
            job = self._take_next_locked()
        self._submit(job)
        return rows

    def pending(self) -> list[str]:
        with self._lock:
            return ([self._busy] if self._busy else []) + list(self._wanted)

    def clear(self) -> None:
        with self._lock:
            self._visible = {}
            self._wanted = []

    def _take_next_locked(self) -> tuple[str, Block] | None:
        if self._busy is not None or not self._wanted:
            return None
        key = self._wanted.pop(0)
        self._busy = key
        return key, self._visible[key]

    def _submit(self, job: tuple[str, Block] | None) -> None:
        """Kilit dışında: çevirmen önbellekten hemen (aynı iş parçacığında) yanıt verebilir."""
        if job is None:
            return
        key, block = job
        started = time.perf_counter()

        def on_result(_stage, translated: str, _took: float) -> None:
            self._done(key, block.text, translated, None, started)

        def on_error(exc: Exception, _stage, _shown) -> None:
            self._done(key, block.text, None, exc, started)

        try:
            self.translator.submit(block.text, None, lambda: True, on_result, on_error, source=SOURCE,
                                   draft=False)
        except RuntimeError as exc:  # çevirmen kapatılıyor
            with self._lock:
                self._busy = None
            self._log(f"telefon: çeviri başlatılamadı ({exc})")

    def _done(self, key: str, text: str, translated: str | None, exc: Exception | None, started: float) -> None:
        with self._lock:
            if self._busy == key:
                self._busy = None
            if exc is not None:
                self._failed[key] = time.monotonic() + self.RETRY_AFTER_S
                if len(self._failed) > KNOWN_MAX:
                    self._failed.pop(next(iter(self._failed)))
            job = self._take_next_locked()
        if exc is not None:
            self._on_error(key, str(exc))
        else:
            if self.debug:
                self._log(f"telefon: çeviri {1000 * (time.perf_counter() - started):.0f} ms: {translated}")
            self._on_result(key, text, translated)
        self._submit(job)


# --- okuma iş parçacığı -------------------------------------------------------------------------


class ScanGate:
    """Ne zaman OCR yapılır: içerik son okumadan beri değişti VE `debounce_s` boyunca durağan
    (kaydırma bitti) VE son OCR'dan beri yük seviyesinin izin verdiği süre geçti. `force`: hemen."""

    def __init__(self, debounce_s: float):
        self.debounce_s = debounce_s
        self._last_sig: np.ndarray | None = None  # bir önceki kare
        self._read_sig: np.ndarray | None = None  # son OCR'daki kare
        self._stable_since = 0.0
        self._last_ocr_at = -1e9
        self.forced = False

    def reset(self) -> None:
        self._last_sig = self._read_sig = None
        self._last_ocr_at = -1e9

    def feed(self, sig: np.ndarray, now: float, min_gap_s: float) -> bool:
        if not imageproc.same_signature(self._last_sig, sig):
            self._stable_since = now
        self._last_sig = sig
        if self.forced:
            ready = True
        else:
            changed = not imageproc.same_signature(self._read_sig, sig)
            ready = (changed and now - self._stable_since >= self.debounce_s
                     and now - self._last_ocr_at >= min_gap_s)
        if ready:
            self.forced = False
            self._read_sig = sig
            self._last_ocr_at = now
        return ready


def content_signature(gray: np.ndarray) -> np.ndarray:
    """Telefon ekranının kaba içerik haritası: açık temada koyu yazı, koyu temada açık yazı; ikisi
    de zeminden (ortanca parlaklık) belirgin farklıdır."""
    if gray.size == 0:
        return imageproc.signature(np.zeros((1, 1), dtype=bool))
    small = gray[::4, ::4]
    background = int(np.median(small))
    mask = np.abs(gray.astype(np.int16) - background) > CONTENT_DELTA
    h, w = gray.shape
    rows = max(16, min(96, h // 12))
    return imageproc.signature(mask, cols=48, rows=rows)


class PhonePipeline(threading.Thread):
    """Telefon bölgesini okur, bloklara ayırır ve çevirir. Arayüze ui kuyruğuyla bildirir:
      ("phone-region", rect | None), ("phone", [(anahtar, İngilizce, Türkçe | None), ...]),
      ("phone-tr", anahtar, Türkçe), ("phone-error", mesaj).
    `get_rect`: telefon modu açık ve bölge seçili/açıksa bölgenin ekran dikdörtgeni, değilse None
    (None iken hiçbir iş yapılmaz: yakalama yok, OCR yok). `grab(rect)`: BGRA kare (varsayılan mss)."""

    def __init__(self, cfg, get_rect: Callable[[], Rect | None], translator, ui: queue.Queue,
                 own_rects: Callable[[], list[Rect]] = lambda: [], log: Callable[[str], None] = print,
                 make_ocr: Callable[[], object] | None = None,
                 grab: Callable[[Rect], np.ndarray] | None = None):
        super().__init__(daemon=True, name="pipeline-phone")
        self.cfg = cfg
        self.region_name = "telefon"
        self.get_rect = get_rect
        self.ui = ui
        self.own_rects = own_rects
        self.log = log
        self._make_ocr = make_ocr
        self._grab = grab
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.reset = threading.Event()
        self._force = threading.Event()
        from .perf import STATIC

        self.load = STATIC
        self.dump_dir = None
        self.ocr_device: str | None = None
        self.engine_name = ""
        self.last_read_at = 0.0
        self.last_scan: tuple[int, int, float] | None = None  # (blok, yeni, OCR ms) — menü için
        self.known = KnownBlocks()
        self.queue = PhoneQueue(translator, self._translated, self._failed, log)
        self._ocr = None

    @property
    def translator(self):
        return self.queue.translator

    @property
    def debug(self) -> bool:
        return self.queue.debug

    @debug.setter
    def debug(self, on: bool) -> None:
        self.queue.debug = on

    def scan_now(self) -> None:
        """Kısayol / menü: değişim beklemeden hemen oku."""
        self._force.set()

    # --- çeviri geri çağrıları (çevirmen iş parçacığı) ---

    def _translated(self, key: str, _text: str, translated: str) -> None:
        self.ui.put(("phone-tr", key, translated))

    def _failed(self, key: str, message: str) -> None:
        self.log(f"telefon: çeviri hatası ({message[:120]})")
        self.ui.put(("phone-error", message))

    # --- okuma ---

    def _ensure_ocr(self):
        if self._ocr is None:
            if self._make_ocr is not None:
                self._ocr = self._make_ocr()
            else:
                from . import ocr as ocr_mod

                self._ocr = ocr_mod.create(self.cfg, self.ocr_device)
            self.engine_name = getattr(self._ocr, "name", "")
            textproc.normalize_post(["warm up"])  # sözlük ilk taramada değil şimdi yüklensin (~0,1 sn)
            if getattr(self._ocr, "boxes", None) is None:
                self.log("telefon: OCR motoru kutu bilgisi vermiyor (Windows OCR); her satır ayrı blok.")
        return self._ocr

    def scan(self, gray: np.ndarray) -> list[Block]:
        """Tek tarama: OCR + bloklar + çeviri sırası. Arayüze ("phone", ...) gönderir."""
        ocr = self._ensure_ocr()
        t0 = time.perf_counter()
        blocks = read_blocks(ocr, gray, self.cfg.ignore_phrases, self.cfg.min_text_chars)
        ocr_ms = 1000 * (time.perf_counter() - t0)
        blocks = self.known.canonical(blocks)
        rows = self.queue.update(blocks)
        new = sum(1 for b, hit in rows if hit is None)
        self.last_scan = (len(blocks), new, ocr_ms)
        if self.debug:
            self.log(f"telefon: {len(blocks)} blok, {new} yeni, OCR {ocr_ms:.0f} ms")
        self.ui.put(("phone", [(b.key, b.text, hit) for b, hit in rows]))
        return blocks

    def _capture(self, sct, rect: Rect) -> np.ndarray:
        if self._grab is not None:
            return self._grab(rect)
        return np.asarray(sct.grab({"left": rect.left, "top": rect.top, "width": rect.width,
                                    "height": rect.height}))

    def _mask_own(self, gray: np.ndarray, rect: Rect) -> None:
        for own in self.own_rects():
            hit = intersect(own, rect)
            if hit is not None:  # kendi panelimiz/menümüz okunmasın
                gray[hit.top - rect.top:hit.bottom - rect.top, hit.left - rect.left:hit.right - rect.left] = 0

    def step(self, sct, gate: ScanGate, state: dict, now: float) -> float:
        """Döngünün bir adımı; bir sonraki adıma kadar beklenecek süre (sn)."""
        forced = self._force.is_set()
        self._force.clear()
        if self.reset.is_set():
            self.reset.clear()
            gate.reset()
            self.queue.clear()
        if self.paused.is_set():
            return 0.05
        rect = self.get_rect()
        if rect != state.get("rect"):
            state["rect"] = rect
            gate.reset()
            self.ui.put(("phone-region", rect))
            if rect is None:
                self.queue.clear()
        if rect is None or rect.width < 8 or rect.height < 8:
            return 0.25
        try:
            frame = self._capture(sct, rect)
        except Exception as exc:
            self.log(f"telefon: ekran yakalanamadı ({exc})")
            return 0.5
        self.last_read_at = now
        gray = imageproc.to_gray(frame)
        self._mask_own(gray, rect)
        gate.forced = gate.forced or forced
        if gate.feed(content_signature(gray), now, self.load.min_ocr_gap_s):
            self.scan(gray)
        return max(MIN_CAPTURE_S, self.load.capture_interval_s)

    def run(self) -> None:
        # Telefon modu kapalı / bölge yokken hiçbir şey yapılmaz (OCR motoru da kurulmaz).
        while not self.stopped.is_set() and self.get_rect() is None:
            time.sleep(0.25)
        if self.stopped.is_set():
            return
        try:
            self._ensure_ocr()
        except Exception as exc:
            self.log(f"telefon: OCR başlatılamadı ({exc})")
            self.ui.put(("phone-error", f"OCR başlatılamadı: {exc}"))
            return
        gate = ScanGate(max(0, int(self.cfg.phone_debounce_ms)) / 1000)
        state: dict = {}
        if self._grab is not None:
            self._loop(None, gate, state)
            return
        import mss

        with mss.mss() as sct:
            self._loop(sct, gate, state)

    def _loop(self, sct, gate: ScanGate, state: dict) -> None:
        while not self.stopped.is_set():
            tick = time.monotonic()
            gate.debounce_s = max(0, int(self.cfg.phone_debounce_ms)) / 1000
            wait = self.step(sct, gate, state, tick)
            elapsed = time.monotonic() - tick
            if wait > elapsed:
                self._force.wait(wait - elapsed)  # "şimdi çevir" beklemeyi keser
