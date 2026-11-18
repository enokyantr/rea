"""OCR motorları: RapidOCR (PP-OCRv4, ONNX, işlemcide) ve Windows'un yerleşik OCR'ı (pywinrt).

Gerçek kayıtta (kontursuz, sıkıştırılmış video altyazısı, açık beton zemin) Windows OCR 300 okumada
"you'd get some bitches on your dick" satırını hiç okuyamadı ("YOI-JICleget sorr on your click.");
RapidOCR aynı satırı ilk denemede neredeyse kusursuz okudu ("you'd get some bjtches on your dlick.",
~40-100 ms/kare). Bu yüzden varsayılan RapidOCR; kurulu değilse Windows OCR kullanılır."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import numpy as np

INSTALL_HINT = (
    "Windows OCR için '{lang}' dil paketi bulunamadı.\n"
    "Yönetici PowerShell'de şunu çalıştırıp programı yeniden başlatın:\n"
    "  Add-WindowsCapability -Online -Name \"Language.OCR~~~{lang}~0.0.1.0\"\n"
    "(veya Ayarlar > Saat ve dil > Dil ve bölge > English (United States) ekleyin)"
)


class OcrUnavailable(RuntimeError):
    pass


class WindowsOcr:
    name = "Windows OCR"
    # Ham görüntü yerine ön işlenmiş (ikili) seçenekler ve büyütme ister.
    raw_input = False

    def __init__(self, language: str = "en-US"):
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine

        engine = None
        if OcrEngine.is_language_supported(Language(language)):
            engine = OcrEngine.try_create_from_language(Language(language))
        if engine is None:
            raise OcrUnavailable(INSTALL_HINT.format(lang=language))
        self._engine = engine
        self.max_dimension = int(OcrEngine.max_image_dimension)
        # Her iş parçacığı kendi döngüsünü kullanır; nesne oluşturulduğu iş parçacığında kullanılmalı.
        self._loop = asyncio.new_event_loop()

    def recognize(self, gray: np.ndarray) -> list[str]:
        """Gri ton (uint8, HxW) görüntüdeki satırları döndürür."""
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap

        h, w = gray.shape
        if h < 8 or w < 8:
            return []
        if max(h, w) > self.max_dimension:
            step = int(np.ceil(max(h, w) / self.max_dimension))
            gray = gray[::step, ::step]
            h, w = gray.shape
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[..., 0] = bgra[..., 1] = bgra[..., 2] = gray
        bgra[..., 3] = 255

        # 4 argümanlı biçim pywinrt v3 ve v4'te aynı (alfa varsayılan: premultiplied; alfa zaten 255).
        bitmap = SoftwareBitmap.create_copy_from_buffer(memoryview(bgra).cast("B"), BitmapPixelFormat.BGRA8, w, h)
        result = self._loop.run_until_complete(self._recognize(bitmap))
        return [line.text for line in result.lines]

    async def _recognize(self, bitmap):
        return await self._engine.recognize_async(bitmap)


class RapidOcr:
    """PP-OCRv4 metin bulma + tanıma (rapidocr-onnxruntime). Modeller paketin içinde gelir."""

    name = "RapidOCR"
    # Doğal görüntüyü (gri ton) olduğu gibi ister; ikili maske ve büyütme okumayı bozar.
    raw_input = True

    def __init__(self, threads: int = 2, gpu: bool = True):
        try:
            from rapidocr_onnxruntime import RapidOCR
            _quiet_rapidocr_logs()
        except ImportError as exc:
            raise OcrUnavailable(f"RapidOCR kurulu değil ({exc}); guncelle.bat çalıştırın.") from exc
        self.last_confidence = 0.0
        self.last_span: tuple[float, float] | None = None
        self.last_box: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1)
        # İşlemci çekirdeklerinin hepsini almasın: yerel taslak çeviri de işlemcide çalışıyor.
        try:
            # Metin bulma girdisi: varsayılan ("min" 736) dar bir altyazı şeridini kısa kenarı 736
            # olacak şekilde ~5000 piksele büyütüp kareyi 0,5 sn'ye çıkarıyordu. "max" modunda
            # büyütme yok (1.4.x boyutu 960/1500/2000 sınırlarından kendisi seçer).
            options = dict(det_limit_type="max",
                           det_intra_op_num_threads=threads, rec_intra_op_num_threads=threads)
            # Ekran kartı (DirectML, onnxruntime-directml): OCR'ı işlemciden alır; işlemci yerel taslak
            # çeviriye kalır. DirectML yoksa RapidOCR kendisi işlemciye döner.
            engine = None
            if gpu:
                try:
                    engine = RapidOCR(**options, det_use_dml=True, rec_use_dml=True)
                except Exception:
                    engine = None
            self._engine = engine or RapidOCR(**options)
        except Exception as exc:  # eski sürüm (< 1.4) bu ayarları tanımıyor, onnxruntime DLL hatası vb.
            raise OcrUnavailable(f"RapidOCR başlatılamadı ({type(exc).__name__}: {exc}); "
                                 "guncelle.bat çalıştırın.") from exc
        self.on_gpu = _first_provider(self._engine) == "DmlExecutionProvider"
        self.name = "RapidOCR (ekran kartı, DirectML)" if self.on_gpu else "RapidOCR (işlemci)"
        # Ekran kartı istendi ama kullanılamadıysa nedeni (kayıtta görünür): OCR işlemcide 5-10 kat yavaş.
        self.gpu_note = ""
        self.gpu_requested = gpu
        if gpu and not self.on_gpu:
            try:
                import onnxruntime as ort

                providers = ", ".join(ort.get_available_providers())
            except Exception as exc:
                providers = f"okunamadı ({exc})"
            self.gpu_note = ("Ekran kartı (DirectML) ile OCR kurulamadı; işlemcide çalışıyor (yavaş, oyunu "
                             f"kastırır). onnxruntime sağlayıcıları: {providers}. Çözüm: guncelle.bat'ı yeniden "
                             "çalıştırın; olmazsa .venv\\Scripts\\python -m pip install --force-reinstall --no-deps "
                             "onnxruntime-directml")

    def recognize(self, gray: np.ndarray) -> list[str]:
        """Gri ton (uint8, HxW) görüntüdeki satırları yukarıdan aşağı döndürür."""
        return [line.text for line in merge_lines(self.boxes(gray))]

    def boxes(self, gray: np.ndarray) -> list[Box]:
        """Bulunan yazı kutuları (metin, güven ve konum; girdi görüntüsü koordinatlarında).
        Telefon okuması satırları ve gönderi bloklarını bunlardan kurar (bkz. phone.segment)."""
        h, w = gray.shape
        if h < 8 or w < 8:
            self.last_confidence, self.last_span, self.last_box = 0.0, None, None
            return []
        result, _ = self._engine(np.repeat(gray[:, :, None], 3, axis=2), use_cls=False)
        found = to_boxes(result or [])
        # Son okumanın en düşük kutu güveni ve yazının yatay kapsamı (Türkçeyi ortalamak için).
        self.last_confidence = min((b.score for b in found), default=0.0)
        if found:
            x0, y0 = min(b.x0 for b in found), min(b.y0 for b in found)
            x1, y1 = max(b.x1 for b in found), max(b.y1 for b in found)
            self.last_span = (x0, x1)
            self.last_box = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)
        else:
            self.last_span = self.last_box = None
        return found


def _quiet_rapidocr_logs() -> None:
    """RapidOCR her motor için "try to use DirectML" bilgi satırı basıyor; yalnızca uyarılar kalsın.
    Kayıtçılarını önceden oluşturup düzeyini düşürür (get_logger önbellekli, düzeyi bir kez ayarlar)."""
    import logging

    try:
        from rapidocr_onnxruntime.utils.logger import get_logger
    except ImportError:
        return
    for name in ("OrtInferSession", "RapidOCR"):
        get_logger(name).setLevel(logging.WARNING)


def _first_provider(engine) -> str:
    try:
        return engine.text_det.infer.session.get_providers()[0]
    except Exception:
        return ""


@dataclass(frozen=True)
class Box:
    """Tek bir OCR kutusu: metni, güveni (0..1) ve eksenlere hizalı sınırları (piksel)."""

    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Line:
    """Aynı satırdaki kutuların soldan sağa birleşimi."""

    text: str
    boxes: tuple[Box, ...]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def score(self) -> float:
        return min((b.score for b in self.boxes), default=0.0)


def to_boxes(raw) -> list[Box]:
    """RapidOCR çıktısı ([dörtgen, metin, güven], ...) -> Box listesi. Ham biçimi yalnızca bu
    fonksiyon yorumlar; satır gruplama ve telefon blokları Box kullanır."""
    out = []
    for quad, text, score in raw:
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        out.append(Box(str(text).strip(), float(score), min(xs), min(ys), max(xs), max(ys)))
    return out


def merge_lines(boxes: list[Box]) -> list[Line]:
    """Kutular -> satırlar (yukarıdan aşağı). Aynı satır bazen birkaç kutuya bölünür ("...that
    oldkyee-y" + "jee ass haircut..."): dikeyde örtüşenler soldan sağa birleşir."""
    groups: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b.y0, b.y1, b.x0, b.text)):
        if groups:
            ltop, lbottom = min(b.y0 for b in groups[-1]), max(b.y1 for b in groups[-1])
            overlap = min(box.y1, lbottom) - max(box.y0, ltop)
            if overlap > 0.5 * min(box.height, lbottom - ltop):
                groups[-1].append(box)
                continue
        groups.append([box])
    lines = []
    for group in groups:
        group.sort(key=lambda b: b.x0)
        lines.append(Line(" ".join(b.text for b in group if b.text), tuple(group),
                          min(b.x0 for b in group), min(b.y0 for b in group),
                          max(b.x1 for b in group), max(b.y1 for b in group)))
    return lines


def group_lines(raw) -> list[str]:
    """RapidOCR kutuları ([dörtgen, metin, güven]) -> satır metinleri."""
    return [line.text for line in merge_lines(to_boxes(raw))]


class SharedRapid:
    """Tüm okuma bölgelerinin ortak kullandığı TEK RapidOCR motoru (kilitle sırayla).

    Her bölge (altyazı, görev, mesaj bildirimi, telefon) kendi motorunu kurunca ekran kartında bölge
    başına 3 DirectML oturumu açılıyordu; 8 GB kartta yerel dil modeli (5-6 GB) ile birlikte bellek
    yetmeyip program çöküyordu. Ortak motorda oturumlar bir kez açılır; OCR GPU'da 50-200 ms olduğu
    için bölgelerin sırayla beklemesi fark edilmez. Her tutamak kendi `last_*` sonucunu tutar
    (okuma bittiği anda kilit altında kopyalanır), böylece iş parçacıkları birbirinin güvenini görmez."""

    raw_input = True

    def __init__(self, engine: "RapidOcr", lock: threading.Lock):
        self._engine = engine
        self._lock = lock
        self.name = engine.name
        self.on_gpu = engine.on_gpu
        self.gpu_note = engine.gpu_note
        self.gpu_requested = engine.gpu_requested
        self.last_confidence = 0.0
        self.last_span: tuple[float, float] | None = None
        self.last_box: tuple[int, int, int, int] | None = None

    def _sync(self) -> None:
        e = self._engine
        self.last_confidence, self.last_span, self.last_box = e.last_confidence, e.last_span, e.last_box

    def recognize(self, gray: np.ndarray) -> list[str]:
        with self._lock:
            lines = self._engine.recognize(gray)
            self._sync()
        return lines

    def boxes(self, gray: np.ndarray) -> list[Box]:
        with self._lock:
            found = self._engine.boxes(gray)
            self._sync()
        return found


_shared: dict[bool, tuple["RapidOcr", threading.Lock]] = {}
_shared_guard = threading.Lock()


def _shared_rapid(gpu: bool) -> SharedRapid:
    with _shared_guard:
        if gpu not in _shared:
            _shared[gpu] = (RapidOcr(gpu=gpu), threading.Lock())
        engine, lock = _shared[gpu]
    return SharedRapid(engine, lock)


def create(cfg, device: str | None = None):
    """`cfg.ocr_engine`: "auto" (RapidOCR, yoksa Windows OCR) | "rapid" | "windows". `device`: RapidOCR'ın
    yeri ("auto" | "cpu"); verilmezse `cfg.ocr_device` (oyun modu bunu perf.game_policy ile değiştirir)."""
    if cfg.ocr_engine in ("auto", "rapid"):
        try:
            return _shared_rapid((device or cfg.ocr_device) != "cpu")
        except OcrUnavailable:
            if cfg.ocr_engine == "rapid":
                raise
    return WindowsOcr(cfg.ocr_language)
