"""OCR motorları: RapidOCR (PP-OCRv4, ONNX, işlemcide) ve Windows'un yerleşik OCR'ı (pywinrt).

Gerçek kayıtta (kontursuz, sıkıştırılmış video altyazısı, açık beton zemin) Windows OCR 300 okumada
"you'd get some bitches on your dick" satırını hiç okuyamadı ("YOI-JICleget sorr on your click.");
RapidOCR aynı satırı ilk denemede neredeyse kusursuz okudu ("you'd get some bjtches on your dlick.",
~40-100 ms/kare). Bu yüzden varsayılan RapidOCR; kurulu değilse Windows OCR kullanılır."""

from __future__ import annotations

import asyncio

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

    def __init__(self, threads: int = 2):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OcrUnavailable(f"RapidOCR kurulu değil ({exc}); guncelle.bat çalıştırın.") from exc
        # İşlemci çekirdeklerinin hepsini almasın: yerel taslak çeviri de işlemcide çalışıyor.
        try:
            # Metin bulma girdisi: varsayılan ("min" 736) dar bir altyazı şeridini kısa kenarı 736
            # olacak şekilde ~5000 piksele büyütüp kareyi 0,5 sn'ye çıkarıyordu. "max" modunda
            # büyütme yok (1.4.x boyutu 960/1500/2000 sınırlarından kendisi seçer).
            self._engine = RapidOCR(det_limit_type="max",
                                    det_intra_op_num_threads=threads, rec_intra_op_num_threads=threads)
        except Exception as exc:  # eski sürüm (< 1.4) bu ayarları tanımıyor, onnxruntime DLL hatası vb.
            raise OcrUnavailable(f"RapidOCR başlatılamadı ({type(exc).__name__}: {exc}); "
                                 "guncelle.bat çalıştırın.") from exc

    def recognize(self, gray: np.ndarray) -> list[str]:
        """Gri ton (uint8, HxW) görüntüdeki satırları yukarıdan aşağı döndürür."""
        h, w = gray.shape
        if h < 8 or w < 8:
            return []
        result, _ = self._engine(np.repeat(gray[:, :, None], 3, axis=2), use_cls=False)
        return group_lines(result or [])


def group_lines(boxes) -> list[str]:
    """RapidOCR kutuları ([dörtgen, metin, güven]) -> satırlar. Aynı satır bazen birkaç kutuya
    bölünür ("...that oldkyee-y" + "jee ass haircut..."): dikeyde örtüşenler soldan sağa birleşir."""
    items = []
    for quad, text, _score in boxes:
        ys = [p[1] for p in quad]
        xs = [p[0] for p in quad]
        items.append((min(ys), max(ys), min(xs), str(text).strip()))
    items.sort()
    lines: list[list[tuple]] = []
    for item in items:
        top, bottom = item[0], item[1]
        if lines:
            ltop, lbottom = min(i[0] for i in lines[-1]), max(i[1] for i in lines[-1])
            overlap = min(bottom, lbottom) - max(top, ltop)
            if overlap > 0.5 * min(bottom - top, lbottom - ltop):
                lines[-1].append(item)
                continue
        lines.append([item])
    return [" ".join(i[3] for i in sorted(line, key=lambda i: i[2]) if i[3]) for line in lines]


def create(cfg):
    """`cfg.ocr_engine`: "auto" (RapidOCR, yoksa Windows OCR) | "rapid" | "windows"."""
    if cfg.ocr_engine in ("auto", "rapid"):
        try:
            return RapidOcr()
        except OcrUnavailable:
            if cfg.ocr_engine == "rapid":
                raise
    return WindowsOcr(cfg.ocr_language)
