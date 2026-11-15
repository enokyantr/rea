"""Windows'un yerleşik OCR motoru (Windows.Media.Ocr), pywinrt üzerinden."""

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
