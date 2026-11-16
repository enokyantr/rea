"""Altyazı bölgesinden metin okuma: ön işleme seçenekleri + en anlamlı OCR sonucunu seçme.

Tek bir ön işleme her sahnede işe yaramıyor (gerçek video: gri ton koyu sahnede kusursuz, açık
sahnede t→c/h→ij hatalı; sıkı kontur maskesi sıkıştırılmış videoda harfleri parçalıyor). Bu yüzden
"auto" modunda birkaç seçenek (kontur, parlak, gri) OCR'lanır ve gerçek metne en çok benzeyen sonuç seçilir. OCR satır
başına ~5-20 ms olduğu için bu ucuzdur; ilk seçenek yeterince iyiyse diğerleri hiç denenmez."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import imageproc, textproc

# İlk seçenek bu kaliteye ulaşırsa diğerleri denenmez (gecikme ve işlemci tasarrufu).
GOOD_ENOUGH = 0.97


@dataclass
class ReadResult:
    text: str
    variant: str = ""
    # (seçenek, okunan metin, puan) — --debug/--dump için
    candidates: list[tuple[str, str, float]] = field(default_factory=list)
    crop: np.ndarray | None = None
    # Motor bildiriyorsa (RapidOCR): okumanın güveni (0..1) ve yazının yatay ortası (girdi
    # görüntüsünün sol kenarına göre piksel).
    confidence: float | None = None
    center_x: float | None = None
    # Bulunan yazının kutusu (x0, y0, x1, y1), girdi görüntüsü koordinatlarında. Değişim tespiti
    # için: yalnızca bu alan değişince yeniden okunur.
    text_box: tuple[int, int, int, int] | None = None


def variants(gray: np.ndarray, cfg, mode: str | None = None) -> list[tuple[str, np.ndarray]]:
    """Denenecek OCR girdileri, sırayla. `cfg.preprocess`: auto | outline | bright | none | tophat.
    "all" (yalnızca --ocr-compare): hepsi."""
    mode = mode or cfg.preprocess
    out: list[tuple[str, np.ndarray]] = []
    # Sıra önemli: ilk okuma yeterince iyiyse diğerleri denenmez. Gerçek kayıtlarda (--ocr-compare,
    # 300 kare) en iyi birleşim kontur + parlak + gri; top-hat katkı vermedi ve en yavaşıydı.
    if mode in ("auto", "outline", "all"):
        mask = imageproc.text_mask(gray, "outline", cfg.bright_threshold, cfg.dark_threshold,
                                   cfg.outline_radius, cfg.outline_white_threshold)
        out.append(("kontur", imageproc.for_ocr(gray, mask, "outline")))
    if mode in ("auto", "bright", "all"):
        out.append(("parlak", imageproc.for_ocr(gray, gray >= cfg.bright_threshold, "bright")))
    if mode in ("auto", "none", "all"):
        out.append(("gri", gray))
    if mode in ("tophat", "all"):
        out.append(("tophat", imageproc.tophat(gray, cfg.tophat_radius)))
    return out


# Windows OCR küçük ve bulanık yazıda harfleri bölüyor ("some" -> "sorrje", "dick" -> "click").
# Yazı satırı bu yükseklikten (piksel) kısaysa OCR'dan önce daha çok büyütülür.
TARGET_TEXT_HEIGHT = 40
MAX_UPSCALE = 4.0


def upscale_factor(crop_height: int, cfg) -> float:
    """`crop_height`: yazı kutusunun yüksekliği (bright_bbox kenar payı dahil, 2 x 8 px)."""
    text_h = crop_height - 16
    if text_h <= 0:
        return cfg.upscale
    return max(cfg.upscale, min(MAX_UPSCALE, TARGET_TEXT_HEIGHT / text_h))


def read_text(gray: np.ndarray, recognize: Callable[[np.ndarray], list[str]], cfg,
              raw: bool = False, engine=None) -> ReadResult:
    """`gray`: bölgenin gri tonu. `recognize`: görüntü -> satırlar. `raw`: motor (RapidOCR) doğal
    görüntüyü olduğu gibi ister; ön işleme seçenekleri ve büyütme uygulanmaz."""
    bbox = imageproc.bright_bbox(gray >= cfg.bright_threshold)
    left = top = 0
    if bbox is None:
        if cfg.skip_without_bright_text:
            return ReadResult("")
    else:
        top, bottom, left, right = bbox
        gray = gray[top:bottom, left:right]

    result = ReadResult("", crop=gray)
    factor = 1.0 if raw else upscale_factor(gray.shape[0] if bbox is not None else 0, cfg)
    best_score = None
    for name, image in ([("ham", gray)] if raw else variants(gray, cfg)):
        text = textproc.normalize(recognize(imageproc.upscale(image, factor)))
        score = textproc.reading_score(text)
        result.candidates.append((name, text, score))
        # Uzun ve temiz okuma kazanır; kısmi okuma ("Go to") tam okumayı yenemez.
        if best_score is None or score > best_score:
            best_score, result.text, result.variant = score, text, name
        if (textproc.text_quality(text) >= GOOD_ENOUGH and text.isascii()
                and len(text) >= cfg.min_text_chars):
            break
    if not textproc.is_plausible(result.text, cfg.min_text_chars):
        result.text = ""
    if raw and engine is not None:
        result.confidence = getattr(engine, "last_confidence", None)
        span = getattr(engine, "last_span", None)
        if span is not None:
            result.center_x = left + (span[0] + span[1]) / 2
        box = getattr(engine, "last_box", None)
        if box is not None and result.text:
            result.text_box = (left + box[0], top + box[1], left + box[2], top + box[3])
    return result
