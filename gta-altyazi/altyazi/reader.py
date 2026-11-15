"""Altyazı bölgesinden metin okuma: ön işleme seçenekleri + en anlamlı OCR sonucunu seçme.

Tek bir ön işleme her sahnede işe yaramıyor (gerçek video: gri ton koyu sahnede kusursuz, açık
sahnede t→c/h→ij hatalı; sıkı kontur maskesi sıkıştırılmış videoda harfleri parçalıyor). Bu yüzden
"auto" modunda birkaç seçenek (top-hat, gri, kontur) OCR'lanır ve gerçek metne en çok benzeyen sonuç seçilir. OCR satır
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


def variants(gray: np.ndarray, cfg, mode: str | None = None) -> list[tuple[str, np.ndarray]]:
    """Denenecek OCR girdileri, sırayla. `cfg.preprocess`: auto | tophat | none | bright | outline.
    "all" (yalnızca --ocr-compare): hepsi."""
    mode = mode or cfg.preprocess
    out: list[tuple[str, np.ndarray]] = []
    if mode in ("auto", "tophat", "all"):
        out.append(("tophat", imageproc.tophat(gray, cfg.tophat_radius)))
    if mode in ("auto", "none", "all"):
        out.append(("gri", gray))
    if mode in ("auto", "outline", "all"):
        mask = imageproc.text_mask(gray, "outline", cfg.bright_threshold, cfg.dark_threshold,
                                   cfg.outline_radius, cfg.outline_white_threshold)
        out.append(("kontur", imageproc.for_ocr(gray, mask, "outline")))
    if mode in ("bright", "all"):
        out.append(("parlak", imageproc.for_ocr(gray, gray >= cfg.bright_threshold, "bright")))
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


def read_text(gray: np.ndarray, recognize: Callable[[np.ndarray], list[str]], cfg) -> ReadResult:
    """`gray`: bölgenin gri tonu. `recognize`: görüntü -> satırlar (Windows OCR)."""
    bbox = imageproc.bright_bbox(gray >= cfg.bright_threshold)
    if bbox is None:
        if cfg.skip_without_bright_text:
            return ReadResult("")
    else:
        top, bottom, left, right = bbox
        gray = gray[top:bottom, left:right]

    result = ReadResult("", crop=gray)
    factor = upscale_factor(gray.shape[0] if bbox is not None else 0, cfg)
    best_score = None
    for name, image in variants(gray, cfg):
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
    return result
