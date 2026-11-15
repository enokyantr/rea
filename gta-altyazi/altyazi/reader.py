"""Altyazı bölgesinden metin okuma: ön işleme seçenekleri + en anlamlı OCR sonucunu seçme.

Tek bir ön işleme her sahnede işe yaramıyor (gerçek video: gri ton koyu sahnede kusursuz, açık
sahnede t→c/h→ij hatalı; sıkı kontur maskesi sıkıştırılmış videoda harfleri parçalıyor). Bu yüzden
"auto" modunda birkaç seçenek OCR'lanır ve gerçek metne en çok benzeyen sonuç seçilir. OCR satır
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
    # (seçenek, okunan metin, kalite) — --debug/--dump için
    candidates: list[tuple[str, str, float]] = field(default_factory=list)
    crop: np.ndarray | None = None


def variants(gray: np.ndarray, cfg) -> list[tuple[str, np.ndarray]]:
    """Denenecek OCR girdileri, sırayla. `cfg.preprocess`: auto | none | bright | outline."""
    mode = cfg.preprocess
    out: list[tuple[str, np.ndarray]] = []
    if mode in ("auto", "none"):
        out.append(("gri", gray))
    if mode in ("auto", "outline"):
        mask = imageproc.text_mask(gray, "outline", cfg.bright_threshold, cfg.dark_threshold,
                                   cfg.outline_radius, cfg.outline_white_threshold)
        out.append(("kontur", imageproc.for_ocr(gray, mask, "outline")))
    if mode == "bright":
        out.append(("parlak", imageproc.for_ocr(gray, gray >= cfg.bright_threshold, "bright")))
    return out


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
    best_quality = -1.0
    for name, image in variants(gray, cfg):
        text = textproc.normalize(recognize(imageproc.upscale(image, cfg.upscale)))
        quality = textproc.text_quality(text)
        result.candidates.append((name, text, quality))
        # Eşit kalitede uzun okuma kazanır (kısmi okumalar kısadır).
        if quality > best_quality or (quality == best_quality and len(text) > len(result.text)):
            best_quality, result.text, result.variant = quality, text, name
        if quality >= GOOD_ENOUGH and len(text) >= cfg.min_text_chars:
            break
    if not textproc.is_plausible(result.text, cfg.min_text_chars):
        result.text = ""
    return result
