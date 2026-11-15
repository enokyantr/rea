"""Görüntü hazırlama ve değişim tespiti (platformdan bağımsız, numpy).

Yakalanan BGRA kare bir kez gri tona çevrilir; diğer fonksiyonlar bu gri görüntüyü kullanır."""

from __future__ import annotations

import numpy as np


def to_gray(bgra: np.ndarray) -> np.ndarray:
    # Tamsayı aritmetiği: float ara dizilerden daha hızlı ve daha az bellek.
    b = bgra[..., 0].astype(np.uint16)
    g = bgra[..., 1].astype(np.uint16)
    r = bgra[..., 2].astype(np.uint16)
    return ((29 * b + 150 * g + 77 * r) >> 8).astype(np.uint8)


def for_ocr(gray: np.ndarray, mode: str, bright_threshold: int) -> np.ndarray:
    """"bright": yalnızca parlak pikseller (beyaz altyazı) siyah yazı / beyaz zemin olarak kalır;
    arkadaki hareketli oyun görüntüsü büyük ölçüde temizlenir."""
    if mode == "bright":
        return np.where(gray >= bright_threshold, 0, 255).astype(np.uint8)
    return gray


def bright_bbox(gray: np.ndarray, bright_threshold: int, margin: int = 8, min_pixels: int = 12):
    """Parlak (yazı) piksellerinin sınır kutusu (top, bottom, left, right) ya da None.

    OCR'a yalnızca bu kısmı vermek görüntüyü küçültür: OCR hızlanır, kenardaki gürültü azalır."""
    mask = gray >= bright_threshold
    if int(np.count_nonzero(mask)) < min_pixels:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    h, w = mask.shape
    return (
        max(0, int(rows[0]) - margin),
        min(h, int(rows[-1]) + 1 + margin),
        max(0, int(cols[0]) - margin),
        min(w, int(cols[-1]) + 1 + margin),
    )


def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    """Küçük yazıyı büyütmek OCR doğruluğunu belirgin artırır."""
    if factor <= 1.0:
        return gray
    from PIL import Image

    h, w = gray.shape
    img = Image.fromarray(gray).resize((round(w * factor), round(h * factor)), Image.BICUBIC)
    return np.asarray(img)


def signature(gray: np.ndarray, bright_threshold: int, cols: int = 96, rows: int = 24) -> np.ndarray:
    """Parlak yazı piksellerinin kaba haritası. Arka plan hareket etse de yazı aynıysa
    büyük ölçüde aynı kalır; böylece gereksiz OCR çağrıları atlanır."""
    mask = gray >= bright_threshold
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((rows, cols), dtype=bool)
    ys = (np.arange(rows) * h) // rows
    xs = (np.arange(cols) * w) // cols
    return np.logical_or.reduceat(np.logical_or.reduceat(mask, ys, axis=0), xs, axis=1)


def same_signature(a: np.ndarray | None, b: np.ndarray, tolerance: float = 0.01) -> bool:
    if a is None or a.shape != b.shape:
        return False
    return float(np.mean(a != b)) <= tolerance
