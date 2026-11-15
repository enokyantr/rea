"""Görüntü hazırlama ve değişim tespiti (platformdan bağımsız, numpy)."""

from __future__ import annotations

import numpy as np


def to_gray(bgra: np.ndarray) -> np.ndarray:
    b, g, r = bgra[..., 0], bgra[..., 1], bgra[..., 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)


def preprocess(bgra: np.ndarray, mode: str, bright_threshold: int) -> np.ndarray:
    """OCR için gri ton görüntü döndürür.

    "bright": sadece parlak pikseller (beyaz altyazı) siyah yazı / beyaz zemin olarak kalır;
    arkadaki hareketli oyun görüntüsü büyük ölçüde temizlenir.
    """
    gray = to_gray(bgra)
    if mode == "bright":
        return np.where(gray >= bright_threshold, 0, 255).astype(np.uint8)
    return gray


def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    """Küçük yazıyı büyütmek OCR doğruluğunu belirgin artırır."""
    if factor <= 1.0:
        return gray
    from PIL import Image

    h, w = gray.shape
    img = Image.fromarray(gray).resize((round(w * factor), round(h * factor)), Image.BICUBIC)
    return np.asarray(img)


def signature(bgra: np.ndarray, bright_threshold: int, cols: int = 96, rows: int = 24) -> np.ndarray:
    """Parlak yazı piksellerinin kaba haritası. Arka plan hareket etse de yazı aynıysa
    büyük ölçüde aynı kalır; böylece gereksiz OCR çağrıları atlanır."""
    mask = to_gray(bgra) >= bright_threshold
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((rows, cols), dtype=bool)
    ys = (np.arange(rows) * h) // rows
    xs = (np.arange(cols) * w) // cols
    # Her hücrede herhangi bir parlak piksel var mı (ızgara örneklemesi + yatay genişletme).
    sampled = np.logical_or.reduceat(np.logical_or.reduceat(mask, ys, axis=0), xs, axis=1)
    return sampled


def same_signature(a: np.ndarray | None, b: np.ndarray, tolerance: float = 0.01) -> bool:
    if a is None or a.shape != b.shape:
        return False
    return float(np.mean(a != b)) <= tolerance
