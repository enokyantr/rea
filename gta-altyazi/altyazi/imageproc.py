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


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Kare yapı elemanıyla genişletme (ayrılabilir: önce satır, sonra sütun)."""
    if radius <= 0:
        return mask
    h, w = mask.shape
    padded = np.pad(mask, radius)
    rows = np.zeros((h + 2 * radius, w), dtype=bool)
    for dx in range(2 * radius + 1):
        rows |= padded[:, dx:dx + w]
    out = np.zeros((h, w), dtype=bool)
    for dy in range(2 * radius + 1):
        out |= rows[dy:dy + h, :]
    return out


def text_mask(gray: np.ndarray, mode: str, bright_threshold: int, dark_threshold: int = 90,
              radius: int = 3, white_threshold: int = 232) -> np.ndarray:
    """Yazı pikselleri. Değişim tespiti, kırpma ve OCR girdisi hep bu maskeden üretilir.

    "outline" (varsayılan): neredeyse saf beyaz (>= white_threshold) VE yakınında koyu piksel olan
    noktalar. Oyun altyazısı beyaz yazı + siyah konturdur. Konturu olmayan parlak gökyüzü/kum/duvar
    maskeye girmez; beyaz eşiği de harflerin hemen dışındaki ve içindeki (o, a, e) açık arka planı
    dışarıda tutar. (Yalnızca parlaklıkla ayırınca açık sahnelerde harf gövdeleri arka plana
    karışıyor, t→c, h→ij, d→cl okunuyordu.) Çok parlak (aşırı pozlanmış) gökyüzünde zayıflar.
    "bright" / "none": yalnızca parlaklık (>= bright_threshold)."""
    if mode != "outline":
        return gray >= bright_threshold
    return (gray >= white_threshold) & _dilate(gray <= dark_threshold, radius)


def for_ocr(gray: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """OCR girdisi: "none" modunda gri ton; diğerlerinde maskeden siyah yazı / beyaz zemin."""
    if mode == "none":
        return gray
    return np.where(mask, 0, 255).astype(np.uint8)


def _minmax(gray: np.ndarray, radius: int, fn) -> np.ndarray:
    """Kare pencerede ayrılabilir min/max süzgeci (kenarlar tekrarlanarak doldurulur)."""
    h, w = gray.shape
    p = np.pad(gray, radius, mode="edge")
    rows = p[:, 0:w].copy()
    for dx in range(1, 2 * radius + 1):
        fn(rows, p[:, dx:dx + w], out=rows)
    out = rows[0:h].copy()
    for dy in range(1, 2 * radius + 1):
        fn(out, rows[dy:dy + h], out=out)
    return out


def tophat(gray: np.ndarray, radius: int = 3, floor: int = 25) -> np.ndarray:
    """Beyaz top-hat: her pikselin yerel arka plandan (açma) ne kadar parlak olduğu.

    İnce parlak harf vuruşları kalır; harften geniş parlak alanlar (gökyüzü, kar, duvar) silinir.
    Siyah yazı / beyaz zemin olarak, en parlak %10'a göre gerilmiş döner. Sentetik, sıkıştırılmış
    altyazı kırpıntılarında kontur maskesinden belirgin daha iyi okundu (CER 0.45 -> 0.09)."""
    opened = _minmax(_minmax(gray, radius, np.minimum), radius, np.maximum)
    th = gray.astype(np.int16) - opened
    # Germe: en parlak vuruşlara göre. Sabit bir eşik (ör. 60) kontursuz, açık zemindeki silik
    # altyazıyı (açık gri betonda beyaz yazı: fark ~40-50) tamamen siliyordu.
    hi = max(int(np.percentile(th, 99.5)), floor)
    return (255 - np.clip(th * (255.0 / hi), 0, 255)).astype(np.uint8)


def bright_bbox(mask: np.ndarray, margin: int = 8, min_pixels: int = 12):
    """Yazı piksellerinin sınır kutusu (top, bottom, left, right) ya da None.

    OCR'a yalnızca bu kısmı vermek görüntüyü küçültür: OCR hızlanır, kenardaki gürültü azalır."""
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


def signature(mask: np.ndarray, cols: int = 96, rows: int = 24) -> np.ndarray:
    """Yazı maskesinin kaba haritası. Arka plan hareket etse de yazı aynıysa büyük ölçüde aynı
    kalır; böylece gereksiz OCR çağrıları atlanır."""
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


def box_signature(gray: np.ndarray, box: tuple[int, int, int, int], threshold: int,
                  pad: int = 4) -> np.ndarray | None:
    """Yazı kutusunun içindeki parlak piksellerin kaba haritası. Altyazı aynı kaldıkça (arkadaki
    video oynasa da) değişmez; satır değişince ya da silinince belirgin değişir."""
    h, w = gray.shape
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    return signature(gray[y0:y1, x0:x1] >= threshold, cols=96, rows=8)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Bulanık (720p civarı akış) yazıyı keskinleştirir (keskinlik maskesi)."""
    from PIL import Image, ImageFilter

    return np.asarray(Image.fromarray(gray).filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2)))


def median3(gray: np.ndarray) -> np.ndarray:
    """Sıkıştırma bloklarını/kumlanmayı (çok düşük bit hızı) yumuşatır, harf kenarını korur."""
    from PIL import Image, ImageFilter

    return np.asarray(Image.fromarray(gray).filter(ImageFilter.MedianFilter(3)))

