"""Arka plana göre en okunaklı yazı rengini seçme (WCAG kontrast oranı, platformdan bağımsız)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# sRGB (0..255) -> doğrusal ışık; WCAG göreli parlaklık hesabı için.
_c = np.arange(256, dtype=np.float64) / 255.0
_LINEAR = np.where(_c <= 0.04045, _c / 12.92, ((_c + 0.055) / 1.055) ** 2.4).astype(np.float32)

WHITE, BLACK = "#FFFFFF", "#000000"
_FILL_LUMINANCE = {WHITE: 1.0, BLACK: 0.0}
# Bu kontrastın altında (orta tonlu ya da çok karışık sahne) kontur kalınlaştırılır.
_THICK_BELOW = 4.5


@dataclass(frozen=True)
class BackgroundStats:
    """Yazının çizileceği alandaki parlaklık dağılımı (0 = siyah, 1 = beyaz)."""

    dark: float  # 10. yüzdelik
    mid: float  # medyan
    bright: float  # 90. yüzdelik


@dataclass(frozen=True)
class Palette:
    fill: str
    outline: str
    outline_px: int


def background_stats(bgra: np.ndarray, step: int = 4) -> BackgroundStats | None:
    px = bgra[::step, ::step]
    if px.size == 0:
        return None
    lum = 0.2126 * _LINEAR[px[..., 2]] + 0.7152 * _LINEAR[px[..., 1]] + 0.0722 * _LINEAR[px[..., 0]]
    dark, mid, bright = np.percentile(lum, (10, 50, 90))
    return BackgroundStats(float(dark), float(mid), float(bright))


def contrast_ratio(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def score(fill: str, bg: BackgroundStats) -> float:
    """Ağırlık çoğunlukla tipik arka planda; karışık sahnede en kötü kısım da hesaba katılır."""
    lum = _FILL_LUMINANCE[fill]
    worst = min(contrast_ratio(lum, bg.dark), contrast_ratio(lum, bg.bright))
    return 0.6 * contrast_ratio(lum, bg.mid) + 0.4 * worst


class ColorPicker:
    """Arka plan değiştikçe yazı rengini değiştirir.

    - Kısa yumuşatma (EMA) tek karelik patlamaları (namlu alevi, flaş) eler ama 1–2 karede tepki verir.
    - Histerezis: diğer renk belirgin şekilde daha iyi olmadıkça renk değişmez (titreme olmaz).
    """

    def __init__(self, base_outline_px: int, smoothing: float = 0.6, switch_margin: float = 1.15):
        self.base_outline_px = base_outline_px
        self.smoothing = smoothing
        self.switch_margin = switch_margin
        self._bg: BackgroundStats | None = None
        self.palette = Palette(WHITE, BLACK, base_outline_px)

    def update(self, stats: BackgroundStats) -> Palette | None:
        """Yeni ölçüm verir; renk değiştiyse yeni paleti döndürür."""
        if self._bg is None:
            self._bg = stats
        else:
            a = self.smoothing
            self._bg = BackgroundStats(
                a * stats.dark + (1 - a) * self._bg.dark,
                a * stats.mid + (1 - a) * self._bg.mid,
                a * stats.bright + (1 - a) * self._bg.bright,
            )
        current = self.palette.fill
        other = BLACK if current == WHITE else WHITE
        current_score, other_score = score(current, self._bg), score(other, self._bg)
        fill = other if other_score > current_score * self.switch_margin else current
        best = other_score if fill == other else current_score

        thick = self.palette.outline_px > self.base_outline_px
        # Kalınlık için de küçük bir histerezis bandı.
        if best < _THICK_BELOW - (0 if thick else 0.5):
            thick = True
        elif best > _THICK_BELOW + 0.5:
            thick = False
        new = Palette(fill, BLACK if fill == WHITE else WHITE, self.base_outline_px + (1 if thick else 0))
        if new != self.palette:
            self.palette = new
            return new
        return None
