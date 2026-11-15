"""Ekran dikdörtgeni ve oransal bölge dönüşümleri (platformdan bağımsız)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Region


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def region_to_screen(region: Region, base: Rect) -> Rect:
    left = base.left + round(_clamp(region.left) * base.width)
    top = base.top + round(_clamp(region.top) * base.height)
    right = base.left + round(_clamp(region.left + region.width) * base.width)
    bottom = base.top + round(_clamp(region.top + region.height) * base.height)
    return Rect(left, top, max(1, right - left), max(1, bottom - top))


def screen_to_region(rect: Rect, base: Rect) -> Region:
    """Seçilen ekran alanını pencereye göre orana çevirir; pencere dışına taşan kısım kırpılır."""
    left = _clamp((rect.left - base.left) / base.width)
    top = _clamp((rect.top - base.top) / base.height)
    right = _clamp((rect.right - base.left) / base.width)
    bottom = _clamp((rect.bottom - base.top) / base.height)
    return Region(
        left=round(left, 4),
        top=round(top, 4),
        width=round(max(0.0, right - left), 4),
        height=round(max(0.0, bottom - top), 4),
    )
