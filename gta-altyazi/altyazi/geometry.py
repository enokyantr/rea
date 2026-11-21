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


def union(a: Rect, b: Rect) -> Rect:
    left, top = min(a.left, b.left), min(a.top, b.top)
    return Rect(left, top, max(a.right, b.right) - left, max(a.bottom, b.bottom) - top)


def intersect(a: Rect, b: Rect) -> Rect | None:
    left, top = max(a.left, b.left), max(a.top, b.top)
    right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return None
    return Rect(left, top, right - left, bottom - top)


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


def fit_aspect(rect: Rect, aspect: float) -> Rect:
    """Pencerede en-boy oranı korunarak ortalanan görüntünün alanı (kenarlardaki siyah bantlar
    hariç). Pencere zaten bu orandaysa (±%1) olduğu gibi döner."""
    if aspect <= 0 or rect.width <= 0 or rect.height <= 0:
        return rect
    current = rect.width / rect.height
    if abs(current / aspect - 1) <= 0.01:
        return rect
    if current > aspect:  # geniş pencere: sağda/solda bant
        width = round(rect.height * aspect)
        return Rect(rect.left + (rect.width - width) // 2, rect.top, width, rect.height)
    height = round(rect.width / aspect)
    return Rect(rect.left, rect.top + (rect.height - height) // 2, rect.width, height)


# Başlığında bunlar geçen pencere tarayıcı sekmesidir (ör. chiaki-ng'nin GitHub sayfası): son seçenek.
_BROWSERS = ("google chrome", "mozilla firefox", "microsoft edge", "microsoft\u200b edge", "opera", "brave",
             "vivaldi", "yandex")


# window_titles içinde "öndeki pencere" anlamına gelen işaret (bu programın pencereleri hariç).
ACTIVE = "@aktif"
DEFAULT_TITLES = ("PS Remote Play", "Chiaki")
MAX_TITLES = 8


def pick_window(title_parts: list[str], windows: list[tuple[str, Rect]],
                active_title: str | None = None) -> tuple[str, Rect] | None:
    """Hedef pencere seçimi. Öncelik: tarayıcı sekmesi olmayan > listede önce gelen ifade (menüden
    seçilen oyun varsayılanların önüne eklenir) > başlığı ifadeyle BAŞLAYAN ("chiaki-ng | ...",
    "PS Remote Play") > en büyük pencere. Tarayıcı sekmesi ("chiaki-ng - GitHub - Google Chrome") son
    seçenek. `ACTIVE` ifadesi başlığı `active_title` olan pencereyle (öndeki pencere) eşleşir."""
    needles = [p.casefold() for p in title_parts if p]
    active = active_title.casefold().strip() if active_title else None
    best, best_key = None, None
    for title, rect in windows:
        folded = title.casefold().strip()
        match = None
        for idx, needle in enumerate(needles):
            if needle == ACTIVE:
                if active is not None and folded == active:
                    match = (idx, True)
                    break
            elif needle in folded:
                match = (idx, folded.startswith(needle))
                break
        if match is None:
            continue
        browser = any(b in folded for b in _BROWSERS)
        key = (not browser, -match[0], match[1], rect.width * rect.height)
        if best_key is None or key > best_key:
            best, best_key = (title, rect), key
    return best


def choose_target(current: list[str], choice: str) -> list[str]:
    """Menüde seçilen hedefi `window_titles` listesine yazar: seçilen başlık (ya da `ACTIVE`) başa gelir,
    daha önce seçilmiş oyunlar ve en sonda varsayılanlar (Remote Play/chiaki) yedek olarak kalır.
    `choice` boşsa yalnızca varsayılanlar."""
    choice = choice.strip()
    defaults = list(DEFAULT_TITLES)
    if not choice:
        return defaults
    folded = {d.casefold() for d in defaults} | {choice.casefold(), ACTIVE}
    # "@aktif" yalnızca seçildiğinde başta durur: belirli bir oyun seçilince yedek olarak kalırsa
    # oyun kapalıyken öndeki herhangi bir pencere (Discord...) okunurdu.
    previous = [t for t in current if t.strip() and t.casefold() not in folded]
    keep = MAX_TITLES - 1 - len(defaults)
    return [choice] + previous[:keep] + [d for d in defaults if d.casefold() != choice.casefold()]


def window_choices(windows: list[tuple[str, Rect]], min_width: int = 320, min_height: int = 200) -> list[str]:
    """Hedef seçimi için pencere başlıkları: küçük araç pencereleri elenir, aynı başlık bir kez,
    büyük pencere önce (oyun genelde en büyüğüdür)."""
    seen: set[str] = set()
    out = []
    for title, rect in sorted(windows, key=lambda w: -w[1].width * w[1].height):
        title = title.strip()
        if not title or rect.width < min_width or rect.height < min_height or title.casefold() in seen:
            continue
        seen.add(title.casefold())
        out.append(title)
    return out
