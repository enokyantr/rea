"""Özel tam ekran (exclusive fullscreen) tespiti: bu modda Windows ekran yakalamasına oyun yerine siyah
(ya da donmuş, tek renk) görüntü verir; overlay de oyunun üstünde görünmez. Ucuz denetim: hedef pencere
varken hem okuma bölgesi hem pencerenin ortası birkaç saniye boyunca tek renkse kullanıcıya bir kez
"kenarlıksız pencere" önerilir. Ara sahnelerdeki siyah bantlar (altyazı bandı) tek başına yetmez:
pencerenin ortası da tek renk olmalı."""

from __future__ import annotations

import numpy as np

# Piksel değerleri arasındaki en büyük fark bu kadar ya da azsa görüntü "tek renk" sayılır
# (sıkıştırılmamış oyun görüntüsünde gerçek sahnede fark onlarca birimdir).
FLAT_TOLERANCE = 6
# Yükleme ekranları/kararmalar birkaç saniye sürebilir: ancak bu kadar sn kesintisiz tek renkse uyar.
FLAT_SECONDS = 8.0

HINT = ("Oyun görüntüsü yakalanamıyor (hep siyah/tek renk). Oyun büyük ihtimalle özel tam ekran "
        "(exclusive fullscreen) modunda: oyunun görüntü ayarlarında \"Kenarlıksız pencere (borderless "
        "windowed)\" ya da \"Pencereli tam ekran\" seçin.")


def is_flat(image: np.ndarray | None, tolerance: int = FLAT_TOLERANCE) -> bool:
    """Görüntü (BGRA/BGR/gri) tek renk mi? Seyrek örnekleme: büyük karede de ~mikrosaniyeler."""
    if image is None:
        return False
    arr = np.asarray(image)
    sample = arr[::4, ::4, :3] if arr.ndim == 3 else arr[::4, ::4]
    if sample.size == 0:
        return False
    return int(sample.max()) - int(sample.min()) <= tolerance


class FlatCaptureWatch:
    """`feed` saniyede bir çağrılır; kesintisiz `seconds` sn tek renk görülünce bir kez True döner."""

    def __init__(self, seconds: float = FLAT_SECONDS):
        self.seconds = seconds
        self.since: float | None = None
        self.warned = False

    def feed(self, flat: bool, window_found: bool, now: float) -> bool:
        if self.warned:
            return False
        if not (flat and window_found):
            self.since = None
            return False
        if self.since is None:
            self.since = now
            return False
        if now - self.since >= self.seconds:
            self.warned = True
            return True
        return False
