"""Özel tam ekran: yakalanan görüntü sürekli siyah/tek renkse bir kez "kenarlıksız pencere" önerilir."""

import numpy as np

from altyazi.fullscreen import FlatCaptureWatch, is_flat


def test_flat_detection_ignores_real_scenes():
    black = np.zeros((90, 400, 4), np.uint8)
    assert is_flat(black)
    assert is_flat(black + 3)                     # neredeyse siyah (renk dönüşümü gürültüsü)
    scene = np.random.default_rng(1).integers(0, 255, (90, 400, 4), dtype=np.uint8)
    assert not is_flat(scene)
    subtitle = black.copy()
    subtitle[40:50, 100:300] = 250                # siyah bant üstünde altyazı
    assert not is_flat(subtitle)
    assert not is_flat(None) and not is_flat(np.zeros((0, 0, 4), np.uint8))


def test_hint_once_after_continuous_flat_seconds_with_window():
    watch = FlatCaptureWatch(seconds=8)
    assert not any(watch.feed(True, True, t) for t in range(0, 8))   # 7 sn: yükleme ekranı olabilir
    assert watch.feed(True, True, 8)
    assert not watch.feed(True, True, 20)                            # yalnızca bir kez

    watch = FlatCaptureWatch(seconds=8)
    for t in range(0, 6):
        watch.feed(True, True, t)
    assert not watch.feed(False, True, 6)                            # görüntü geldi: sayaç sıfırlanır
    assert not any(watch.feed(True, True, t) for t in range(7, 15))
    assert not any(watch.feed(True, False, t) for t in range(15, 40))  # hedef pencere yok: uyarı yok
