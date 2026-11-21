import numpy as np

from altyazi.contrast import BLACK, WHITE, BackgroundStats, ColorPicker, background_stats, contrast_ratio


def solid(rgb, h=40, w=200):
    a = np.empty((h, w, 4), dtype=np.uint8)
    a[..., 0], a[..., 1], a[..., 2], a[..., 3] = rgb[2], rgb[1], rgb[0], 255
    return a


def test_background_stats_luminance_range():
    assert background_stats(solid((255, 255, 255))).mid > 0.99
    assert background_stats(solid((0, 0, 0))).mid < 0.01
    assert background_stats(np.zeros((0, 0, 4), dtype=np.uint8)) is None


def test_contrast_ratio_matches_wcag_extremes():
    assert abs(contrast_ratio(1.0, 0.0) - 21.0) < 1e-9
    assert contrast_ratio(0.5, 0.5) == 1.0


def test_switches_to_black_on_white_scene_and_back():
    p = ColorPicker(base_outline_px=2)
    assert p.palette.fill == WHITE
    dark = background_stats(solid((20, 25, 30)))
    snow = background_stats(solid((240, 245, 250)))
    assert p.update(dark) is None  # zaten beyaz
    changed = None
    for _ in range(3):  # yumuşatma: en geç birkaç karede
        changed = p.update(snow) or changed
    assert changed is not None and changed.fill == BLACK and changed.outline == WHITE
    for _ in range(3):
        p.update(dark)
    assert p.palette.fill == WHITE and p.palette.outline == BLACK


def test_reacts_within_two_frames():
    p = ColorPicker(base_outline_px=2)
    p.update(background_stats(solid((10, 10, 10))))
    snow = background_stats(solid((250, 250, 250)))
    p.update(snow)
    p.update(snow)
    assert p.palette.fill == BLACK


def test_single_flash_does_not_flip():
    p = ColorPicker(base_outline_px=2)
    dark = background_stats(solid((15, 15, 15)))
    for _ in range(5):
        p.update(dark)
    # Tek karelik çok kısa parlama (yarım parlaklıkta) rengi değiştirmemeli.
    p.update(BackgroundStats(0.1, 0.3, 0.5))
    assert p.palette.fill == WHITE


def test_hysteresis_prevents_flicker_near_midtone():
    p = ColorPicker(base_outline_px=2)
    flips = 0
    last = p.palette.fill
    for i in range(40):  # ~orta ton, hafif dalgalanan sahne
        mid = 0.18 + (0.02 if i % 2 else -0.02)
        p.update(BackgroundStats(mid - 0.05, mid, mid + 0.05))
        if p.palette.fill != last:
            flips += 1
            last = p.palette.fill
    assert flips <= 1


def test_thicker_outline_on_mixed_scene():
    p = ColorPicker(base_outline_px=2)
    mixed = BackgroundStats(0.01, 0.2, 0.95)  # yarısı karanlık, yarısı parlak
    for _ in range(4):
        p.update(mixed)
    assert p.palette.outline_px == 3
    clean = BackgroundStats(0.0, 0.005, 0.01)
    for _ in range(6):
        p.update(clean)
    assert p.palette.outline_px == 2
