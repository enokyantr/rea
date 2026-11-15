import json

import numpy as np

from altyazi import config as config_mod
from altyazi.config import Region
from altyazi.geometry import Rect, region_to_screen, screen_to_region
from altyazi.imageproc import bright_bbox, for_ocr, same_signature, signature, to_gray
from altyazi.textproc import Clear, Show, Stabilizer, is_plausible, normalize
from altyazi.translate import CachedTranslator


def test_normalize_joins_lines_and_cleans_edges():
    assert normalize(["  You think this is ", "a game, Jason?  "]) == "You think this is a game, Jason?"
    assert normalize(["| Get in the car!"]) == "Get in the car!"
    assert normalize(["We need some- thing fast."]) == "We need something fast."
    assert normalize(["", "   "]) == ""


def test_plausible_rejects_noise():
    assert is_plausible("Get in the car!", 3)
    assert not is_plausible("|| -- ..", 3)
    assert not is_plausible("ab", 3)


def test_stabilizer_requires_repeated_reads_and_ignores_jitter():
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.5)
    assert s.feed("Get in the car", 0.0) is None
    assert s.feed("Get in the car!", 0.1) == Show("Get in the car!", 0.0)
    # OCR titremesi: neredeyse aynı metin tekrar gösterilmez
    assert s.feed("Get in the car.", 0.2) is None
    assert s.feed("Get in the car!", 0.3) is None
    # Yeni satır
    assert s.feed("Where are we going?", 0.4) is None
    assert s.feed("Where are we going?", 0.5) == Show("Where are we going?", 0.4)


def test_stabilizer_clears_after_timeout_only():
    s = Stabilizer(stable_reads=1, similarity=0.88, clear_after=0.5)
    assert s.feed("Hello there", 0.0) == Show("Hello there", 0.0)
    assert s.feed("", 0.2) is None  # kısa boşluk (OCR kaçırdı) silmez
    assert s.feed("Hello there", 0.3) is None
    assert s.feed("", 0.5) is None
    assert s.feed("", 0.81) == Clear()
    assert s.feed("", 1.5) is None
    # Aynı satır tekrar gelirse yeniden gösterilir
    assert s.feed("Hello there", 2.0) == Show("Hello there", 2.0)


def test_stabilizer_keeps_longest_candidate():
    s = Stabilizer(stable_reads=3, similarity=0.8, clear_after=1)
    s.feed("I told you we should", 0.0)
    s.feed("I told you we should go", 0.1)
    assert s.feed("I told you we should go", 0.2) == Show("I told you we should go", 0.0)


def test_region_roundtrip_and_clipping():
    base = Rect(100, 50, 1920, 1080)
    region = Region(0.15, 0.8, 0.7, 0.15)
    rect = region_to_screen(region, base)
    assert rect == Rect(100 + 288, 50 + 864, 1344, 162)
    back = screen_to_region(rect, base)
    assert abs(back.left - 0.15) < 1e-3 and abs(back.width - 0.7) < 1e-3
    # Pencere dışına taşan seçim kırpılır
    clipped = screen_to_region(Rect(0, 1000, 5000, 500), base)
    assert clipped.left == 0 and clipped.width == 1.0 and clipped.top + clipped.height == 1.0


def _frame_with_text(seed: int, text_on: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 150, size=(60, 400, 4), dtype=np.uint8)  # hareketli koyu arka plan
    if text_on:
        frame[20:40, 50:350:7, :3] = 255  # beyaz "yazı"
    return frame


def test_signature_ignores_moving_background_but_sees_text_change():
    a = signature(to_gray(_frame_with_text(1, True)), 190)
    b = signature(to_gray(_frame_with_text(2, True)), 190)
    c = signature(to_gray(_frame_with_text(3, False)), 190)
    assert same_signature(a, b)
    assert not same_signature(a, c)
    assert not same_signature(None, a)


def test_bright_mode_makes_dark_text_on_white():
    out = for_ocr(to_gray(_frame_with_text(1, True)), "bright", 190)
    assert out[30, 50] == 0 and out[5, 5] == 255


def test_bright_bbox_crops_to_text_with_margin():
    gray = to_gray(_frame_with_text(1, True))
    assert bright_bbox(gray, 190, margin=4) == (16, 44, 46, 349)
    assert bright_bbox(to_gray(_frame_with_text(1, False)), 190) is None


class CountingBackend:
    def __init__(self):
        self.calls = []

    def translate(self, text, context):
        self.calls.append((text, context))
        return f"tr:{text}"


def test_cache_persists_and_skips_repeat_calls(tmp_path):
    path = tmp_path / "cache.jsonl"
    backend = CountingBackend()
    t = CachedTranslator(backend, path)
    assert t.translate("Get in the car!", context="Hey") == "tr:Get in the car!"
    assert t.translate("get in  the car!") == "tr:Get in the car!"
    assert len(backend.calls) == 1
    path.write_text(path.read_text(encoding="utf-8") + "bozuk satir\n", encoding="utf-8")

    backend2 = CountingBackend()
    t2 = CachedTranslator(backend2, path)
    assert t2.translate("Get in the car!") == "tr:Get in the car!"
    assert backend2.calls == []


def test_config_defaults_and_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    cfg = config_mod.load(path)
    assert path.exists() and cfg.target_lang == "TR"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["region"] = {"left": 0.1, "top": 0.7, "width": 0.8, "height": 0.2}
    data["eski_alan"] = 1
    data["font_size"] = 30
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = config_mod.load(path)
    assert cfg.region == Region(0.1, 0.7, 0.8, 0.2) and cfg.font_size == 30


def test_stabilizer_reset_shows_same_line_again():
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.5)
    s.feed("Get in the car!", 0.0)
    assert s.feed("Get in the car!", 0.1) == Show("Get in the car!", 0.0)
    s.reset()  # duraklat/devam: ekranda hâlâ aynı satır var
    assert s.feed("Get in the car!", 0.2) is None
    assert s.feed("Get in the car!", 0.3) == Show("Get in the car!", 0.2)


def test_stabilizer_retries_after_failed_translation():
    s = Stabilizer(stable_reads=1, similarity=0.88, clear_after=0.5)
    assert s.feed("Hello there", 0.0) == Show("Hello there", 0.0)
    assert s.feed("Hello there", 0.1) is None
    s.forget_shown()
    assert s.feed("Hello there", 0.2) == Show("Hello there", 0.2)


def test_stabilizer_pending_while_unconfirmed():
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.5)
    assert not s.pending
    s.feed("Get in the car!", 0.0)
    assert s.pending  # doğrulama okuması hemen yapılmalı
    s.feed("Get in the car!", 0.01)
    assert not s.pending
