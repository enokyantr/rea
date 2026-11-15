"""reader.read_text: birden çok ön işleme seçeneği arasından en anlamlı OCR okumasını seçer."""

import numpy as np

from altyazi import reader
from altyazi.config import Config


def _gray_with_text():
    gray = np.full((60, 400), 30, dtype=np.uint8)
    gray[20:40, 50:350:5] = 255  # parlak "yazı"
    gray[18:20, 50:350] = 0      # koyu kontur
    return gray


class FakeOcr:
    """Sırayla okuma döndürür ("auto": top-hat, gri ton, kontur maskesi)."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return [self.readings[self.calls - 1]]


def test_picks_the_most_word_like_reading():
    ocr = FakeOcr("Dog, I ajntc sure chac joke 'Jyork3, clog.", "D0g, 1 4jn7 5ur3", "Dog, I ain't sure that joke works, dog.")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Dog, I ain't sure that joke works, dog."
    assert result.variant == "kontur" and [c[0] for c in result.candidates] == ["tophat", "gri", "kontur"]


def test_good_first_reading_skips_other_variants():
    ocr = FakeOcr("Be cool, fool, we got the paperwork.", "should not be called")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Be cool, fool, we got the paperwork." and ocr.calls == 1


def test_garbage_everywhere_returns_nothing():
    ocr = FakeOcr("ntpgmfl/ Wag", "3hjc.", "rjj.")
    assert reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto")).text == ""


def test_no_bright_pixels_skips_ocr():
    ocr = FakeOcr("x")
    result = reader.read_text(np.full((60, 400), 30, dtype=np.uint8), ocr, Config())
    assert result.text == "" and ocr.calls == 0


def test_forced_single_mode():
    ocr = FakeOcr("Lose the Cops.")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="none"))
    assert result.variant == "gri" and ocr.calls == 1


def test_longer_clean_reading_beats_short_partial():
    # Kullanıcının logu: "Go to" (kısa) seçiliyor, "Go to Franklin's house." eleniyordu.
    ocr = FakeOcr("Go to Frånklin's house.", "Go to", "Go")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Go to Franklin's house."  # aksan OCR gürültüsüdür, katlanır


def test_tophat_keeps_thin_strokes_and_drops_wide_bright_background():
    from altyazi.imageproc import tophat

    gray = np.full((40, 200), 40, dtype=np.uint8)
    gray[:, 120:] = 235              # geniş, parlak gökyüzü
    gray[10:30, 20:22] = 250         # ince harf vuruşu (koyu zeminde)
    gray[10:30, 160:162] = 255       # ince harf vuruşu (parlak zeminde, siyah konturlu)
    gray[10:30, 158:160] = 20
    gray[10:30, 162:164] = 20
    out = tophat(gray, radius=3)
    assert out[20, 20] < 60 and out[20, 160] < 128   # yazı koyu
    assert out[20, 140] > 200 and out[20, 60] > 200  # zemin (gökyüzü dahil) beyaz


def test_tophat_keeps_faint_outline_less_text_on_light_background():
    # Kullanıcının ekran görüntüsü: kontursuz beyaz yazı, açık gri beton (fark ~45).
    from altyazi.imageproc import tophat

    gray = np.full((40, 200), 195, dtype=np.uint8)
    gray[10:30, 50:52] = 240
    out = tophat(gray, radius=3)
    assert out[20, 50] < 60 and out[20, 100] > 200


def test_small_text_is_upscaled_more():
    cfg = Config()
    assert reader.upscale_factor(16 + 14, cfg) > 2.5   # ~14 px yazı (düşük çözünürlüklü video)
    assert reader.upscale_factor(16 + 40, cfg) == cfg.upscale
    assert reader.upscale_factor(16 + 4, cfg) == 4.0
