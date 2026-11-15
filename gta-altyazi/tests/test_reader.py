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
    """Seçeneğe göre farklı okuma döndürür: ilk çağrı gri ton, ikinci kontur maskesi."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return [self.readings[self.calls - 1]]


def test_picks_the_most_word_like_reading():
    ocr = FakeOcr("Dog, I ajntc sure chac joke 'Jyork3, clog.", "Dog, I ain't sure that joke works, dog.")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Dog, I ain't sure that joke works, dog."
    assert result.variant == "kontur" and len(result.candidates) == 2


def test_good_first_reading_skips_other_variants():
    ocr = FakeOcr("Be cool, fool, we got the paperwork.", "should not be called")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Be cool, fool, we got the paperwork." and ocr.calls == 1


def test_garbage_everywhere_returns_nothing():
    ocr = FakeOcr("ntpgmfl/ Wag", "3hjc.")
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
    ocr = FakeOcr("Go to Frånklin's house.", "Go to")
    result = reader.read_text(_gray_with_text(), ocr, Config(preprocess="auto"))
    assert result.text == "Go to Franklin's house."  # aksan OCR gürültüsüdür, katlanır
