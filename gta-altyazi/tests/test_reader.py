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
    """Sırayla okuma döndürür ("auto": kontur maskesi, parlak pikseller, gri ton)."""

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
    assert result.variant == "gri" and [c[0] for c in result.candidates] == ["kontur", "parlak", "gri"]


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


def test_group_lines_joins_boxes_split_on_one_line_and_orders_lines():
    from altyazi.ocr import group_lines

    # Gerçek RapidOCR çıktısı biçimi: bir satır iki kutuya bölünmüş + altında ikinci satır
    boxes = [
        ([[300, 2], [600, 2], [600, 30], [300, 30]], "jee ass haircut you got", 0.98),
        ([[0, 0], [290, 0], [290, 31], [0, 31]], "Maybe if you got rid of that oldkyee-y", 0.94),
        ([[40, 40], [500, 40], [500, 70], [40, 70]], "second line", 0.9),
    ]
    assert group_lines(boxes) == ["Maybe if you got rid of that oldkyee-y jee ass haircut you got", "second line"]


def test_raw_engine_gets_the_crop_as_is():
    seen = []

    def recognize(image):
        seen.append(image)
        return ["Go to Franklin's house."]

    gray = _gray_with_text()
    result = reader.read_text(gray, recognize, Config(preprocess="auto"), raw=True)
    assert result.text == "Go to Franklin's house." and result.variant == "ham"
    assert len(seen) == 1 and seen[0].shape[0] < gray.shape[0]  # yalnızca kırpıldı, büyütülmedi
    assert set(np.unique(seen[0])) - {0, 255}  # ikili maske değil, gri ton


def test_rapidocr_reads_an_outline_less_subtitle_on_light_background():
    import pytest
    pytest.importorskip("rapidocr_onnxruntime")
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from altyazi.ocr import OcrUnavailable, RapidOcr

    try:
        ocr = RapidOcr()
    except OcrUnavailable as exc:  # bu makinede eski sürüm
        pytest.skip(str(exc))
    img = Image.new("L", (900, 90), 190)  # açık gri zemin (beton), yazının konturu yok
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default(34)
    ImageDraw.Draw(img).text((20, 22), "you'd get some money on your card.", font=font, fill=245)
    gray = np.asarray(img.filter(ImageFilter.GaussianBlur(0.8)))
    text = reader.read_text(gray, ocr.recognize, Config(bright_threshold=200), raw=True).text
    from altyazi.textproc import similar
    assert similar(text, "you'd get some money on your card.", 0.85), text


def test_raw_engine_reports_confidence_and_text_center_in_input_coordinates():
    class Engine:
        raw_input = True
        last_confidence = 0.97
        last_span = (10.0, 110.0)  # kırpılmış görüntüde

        def recognize(self, image):
            return ["Get him a hybrid."]

    engine = Engine()
    gray = _gray_with_text()  # parlak yazı x=50..350
    result = reader.read_text(gray, engine.recognize, Config(), raw=True, engine=engine)
    assert result.confidence == 0.97
    left = 50 - 8  # bright_bbox kenar payı
    assert result.center_x == left + 60
