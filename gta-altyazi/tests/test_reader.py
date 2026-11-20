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


class _EmptyEngine:
    """Hiç yazı bulamayan motor (yazısız parlak sahne)."""

    def __init__(self):
        self.calls = 0
        self.last_confidence, self.last_span, self.last_box = 0.0, None, None

    def recognize(self, image):
        self.calls += 1
        return []


def _bright():
    import numpy as np
    gray = np.zeros((40, 200), np.uint8)
    gray[10:30, 20:180] = 250
    return gray


def test_empty_read_retries_only_when_allowed():
    from altyazi import reader
    from altyazi.config import Config

    eng = _EmptyEngine()
    reader.read_text(_bright(), eng.recognize, Config(), True, eng)
    assert eng.calls == 3  # ham + keskin + medyan
    eng.calls = 0
    reader.read_text(_bright(), eng.recognize, Config(), True, eng, retry_empty=False)
    assert eng.calls == 1


class _GarbageEngine:
    def __init__(self, text, conf):
        self.text, self.conf = text, conf
        self.last_span, self.last_box = (10.0, 90.0), (10, 2, 90, 20)

    def recognize(self, image):
        self.last_confidence = self.conf
        return [self.text]


def test_low_confidence_non_words_are_dropped_but_real_lines_kept():
    from altyazi import reader
    from altyazi.config import Config

    cfg = Config(ocr_retry_confidence=0.0)  # ek okuma yok: yalnızca süzgeç sınanır
    eng = _GarbageEngine("Selitiyphamaykaot.", 0.59)
    assert reader.read_text(_bright(), eng.recognize, cfg, True, eng).text == ""
    eng = _GarbageEngine("Geralt of Rivia. Witcher with tenure.", 0.62)
    assert reader.read_text(_bright(), eng.recognize, cfg, True, eng).text.startswith("Geralt")


def test_button_prompt_region_reads_to_nothing():
    from altyazi import reader
    from altyazi.config import Config

    eng = _GarbageEngine("SCROLL SEND BACK", 0.98)
    assert reader.read_text(_bright(), eng.recognize, Config(), True, eng).text == ""


def test_pipelines_share_one_ocr_engine_but_keep_their_own_results(monkeypatch):
    import threading
    import altyazi.ocr as ocr_mod

    built = []

    class FakeRapid:
        def __init__(self, gpu=True):
            built.append(gpu)
            self.name, self.on_gpu, self.gpu_note, self.gpu_requested = "sahte", True, "", gpu
            self.last_confidence, self.last_span, self.last_box = 0.0, None, None

        def recognize(self, gray):
            self.last_confidence = float(gray[0, 0]) / 100
            return ["x"]

        def boxes(self, gray):
            return []

    monkeypatch.setattr(ocr_mod, "RapidOcr", FakeRapid)
    ocr_mod._shared.clear()
    import numpy as np
    from altyazi.config import Config

    a, b = ocr_mod.create(Config()), ocr_mod.create(Config())
    assert built == [True]  # tek motor kuruldu
    a.recognize(np.full((10, 10), 90, np.uint8))
    b.recognize(np.full((10, 10), 30, np.uint8))
    assert (a.last_confidence, b.last_confidence) == (0.9, 0.3)  # birbirinin sonucunu ezmez
    assert ocr_mod.create(Config(), "cpu") is not None and built == [True, False]
    ocr_mod._shared.clear()
    assert threading  # içe aktarma kullanılsın


def test_gpu_ocr_crash_marker_forces_cpu_next_time(monkeypatch, tmp_path):
    import altyazi.ocr as ocr_mod
    from altyazi.config import Config

    built = []

    class FakeRapid:
        def __init__(self, gpu=True):
            built.append((gpu, ocr_mod.GPU_MARKER.exists()))
            self.name, self.on_gpu, self.gpu_note, self.gpu_requested = "sahte", gpu, "", gpu
            self.last_confidence, self.last_span, self.last_box = 0.0, None, None

        def recognize(self, gray):
            return []

    monkeypatch.setattr(ocr_mod, "RapidOcr", FakeRapid)
    monkeypatch.setattr(ocr_mod, "GPU_MARKER", tmp_path / ".marker")
    ocr_mod._shared.clear()
    ocr_mod.create(Config())
    assert built == [(True, True)] and not ocr_mod.GPU_MARKER.exists()  # kurulurken işaret vardı, başarıyla silindi

    ocr_mod._shared.clear()
    ocr_mod.GPU_MARKER.write_text("çöktü")  # önceki açılış kurulum sırasında çöktü
    handle = ocr_mod.create(Config())
    assert built[-1] == (False, True) and "önceki açılışta" in handle.gpu_note  # işlemcide başladı
    ocr_mod._shared.clear()


def test_controller_prompts_and_labels_are_dropped_but_dialogue_stays():
    from altyazi.textproc import drop_hud_lines, expand_slang

    assert drop_hud_lines(["Hold L2 or R2 and aim around until you locate a signal"]) == []
    assert drop_hud_lines(["THROW R2 TAKE BEER R1 DROP"]) == []
    assert drop_hud_lines(["STASH"]) == [] and drop_hud_lines(["HANDLING 8.4 RANGE 4.4"]) == []
    assert drop_hud_lines(["Cal Check it out!2lbs! DIVE R1"]) == ["Cal Check it out!2lbs!"]
    assert drop_hud_lines(['Kenton: Okay? CAUTION MINDYOUR STEP']) == ["Kenton: Okay?"]
    assert drop_hud_lines(["LOOT Locals: Police? I need you."]) == ["Locals: Police? I need you."]
    assert drop_hud_lines(["Jason: Okay.", "NO!", "Kenton: you're ready, JASON"]) == \
        ["Jason: Okay.", "NO!", "Kenton: you're ready, JASON"]
    assert expand_slang("I'mma tell you, lemme go 'cause I'm gonna") == \
        "I'm going to tell you, let me go because I'm going to"


def test_phone_focus_follows_the_dense_text_column():
    from altyazi.ocr import Line
    from altyazi.phone import focus_phone

    def line(t, x0, y0, x1, y1):
        return Line(t, (), x0, y0, x1, y1)

    lines = [line("post text goes right here", 700, 100, 900, 120), line("second phone line here ok", 705, 160, 905, 180),
             line("HUD thing", 100, 300, 250, 320)]
    kept, x = focus_phone(lines, 1000, 500)
    assert [ln.text for ln in kept] == [lines[0].text, lines[1].text] and x is not None
    # Dar bölge: dokunulmaz
    assert focus_phone(lines, 300, 500) == (lines, None)
    # Telefon ortaya taşındı: şerit yeni konuma geçer
    moved = [line("another post in the middle", 380, 100, 580, 120), line("and one more line here", 382, 160, 582, 180)]
    kept2, x2 = focus_phone(moved, 1000, 500, prev_x=x)
    assert [ln.text for ln in kept2] == [m.text for m in moved]
