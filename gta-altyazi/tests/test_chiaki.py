"""chiaki-ng / düşük kaliteli akış: pencere seçimi, siyah bantlar, emin olunmayan okumada süzgeçler."""

import numpy as np

from altyazi import reader
from altyazi.config import Config
from altyazi.geometry import Rect, fit_aspect, pick_window


def test_stream_window_beats_browser_tab_and_settings_window():
    windows = [("chiaki-ng - GitHub - Google Chrome", Rect(0, 0, 2560, 1400)),
               ("chiaki-ng", Rect(100, 100, 800, 600)),
               ("chiaki-ng | PS5 Stream", Rect(0, 0, 1920, 1080))]
    assert pick_window(["PS Remote Play", "Chiaki"], windows)[0] == "chiaki-ng | PS5 Stream"
    only_browser = windows[:1]
    assert pick_window(["Chiaki"], only_browser)[0].startswith("chiaki-ng - GitHub")
    assert pick_window(["Chiaki"], [("Not Defteri", Rect(0, 0, 10, 10))]) is None


def test_letterboxed_window_uses_video_area_only():
    wide = fit_aspect(Rect(0, 0, 2560, 1080), 16 / 9)   # yanlarda bant
    assert wide == Rect(320, 0, 1920, 1080)
    tall = fit_aspect(Rect(0, 0, 1920, 1200), 16 / 9)   # üstte/altta bant
    assert tall == Rect(0, 60, 1920, 1080)
    assert fit_aspect(Rect(5, 5, 1920, 1080), 16 / 9) == Rect(5, 5, 1920, 1080)
    assert fit_aspect(Rect(0, 0, 2560, 1080), 0) == Rect(0, 0, 2560, 1080)  # kapalı (Esnet modu)


class Engine:
    """Ham görüntüde emin değil ve yanlış; keskinleştirilmiş görüntüde doğru okur."""

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        sharp = self.calls == 2
        self.last_confidence = 0.97 if sharp else 0.62
        self.last_span, self.last_box = (10.0, 90.0), (10, 2, 90, 20)
        return ["You're not gonna rob me again?"] if sharp else ["Youre n0t gonna rbo rne agaln?"]


def _gray():
    gray = np.zeros((40, 200), np.uint8)
    gray[15:25, 20:180] = 250
    return gray


def test_low_confidence_read_retries_filtered_images_and_keeps_best():
    eng = Engine()
    result = reader.read_text(_gray(), eng.recognize, Config(), True, eng)
    assert result.text == "You're not gonna rob me again?" and result.variant == "keskin"
    assert result.confidence == 0.97 and eng.calls == 2  # keskin yeterince iyi: medyan denenmedi


def test_confident_read_is_not_retried():
    eng = Engine()
    eng.calls = 1  # ilk çağrı "emin" dalına düşsün
    result = reader.read_text(_gray(), eng.recognize, Config(), True, eng)
    assert eng.calls == 2 and result.variant == "ham"


def test_retry_threshold_follows_config():
    eng = Engine()
    eng.calls = 1  # ham okuma güveni 0.97
    cfg = Config(ocr_retry_confidence=0.98)  # "düşük kalite": 0.97 bile yeniden denenir
    reader.read_text(_gray(), eng.recognize, cfg, True, eng)
    assert eng.calls > 2
