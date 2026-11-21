"""Telefon okuma modu: blok ayırma, arayüz süzme, korunan parçalar, çeviri sırası ve okuma döngüsü."""

import queue

import numpy as np
import pytest

from altyazi import textproc
from altyazi.config import Config
from altyazi.geometry import Rect
from altyazi.hybrid import FINAL, HybridTranslator
from altyazi.ocr import Box, group_lines, merge_lines, to_boxes
from altyazi.phone import (Block, KnownBlocks, PhonePipeline, PhoneQueue, ScanGate, content_signature,
                           segment, segment_lines)
from altyazi.translate import CachedTranslator, DryRunBackend


def line(text, x0, y0, h=18, w=None, score=0.98):
    """Tek kutuluk satır (genişlik metin uzunluğundan)."""
    return Box(text, score, x0, y0, x0 + (w or 9 * len(text)), y0 + h)


def texts(blocks):
    return [b.text for b in blocks]


# --- blok ayırma -----------------------------------------------------------------------------------


def test_social_posts_keep_header_with_post_and_drop_stats_and_status_bar():
    boxes = [
        # durum çubuğu: saat, şebeke, pil
        Box("12:45", 0.99, 10, 2, 50, 16), Box("5G", 0.95, 250, 2, 270, 16), Box("87%", 0.97, 280, 2, 310, 16),
        # gönderi 1: başlık (ad + kullanıcı adı + süre) iki kutuda, metin iki satır, beğeni satırı
        Box("Lenny Avery", 0.97, 60, 40, 160, 58), Box("@lennyavery · 3h", 0.96, 166, 40, 300, 58),
        line("Cops shut down the whole pier again", 60, 62),
        line("smh #LosSantos #ViceCity", 60, 84),
        Box("♡ 2.1K", 0.9, 60, 108, 110, 124), Box("↻ 340", 0.9, 140, 108, 190, 124),
        # gönderi 2: başlık boşlukla ayrılmış (yalnızca kullanıcı adı) -> gönderisine eklenir
        line("@vinewood_babe", 60, 170),
        line("who else saw the blimp crash lmaooo", 60, 200),
        line("www.lifeinvader.com/p/88213", 60, 222),
        line("Reply", 60, 260), line("2m", 140, 260, w=20),
    ]
    blocks = segment(boxes)
    assert texts(blocks) == [
        "Lenny Avery @lennyavery · 3h Cops shut down the whole pier again smh #LosSantos #ViceCity",
        "@vinewood_babe who else saw the blimp crash lmaooo www.lifeinvader.com/p/88213",
    ]
    assert blocks[0].top < blocks[1].top


def test_news_headline_and_body_are_separate_blocks_and_hyphenation_is_fixed():
    boxes = [
        line("WEAZEL NEWS", 20, 10, h=14),
        line("Vice City mayor caught in", 20, 40, h=30), line("yacht party scandal", 20, 74, h=30),
        line("The mayor's office an-", 20, 116), line("nounced an investigation into the", 20, 138),
        line("Leonida Keys event on Tuesday.", 20, 160),
    ]
    blocks = segment(boxes)
    assert texts(blocks) == [
        "WEAZEL NEWS",
        "Vice City mayor caught in yacht party scandal",
        "The mayor's office announced an investigation into the Leonida Keys event on Tuesday.",
    ]


def test_chat_bubbles_split_by_side_and_timestamps_dropped():
    boxes = [
        line("Lucia", 120, 5, h=16),  # sohbet başlığı (kişi adı)
        line("where u at?", 20, 40), line("12:41", 20, 62, h=12, w=30),
        line("on my way, five", 170, 90), line("minutes tops", 170, 112), line("12:42", 250, 134, h=12, w=30),
        line("hurry up the deal", 20, 160), line("starts at 9", 20, 182),
        line("Delivered", 230, 210, h=12),
    ]
    blocks = segment(boxes)
    assert texts(blocks) == ["Lucia", "where u at?", "on my way, five minutes tops", "hurry up the deal starts at 9"]


def test_chrome_turkish_ui_and_ignored_phrases_are_dropped():
    boxes = [
        line("3h", 10, 10, w=20), line("2.1K", 10, 40, w=40), line("ok", 10, 70, w=20),
        line("Bildirimler ve ayarlar için kaydırın", 10, 110),
        line("Press X to open the browser", 10, 150),
        line("Free bail bonds, call now", 10, 190),
    ]
    assert texts(segment(boxes, ignore=["Press * to *"])) == ["Free bail bonds, call now"]


def test_duplicate_blocks_get_distinct_keys():
    boxes = [line("lol same", 10, 10), line("lol same", 10, 60)]
    keys = [b.key for b in segment(boxes)]
    assert len(set(keys)) == 2


def test_windows_ocr_fallback_treats_each_line_as_block():
    blocks = segment_lines(["12:45", "@lenny cops everywhere lol", "Reply · 3h", "Breaking: pier closed"])
    assert texts(blocks) == ["@lenny cops everywhere lol", "Breaking: pier closed"]


def test_raw_rapidocr_boxes_feed_both_subtitle_lines_and_phone_blocks():
    raw = [([[60, 40], [160, 40], [160, 58], [60, 58]], "Lenny Avery", 0.97),
           ([[166, 41], [300, 41], [300, 57], [166, 57]], "@lennyavery", 0.96),
           ([[60, 62], [360, 62], [360, 80], [60, 80]], "pier is closed again", 0.99)]
    assert group_lines(raw) == ["Lenny Avery @lennyavery", "pier is closed again"]
    lines = merge_lines(to_boxes(raw))
    assert lines[0].x0 == 60 and lines[0].y1 == 58 and lines[0].score == pytest.approx(0.96)
    assert texts(segment(to_boxes(raw))) == ["Lenny Avery @lennyavery pier is closed again"]


# --- metin temizliği ---------------------------------------------------------------------------------


def test_normalize_post_keeps_handles_hashtags_urls_numbers_and_slang():
    text = textproc.normalize_post(["@lennyavery Moneyisntanissue when ur #LosSantos rich lmaooo 2.1K",
                                    "see www.lifeinvader.com/post tbh some-", "thing"])
    assert text == ("@lennyavery Money isn't an issue when ur #LosSantos rich lmaooo 2.1K "
                    "see www.lifeinvader.com/post tbh something")
    # Altyazı temizliği aynı metni bozar (karşılaştırma: bu yüzden ayrı yol)
    assert "@lennyavery" not in textproc.normalize(["@lennyavery Moneyisntanissue"])


@pytest.mark.parametrize("text,chrome", [
    ("12:45", True), ("3h", True), ("2.1K", True), ("♡ 2.1K ↻ 340", True), ("Reply · Share", True),
    ("10 min ago", True), ("87%", True), ("Yesterday 9:41 PM", True),
    ("Cops everywhere lol", False), ("@lenny", False), ("Breaking: 3 dead", False),
])
def test_ui_chrome_detection(text, chrome):
    assert textproc.is_ui_chrome(text) is chrome


def test_known_blocks_bind_ocr_jitter_to_the_same_key():
    known = KnownBlocks()
    first = known.canonical([Block("a", "Cops shut down the whole pier again 2.1K", 0, 1)])
    again = known.canonical([Block("b", "Cops shut down the whole pier again 2.lK", 5, 6)])
    assert again[0].key == first[0].key and again[0].text == first[0].text
    other = known.canonical([Block("c", "Totally different post about the blimp", 0, 1)])
    assert other[0].key == "c"
    # Aynı taramadaki iki farklı blok aynı anahtara bağlanmaz
    both = known.canonical([Block("x", "Cops shut down the whole pier again 2.1K", 0, 1),
                            Block("y", "Cops shut down the whole pier again 2.1k!", 2, 3)])
    assert both[0].key != both[1].key


# --- çeviri sırası -------------------------------------------------------------------------------------


class FakeTranslator:
    final = object()

    def __init__(self):
        self.cache: dict[str, str] = {}
        self.jobs: list = []

    def cached(self, text, source="ocr", draft=True):
        assert source == "phone" and draft is False
        return self.cache.get(text)

    def submit(self, text, context, is_current, on_result, on_error, source="ocr", draft=True):
        assert source == "phone" and draft is False and context is None
        self.jobs.append((text, on_result, on_error))

    def finish(self, translated=None):
        text, on_result, _ = self.jobs.pop(0)
        self.cache[text] = translated or f"TR {text}"
        on_result(FINAL, self.cache[text], 0.1)


def blocks_of(*names):
    return [Block(n, n, i, i + 1) for i, n in enumerate(names)]


def test_queue_translates_one_block_at_a_time_top_to_bottom():
    tr = FakeTranslator()
    done = []
    q = PhoneQueue(tr, lambda key, text, t: done.append((key, t)))
    rows = q.update(blocks_of("A", "B", "C"))
    assert [hit for _, hit in rows] == [None, None, None]
    assert [j[0] for j in tr.jobs] == ["A"]  # aynı anda yalnızca biri
    tr.finish()
    assert [j[0] for j in tr.jobs] == ["B"]
    tr.finish()
    tr.finish()
    assert done == [("A", "TR A"), ("B", "TR B"), ("C", "TR C")] and not tr.jobs


def test_queue_cancels_blocks_that_scrolled_out_and_reuses_cache_when_scrolling_back():
    tr = FakeTranslator()
    done = []
    q = PhoneQueue(tr, lambda key, text, t: done.append(key))
    q.update(blocks_of("A", "B", "C"))
    q.update(blocks_of("C", "D"))  # kaydırıldı: B sırası iptal; A sürüyor (bitince önbelleğe)
    assert q.pending() == ["A", "C", "D"]
    tr.finish()  # A
    assert [j[0] for j in tr.jobs] == ["C"]
    tr.finish()
    tr.finish()
    assert done == ["A", "C", "D"]
    rows = q.update(blocks_of("A", "C"))  # geri kaydırma: hepsi önbellekte, istek yok
    assert [hit for _, hit in rows] == ["TR A", "TR C"] and not tr.jobs


def test_queue_failed_block_is_not_retried_immediately():
    tr = FakeTranslator()
    errors = []
    q = PhoneQueue(tr, lambda *a: None, lambda key, msg: errors.append(key))
    q.update(blocks_of("A", "B"))
    tr.jobs.pop(0)[2](RuntimeError("Ollama yok"), FINAL, False)
    assert errors == ["A"] and [j[0] for j in tr.jobs] == ["B"]
    tr.finish()
    q.update(blocks_of("A", "B"))
    assert not tr.jobs  # A bir süre beklenir, B önbellekte


class RecordingBackend:
    """Son çevirmen: istem kaynağını kaydeder (CachedTranslator cache_source ile geçirir)."""

    def __init__(self):
        self.calls = []

    def cache_source(self, source):
        return "" if source == "ocr" else source

    def translate(self, text, context, source="ocr"):
        self.calls.append((text, context, source))
        return f"çeviri: {text}"


class ExplodingDraft:
    def translate(self, text, context):
        raise AssertionError("telefon taslak kullanmamalı")


def test_hybrid_phone_uses_only_final_and_falls_back_to_draft_without_final():
    import threading

    backend = RecordingBackend()
    final = CachedTranslator(backend, None)
    hybrid = HybridTranslator(final, ExplodingDraft(), final_for=lambda text: False)
    got = threading.Event()
    results = []
    q = PhoneQueue(hybrid, lambda key, text, t: (results.append(t), got.set()))
    q.update(blocks_of("lol same"))
    assert got.wait(5)
    assert results == ["çeviri: lol same"] and backend.calls == [("lol same", None, "phone")]
    assert final.key("lol same", source="phone") != final.key("lol same")
    assert q.update(blocks_of("lol same"))[0][1] == "çeviri: lol same"  # önbellekten anında
    hybrid.shutdown()

    class Draft:
        def translate(self, text, context):
            return f"taslak: {text}"

    local = HybridTranslator(None, Draft())
    got.clear()
    stages = []
    local.submit("hey", None, lambda: True, lambda stage, t, took: (stages.append((stage, t)), got.set()),
                 lambda *a: None, source="phone", draft=False)
    assert got.wait(5) and stages == [(FINAL, "taslak: hey")]
    assert local.cached("hey", source="phone", draft=False) == "taslak: hey"
    local.shutdown()

    dry = HybridTranslator(CachedTranslator(DryRunBackend(), None), None)
    assert dry.cached("x", source="phone", draft=False) is None
    dry.shutdown()


# --- okuma döngüsü ---------------------------------------------------------------------------------------


def frame(pattern: int) -> np.ndarray:
    """BGRA kare: koyu zeminde açık yazı şeridi; `pattern` şeridin yerini belirler (kaydırma)."""
    img = np.full((200, 120, 4), 20, dtype=np.uint8)
    top = 10 + 30 * pattern
    img[top:top + 12, 10:110, :3] = 230
    img[top + 4:top + 8, 20:100:6, :3] = 20  # harf boşlukları
    return img


class FakeOcr:
    name = "sahte"

    TEXTS = ["cops closed the pier again", "who saw that blimp crash lol", "new phone who dis",
             "breaking: mayor resigns", "free bail bonds today"]

    def __init__(self):
        self.calls = 0
        self.last_gray = None

    def boxes(self, gray):
        self.calls += 1
        self.last_gray = gray.copy()
        return [line(self.TEXTS[self.calls - 1], 5, 5)]


class Load:
    capture_interval_s = 0.033
    min_ocr_gap_s = 0.0


def make_pipe(frames, rect=Rect(100, 100, 120, 200), own=()):
    cfg = Config()
    cfg.phone_mode = True
    tr = FakeTranslator()
    ui = queue.Queue()
    ocr = FakeOcr()
    made = []
    current = {"rect": rect}
    grabs = []

    def grab(r):
        grabs.append(r)
        return frames[0]

    pipe = PhonePipeline(cfg, lambda: current["rect"], tr, ui, own_rects=lambda: list(own), log=lambda m: None,
                         make_ocr=lambda: made.append(1) or ocr, grab=grab)
    pipe.load = Load()
    return pipe, ocr, ui, current, grabs, made, tr


def drain(ui):
    out = []
    while not ui.empty():
        out.append(ui.get_nowait())
    return out


def test_scan_waits_for_change_and_stability_then_reads_once():
    frames = [frame(0)]
    pipe, ocr, ui, _, _, _, _ = make_pipe(frames)
    gate, state = ScanGate(0.3), {}
    for t in (0.0, 0.1, 0.2):
        pipe.step(None, gate, state, t)
    assert ocr.calls == 0  # henüz 300 ms durağan değil
    pipe.step(None, gate, state, 0.35)
    assert ocr.calls == 1
    for t in (0.5, 0.6, 1.5, 3.0):
        pipe.step(None, gate, state, t)
    assert ocr.calls == 1  # içerik aynı: OCR yok
    # kaydırma: her karede değişiyor -> okunmaz; durunca 300 ms sonra bir kez
    for i, t in enumerate((3.1, 3.2, 3.3, 3.4)):
        frames[0] = frame(1 + i)
        pipe.step(None, gate, state, t)
    assert ocr.calls == 1
    pipe.step(None, gate, state, 3.6)
    assert ocr.calls == 1
    pipe.step(None, gate, state, 3.75)
    assert ocr.calls == 2
    msgs = drain(ui)
    assert msgs[0] == ("phone-region", Rect(100, 100, 120, 200))
    phone = [m for m in msgs if m[0] == "phone"]
    assert len(phone) == 2 and phone[-1][1][0][1] == "who saw that blimp crash lol"
    assert pipe.last_scan[0] == 1 and pipe.last_scan[1] == 1


def test_force_scan_pause_load_gap_and_disabled_region():
    frames = [frame(0)]
    pipe, ocr, ui, current, grabs, made, _ = make_pipe(frames)
    gate, state = ScanGate(0.3), {}
    pipe.step(None, gate, state, 0.0)
    pipe.step(None, gate, state, 0.4)
    assert ocr.calls == 1
    pipe.scan_now()
    pipe.step(None, gate, state, 0.5)
    assert ocr.calls == 2  # değişim beklenmedi
    # yük: iki OCR arası en az 1 sn
    pipe.load.min_ocr_gap_s = 1.0
    frames[0] = frame(3)
    pipe.step(None, gate, state, 0.6)
    pipe.step(None, gate, state, 1.0)
    assert ocr.calls == 2
    pipe.step(None, gate, state, 1.55)
    assert ocr.calls == 3
    # duraklatılmışken yakalama da yok
    n = len(grabs)
    pipe.paused.set()
    frames[0] = frame(5)
    for t in (2.0, 3.0, 4.0):
        pipe.step(None, gate, state, t)
    assert len(grabs) == n and ocr.calls == 3
    pipe.paused.clear()
    # bölge kapalı / telefon modu kapalı: yakalama ve OCR yok
    current["rect"] = None
    for t in (5.0, 6.0):
        assert pipe.step(None, gate, state, t) >= 0.25
    assert len(grabs) == n and ("phone-region", None) in drain(ui)


def test_nothing_runs_until_phone_mode_is_active():
    import threading

    pipe, ocr, ui, current, grabs, made, _ = make_pipe([frame(0)], rect=None)
    pipe.start()
    threading.Event().wait(0.4)
    assert not made and not grabs  # OCR motoru bile kurulmadı
    pipe.stopped.set()
    pipe.join(2)


def test_own_panel_is_masked_before_ocr():
    frames = [frame(0)]
    own = [Rect(150, 100, 200, 300)]  # bölgenin sağ yarısını örten panel
    pipe, ocr, *_ = make_pipe(frames, own=own)
    gate, state = ScanGate(0.0), {}
    pipe.step(None, gate, state, 0.0)
    assert ocr.calls == 1
    assert not ocr.last_gray[:, 50:].any() and ocr.last_gray[:, :50].any()


def test_content_signature_sees_dark_text_on_light_theme():
    light = np.full((200, 120), 240, dtype=np.uint8)
    other = light.copy()
    light[20:32, 10:110] = 30
    other[80:92, 10:110] = 30
    from altyazi import imageproc

    assert not imageproc.same_signature(content_signature(light), content_signature(other))
    assert imageproc.same_signature(content_signature(light), content_signature(light.copy()))


# Gerçek RapidOCR (PP-OCRv4, işlemci) çıktısı: PIL ile çizilmiş sentetik telefon ekranları (Inter yazı
# tipi, 420x860). OCR boşlukları atlıyor, simgeleri harf okuyor ("♡ 2.1K ↻ 340" -> "2.1KC340").
NEWS_BOXES = [
    Box('12:45', 0.99, 18, 10, 63, 32), Box('5G87%', 1.00, 328, 10, 394, 31),
    Box('WEAZEL NEWS', 0.98, 20, 50, 191, 73),
    Box('ViceCitymayorcaughtin', 0.99, 21, 106, 345, 132), Box('yacht party scandal', 0.96, 18, 138, 275, 167),
    Box('2 hours ago', 1.00, 19, 174, 95, 194),
    Box("The mayor's office announced an investigation", 0.99, 19, 205, 397, 228),
    Box('into the Leonida Keys event on Tuesday, after', 0.95, 18, 228, 387, 252),
    Box('photosof thepartywentviralonline.', 0.99, 20, 255, 313, 275),
    Box('Gas prices hit record high', 0.99, 19, 311, 345, 340),
    Box('Drivers in Port Gellhorn are paying more than', 0.97, 18, 349, 384, 374),
    Box('ever at the pump.', 0.99, 18, 373, 164, 400),
]
SOCIAL_BOXES = [
    Box('12:45', 0.99, 16, 9, 65, 31), Box('5G 87%', 0.99, 327, 10, 395, 31),
    Box('Lifeinvader', 1.00, 18, 47, 143, 71),
    Box('Lenny Avery @lennyavery · 3h', 0.95, 70, 90, 319, 115),
    Box('Copsshutdown thewholepierAGAIN.', 0.98, 73, 120, 385, 140),
    Box('Third time thisweeksmh#ViceCity', 0.98, 70, 142, 356, 164),
    Box('2.1KC340', 0.96, 71, 171, 177, 191),
    Box('Brianna @bri_on_ocean·45m', 1.00, 70, 219, 310, 243),
    Box('whoelsesaw thatblimpcrashintothe', 0.98, 72, 250, 382, 270),
    Box("stadiumImaoooIcan'tbreathe", 0.97, 70, 273, 322, 293),
    Box('12KC3.4K', 0.96, 71, 302, 178, 322),
    Box('WeazelNews@WeazelNews·1h', 0.95, 72, 352, 337, 372),
    Box('BREAKING: Leonida Keys bridge closed', 0.98, 71, 377, 392, 401),
    Box('after armed robbery, suspects still at', 0.98, 70, 403, 372, 426),
    Box('large.www.weazel.com/live', 1.00, 70, 427, 300, 450),
    Box('640C1.2K', 0.99, 71, 455, 177, 475),
    Box('Dre @dre305·2m', 0.98, 70, 504, 220, 529),
    Box('ngl that new club on Ocean Driveis', 0.93, 69, 531, 360, 556),
    Box('overrated, $40 for a drink??', 0.99, 69, 555, 299, 579),
    Box('18C2', 0.94, 70, 584, 146, 604),
]


def test_real_rapidocr_news_screen_splits_headlines_from_bodies():
    assert texts(segment(NEWS_BOXES)) == [
        "WEAZEL NEWS",
        "Vice City mayor caught in yacht party scandal",
        "The mayor's office announced an investigation into the Leonida Keys event on Tuesday, after photos "
        "of the party went viral online.",
        "Gas prices hit record high",
        "Drivers in Port Gellhorn are paying more than ever at the pump.",
    ]


def test_real_rapidocr_social_screen_one_block_per_post_with_handles_intact():
    blocks = texts(segment(SOCIAL_BOXES))
    assert len(blocks) == 5 and blocks[0] == "Lifeinvader"  # durum çubuğu ve beğeni satırları atıldı
    assert blocks[1].startswith("Lenny Avery @lennyavery · 3h ") and blocks[1].endswith("#ViceCity")
    assert "the whole pier AGAIN" in blocks[1]
    assert blocks[2].startswith("Brianna @bri_on_ocean · 45m who else saw that blimp crash into the")
    assert blocks[3] == ("WeazelNews @WeazelNews · 1h BREAKING: Leonida Keys bridge closed after armed robbery, "
                         "suspects still at large. www.weazel.com/live")
    assert blocks[4].startswith("Dre @dre305 · 2m ngl that new club") and blocks[4].endswith("$40 for a drink??")
    assert not any(c in " ".join(blocks) for c in ("2.1KC340", "12KC3.4K", "5G", "18C2"))
