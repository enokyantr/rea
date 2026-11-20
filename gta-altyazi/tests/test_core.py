import json
from dataclasses import asdict

import numpy as np

from altyazi import config as config_mod
from altyazi.config import Region
from altyazi.geometry import Rect, region_to_screen, screen_to_region
from altyazi.imageproc import bright_bbox, for_ocr, same_signature, signature, text_mask, to_gray
from altyazi.textproc import split_sentences, Clear, Show, Stabilizer, is_plausible, normalize
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


def _mask(frame, mode="bright"):
    return text_mask(to_gray(frame), mode, 190)


def test_signature_ignores_moving_background_but_sees_text_change():
    a = signature(_mask(_frame_with_text(1, True)))
    b = signature(_mask(_frame_with_text(2, True)))
    c = signature(_mask(_frame_with_text(3, False)))
    assert same_signature(a, b)
    assert not same_signature(a, c)
    assert not same_signature(None, a)


def test_ocr_input_is_dark_text_on_white():
    frame = _frame_with_text(1, True)
    out = for_ocr(to_gray(frame), _mask(frame), "bright")
    assert out[30, 50] == 0 and out[5, 5] == 255


def test_bright_bbox_crops_to_text_with_margin():
    assert bright_bbox(_mask(_frame_with_text(1, True)), margin=4) == (16, 44, 46, 349)
    assert bright_bbox(_mask(_frame_with_text(1, False))) is None


def _outlined_subtitle(background):
    """Oyun altyazısı gibi: beyaz yazı, siyah kontur."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (700, 80), background)
    font = ImageFont.load_default(32)
    ImageDraw.Draw(img).text((20, 20), "That joke works, dog.", font=font, fill=(255, 255, 255),
                             stroke_width=2, stroke_fill=(0, 0, 0))
    rgb = np.asarray(img)
    bgra = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    return bgra


def test_outline_mask_drops_bright_background_but_keeps_text():
    """Açık sahne (gökyüzü): yalnızca parlaklıkla ayırmak arka planı yazı sanıyordu (t→c, d→cl)."""
    sky = _outlined_subtitle((214, 226, 238))
    bright_only = _mask(sky, "bright")
    outline = _mask(sky, "outline")
    assert bright_only.mean() > 0.5  # eski yöntem: neredeyse bütün kare "yazı"
    assert 0.005 < outline.mean() < 0.15  # yeni: yalnızca harfler
    assert not outline[:, :10].any() and not outline[:5, :].any()  # yazıdan uzak gökyüzü maskede yok
    # Koyu sahnede de aynı harfleri bulur.
    dark = _mask(_outlined_subtitle((40, 55, 70)), "outline")
    assert abs(int(dark.sum()) - int(outline.sum())) < 0.25 * dark.sum()
    # Yazı kırpması gökyüzünü değil, yazıyı çerçeveler.
    top, bottom, left, right = bright_bbox(outline, margin=0)
    assert top > 5 and left > 10 and right < 690


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


def test_split_sentences_for_local_translation():
    assert split_sentences("You think this is a game, Jason? This is our life now.") == [
        "You think this is a game, Jason?", "This is our life now."]
    assert split_sentences("Get in the car!") == ["Get in the car!"]
    assert split_sentences('He said "Run!" Then he left.') == ['He said "Run!"', "Then he left."]
    assert split_sentences("Wait... what?") == ["Wait... what?"]  # küçük harf: tek cümle
    assert split_sentences("Mr. Smith is here. Go!") == ["Mr. Smith is here.", "Go!"]


def test_repicking_extra_region_keeps_display_mode():
    cfg = config_mod.Config()
    cfg.extra_regions = [config_mod.ExtraRegion("gorev", Region(0, 0, 1, 1), "adaptive")]
    cfg.set_extra("gorev", Region(0.1, 0.1, 0.5, 0.1))
    assert cfg.extra("gorev").display_mode == "adaptive"
    cfg.set_extra("gorev", None)
    assert cfg.extra("gorev") is None


def test_needs_uncensored_detects_profanity_not_lookalikes():
    from altyazi.textproc import needs_uncensored
    for line in ["What the fuck are you doing?", "Shut the fuck up, asshole!", "You stupid son of a bitch!",
                 "Holy shit, that was close.", "This motherfucker owes me money.", "Get your ass in the car, bro.",
                 "Damn it!", "Screw you, man.", "You dumbass.", "Fucking cops!"]:
        assert needs_uncensored(line), line
    for line in ["Get in the car, now!", "Cops are right behind us!", "We need to pass them on the left.",
                 "First class tickets.", "The grass is wet.", "Don't embarrass me.", "Take the next left, quick!",
                 "Ask her about the brass knuckles.", "Assassin's creed."]:
        assert not needs_uncensored(line), line


def test_plausible_rejects_ocr_garbage_keeps_real_lines():
    for line in ["Be cool, fool, we got the paperwork.", "Lose the Cops.", "Whatever, nigga, it ain't gonna make you go no faster.",
                 "Oh shit, the one time!", "I might just be", "Hmm.", "Get in the car, now!"]:
        assert is_plausible(normalize([line]), 3), line
    # Not: yayın sohbetindeki kullanıcı adları ("Zilowzz") kelimeye benzediği için elenmez.
    for line in ["ntpgmfl/ Wag", "3hjc.", "nfP42if1/ Wag"]:
        assert not is_plausible(normalize([line]), 3), line


def test_config_saves_only_changes_and_migrates_frozen_old_defaults(tmp_path):
    path = tmp_path / "config.json"
    cfg = config_mod.Config()
    cfg.deepl_auth_key = "k:fx"
    config_mod.save(cfg, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"config_version": config_mod.CONFIG_VERSION, "deepl_auth_key": "k:fx"}

    # Sürüm 1 dosyası: tüm ayarlar yazılmış, aralarında artık eskimiş varsayılanlar var.
    old = asdict(config_mod.Config())
    old.update(preprocess="none", capture_interval_ms=50, deepl_auth_key="k:fx", font_size=30)
    old.pop("config_version", None)
    path.write_text(json.dumps(old), encoding="utf-8")
    cfg = config_mod.load(path)
    assert cfg.preprocess == config_mod.Config().preprocess != "none"  # donmuş eski varsayılan yerine yeni
    assert cfg.capture_interval_ms == config_mod.Config().capture_interval_ms
    assert cfg.font_size == 30 and cfg.deepl_auth_key == "k:fx"  # kullanıcının seçimleri korunur
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"config_version": config_mod.CONFIG_VERSION, "deepl_auth_key": "k:fx", "font_size": 30}


def _replay(readings, stable_reads=2):
    """Kullanıcının gerçek oyun logundaki ardışık okumaları besler, gösterilen satırları döndürür."""
    s = Stabilizer(stable_reads=stable_reads, similarity=0.88, clear_after=0.7)
    shown = []
    for i, text in enumerate(readings):
        event = s.feed(text, i * 0.04)
        if isinstance(event, Show):
            shown.append(event.text)
    return shown


def test_real_log_partial_readings_do_not_become_new_lines():
    # 01:33:59-01:34:00: aynı altyazı ("I'll see you at work.") karelerde parça parça okundu;
    # eskiden her parça ayrı satır sayılıp çevriliyordu.
    shown = _replay(["see", "see", "II see yåu", "see y'u", "sep yåu", "I'll see", "see you, at",
                     "see you t work-I", "see you t work'", "0 I'll see you at work.", "I'll see you at work",
                     "I'll see you at work.", "I'll se", "I'll see you at work", "I'll seek ou", "I'll see you at work"])
    # Eskiden 9 ayrı satır çevriliyordu; şimdi aynı satırın kademeli olarak iyileşen okumaları
    # (en fazla birkaç güncelleme) ve sonunda tam satır.
    assert len(shown) <= 4, shown
    assert shown[-1] in ("0 I'll see you at work.", "I'll see you at work.", "I'll see you at work")


def test_real_log_better_reading_upgrades_shown_line():
    shown = _replay(["Go to", "Go to", "Go to Franklin's house.", "Go to Franklin's house.", "Go to"])
    assert shown[0] == "Go to" and shown[-1] == "Go to Franklin's house." and len(shown) == 2


def test_new_line_is_still_detected():
    shown = _replay(["Man, fuck you.", "Man, fuck you.", "I'll see you at work.", "I'll see you at work."])
    assert shown == ["Man, fuck you.", "I'll see you at work."]


import pytest  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    # Gerçek --debug logundan kontur okumaları
    ("donrt hate me", "don't hate me"),
    ("Irm not doing this", "I'm not doing this"),
    ("[ike my women", "like my women"),
    ("(ike my women", "like my women"),
    ("that's exect[y it", "that's exactly it"),
    ("Itrs fine, yourre late, we can't", "It's fine, you're late, we can't"),
    ("I wonrt go, I cAnrt", "I won't go, I cAn't"),
    ("me 'cause lfm beautiful", "me 'cause I'm beautiful"),
    ("exactly how I Ilke my women", "exactly how I like my women"),
    # Doğru okumalara ve gerçek parantezlere dokunulmaz
    ("I [laughs] am here", "I [laughs] am here"),
    ("We (the crew) go", "We (the crew) go"),
    ("Others hers ours", "Others hers ours"),
])
def test_normalize_fixes_outline_ocr_confusions(raw, expected):
    assert normalize([raw]) == expected


def test_same_video_frame_cannot_confirm_a_garbage_reading():
    # Gerçek log: "I'll see you at work" gösterilirken bir geçiş karesi "I'll se - ouewo" okundu ve
    # 8 ms sonraki okuma (aynı kare) onu "doğruladı"; çöp satır çevrildi.
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.7, min_hold=0.04)
    t = 10.0
    assert s.feed("I'll see you at work", t) is None
    assert s.feed("I'll see you at work", t + 0.008) is None  # aynı kare: henüz değil
    shown = s.feed("I'll see you at work", t + 0.05)  # sonraki kare de aynı: gerçek satır
    assert isinstance(shown, Show) and shown.first_seen == t
    assert s.feed("I'll se - ouewo", t + 0.1) is None
    assert s.feed("I'll se - ouewo", t + 0.108) is None  # aynı karenin tekrarı
    assert s.feed("I'll see you at work", t + 0.14) is None  # sonraki kare: satır değişmedi


def test_upgrade_of_shown_line_needs_confirmation_too():
    # Gerçek log: tek bir uzun ama bozuk okuma gösterilen satırı hemen "güncelliyordu".
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.7, min_hold=0.04)
    s.feed("Ah, nigga, don't hat", 0.0)
    assert isinstance(s.feed("Ah, nigga, don't hat", 0.05), Show)
    assert s.feed("Ah, nigga, don't hat c zuciil&l, niggz.", 0.1) is None
    assert s.feed("Ah, nigga. don't hate me 'cause I'm beautiful, nigga", 0.12) is None
    assert s.feed("Ah, nigga, don't hat", 0.14) is None  # sıradan okuma bekleyeni bozmaz
    up = s.feed("Ah, nigga. don't hate me 'cause I'm beautiful, nigga", 0.16)
    assert isinstance(up, Show) and up.text == "Ah, nigga. don't hate me 'cause I'm beautiful, nigga"


def test_street_slang_goes_to_llm_even_without_profanity():
    from altyazi.textproc import needs_llm

    # Gerçek loglardan: taslak model bunları bozuyordu ("Yall anlatıyor").
    for line in ("Yall telling me.", "Who you callin a nigga?", "Can a loc come up in your crib?",
                 "I ain't going nowhere", "'Cause this man right here.", "You trippin', dog."):
        assert needs_llm(line), line
    for line in ("Go to Franklin's house.", "Can you hook me up with a blimp?",
                 "I'll send one out to you right away.", "Begin the mission again.",
                 "That's a real man's car."):
        assert not needs_llm(line), line


@pytest.mark.parametrize("worse,better", [
    # Gerçek --ocr-compare kayıtlarından aynı satırın iki okuması
    ("I'fJ see you aevybrk.", "I'll see you at works"),
    ("Ilarj, fuck you.", "Man. fuck you."),
    ("No, mad for the peniss That's exactly how I Tike my women",
     "No, mad for the penis, That's exactly how I like my women"),
    ("Ah, nigga. dontt hate me Icause I'm beautiful, nibPF!u", "Ah, nigga, don't hate me 'cause I'm beautiful"),
    ("Go to", "Go to Franklin's house."),
])
def test_reading_score_prefers_real_english_words(worse, better):
    from altyazi.textproc import reading_score

    assert reading_score(better) > reading_score(worse)


@pytest.mark.parametrize("raw,expected", [
    # Gerçek RapidOCR okumaları (kullanıcının kayıtları)
    ("you'd get some bitches on your dlick.", "you'd get some bitches on your dick."),
    ("maybe Tanisha'll caill your dlog-ass", "maybe Tanisha'll call your dog-ass"),
    ("if she ever stop fucking with that brain surgeor", "if she ever stop fucking with that brain surgeon"),
    ("or lawyer she fuckinig with.", "or lawyer she fucking with."),
    ("Niggal....", "Nigga...."),
    ("Ah, nigga, don't hate me 'cause T'm beautiful, nigga.", "Ah, nigga, don't hate me 'cause I'm beautiful, nigga."),
    # Doğru kelimeler, isimler ve kesme işaretli biçimler değişmez
    ("Wassup, can a loc come up in your crib?", "Wassup, can a loc come up in your crib?"),
    ("I wouldn't go to Lamar and Tanisha's house.", "I wouldn't go to Lamar and Tanisha's house."),
    ("Meet Lester at the Vanilla Unicorn, Franklin.", "Meet Lester at the Vanilla Unicorn, Franklin."),
])
def test_normalize_spell_fixes_rapidocr_slips(raw, expected):
    assert normalize([raw]) == expected


@pytest.mark.parametrize("raw,expected", [
    # Gerçek RapidOCR okumaları: bazı karelerde boşluklar atlanıyor
    ("maybehe'snotaracist", "maybe he's not a racist"),
    ("Moneyisntanissue", "Money isn't an issue"),
    ("Thisisthebestpart", "This is the best part"),
    ("getataxrebate", "get a tax rebate"),
    ("This is thebest part", "This is the best part"),
    ("thats crazy", "that's crazy"),
    # İsimler ve tek harflik OCR hatası bölünmez
    ("Franklin and Lamar went to Vinewood.", "Franklin and Lamar went to Vinewood."),
    ("maybe Tanishall call you", "maybe Tanishall call you"),
    ("with that brain surgeor", "with that brain surgeon"),
])
def test_normalize_splits_glued_words(raw, expected):
    assert normalize([raw]) == expected


def test_confident_reading_is_shown_without_waiting_for_a_second_read():
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.7, min_hold=0.04)
    shown = s.feed("I think you are right, Lamar.", 5.0, confident=True)
    assert isinstance(shown, Show) and shown.text == "I think you are right, Lamar."
    assert s.feed("I think you are right, Lamar.", 5.2, confident=True) is None  # aynı satır tekrar gösterilmez
    assert s.feed("Get him a hybrid.", 6.0) is None  # emin olunmayan yeni satır doğrulama bekler
    assert isinstance(s.feed("Get him a hybrid.", 6.05), Show)


def test_fast_accept_needs_real_words():
    from altyazi.textproc import all_words_known

    assert not all_words_known("acist")  # satır başının yarım okuması
    assert all_words_known("Hey, Simeon")  # isim
    assert all_words_known("Wassup, can a loc come up in your crib?")  # argo listesi


def test_box_signature_ignores_background_motion_but_sees_a_new_line():
    from altyazi.imageproc import box_signature

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 120, (120, 600), dtype=np.uint8)  # koyu, hareketli sahne
    frame[60:80, 100:500:6] = 250  # altyazı harfleri
    box = (100, 60, 500, 80)
    sig = box_signature(frame, box, 190)
    moved = frame.copy()
    moved[:55] = rng.integers(0, 120, (55, 600), dtype=np.uint8)  # arka plan değişti
    moved[60:80] = np.where(moved[60:80] >= 190, moved[60:80], rng.integers(0, 120, (20, 600)))
    assert same_signature(sig, box_signature(moved, box, 190), 0.03)
    new_line = frame.copy()
    new_line[60:80, 100:500] = rng.integers(0, 120, (20, 400), dtype=np.uint8)
    new_line[62:78, 120:480:4] = 250  # başka bir satır
    assert not same_signature(sig, box_signature(new_line, box, 190), 0.03)
    gone = frame.copy()
    gone[60:80, 100:500] = 40
    assert not same_signature(sig, box_signature(gone, box, 190), 0.03)


def test_hotkeys_parse_normalize_and_merge():
    from altyazi import hotkeys as hk

    assert hk.parse("ctrl+alt+r") == (hk.MOD_CONTROL | hk.MOD_ALT, 0x52)
    assert hk.normalize("Alt+Ctrl+PageUp") == "ctrl+alt+pageup"
    assert hk.display("ctrl+alt+up") == "Ctrl+Alt+↑"
    with pytest.raises(hk.HotkeyError):
        hk.parse("shift+r")  # oyunda yazmayı bozardı
    with pytest.raises(hk.HotkeyError):
        hk.parse("ctrl+alt+nope")
    merged = hk.merged({"pause": "ctrl+alt+p", "menu": "bozuk+x", "eski-islem": "ctrl+alt+z"})
    assert merged["pause"] == "ctrl+alt+p" and merged["menu"] == hk.DEFAULT_HOTKEYS["menu"]
    assert "eski-islem" not in merged
    assert hk.conflicts({"a": "ctrl+alt+x", "b": "ctrl+alt+x", "c": ""}) == {"ctrl+alt+x": ["a", "b"]}
    assert hk.from_tk_event("Prior", 0x4 | 0x20000) == "ctrl+alt+pageup"
    assert hk.from_tk_event("Control_L", 0x4) is None


def test_new_phone_scan_hotkey_is_free_and_yields_to_user_assignments():
    from altyazi import hotkeys as hk

    assert hk.DEFAULT_HOTKEYS["phone-scan"] == "ctrl+alt+y"
    assert not hk.conflicts(hk.DEFAULT_HOTKEYS)
    # Eski sürümün kaydettiği atamalar (phone-scan yok): varsayılanı alır
    assert hk.merged({"pause": "ctrl+alt+p"})["phone-scan"] == "ctrl+alt+y"
    # Kullanıcı Ctrl+Alt+Y'yi başka işleme atamışsa yeni işlem atanmamış başlar, çakışma olmaz
    merged = hk.merged({"pause": "ctrl+alt+y"})
    assert merged["pause"] == "ctrl+alt+y" and merged["phone-scan"] == "" and not hk.conflicts(merged)
    assert hk.merged({"pause": "ctrl+alt+y", "phone-scan": "ctrl+alt+y"})["phone-scan"] == "ctrl+alt+y"


def test_phone_config_fields_round_trip(tmp_path):
    path = tmp_path / "config.json"
    cfg = config_mod.Config()
    assert cfg.phone_mode is False and cfg.phone_debounce_ms == 300
    cfg.phone_mode, cfg.phone_panel_geometry = True, "+1500+200"
    config_mod.save(cfg, path)
    loaded = config_mod.load(path)
    assert loaded.phone_mode is True and loaded.phone_panel_geometry == "+1500+200"


def test_config_keeps_hotkeys_and_region_switches(tmp_path):
    path = tmp_path / "config.json"
    cfg = config_mod.Config()
    cfg.hotkeys["pause"] = "ctrl+alt+p"
    cfg.set_extra("gorev", Region(0.1, 0.1, 0.2, 0.2))
    cfg.extra("gorev").enabled = False
    cfg.region_enabled = False
    config_mod.save(cfg, path)
    loaded = config_mod.load(path)
    assert loaded.hotkeys["pause"] == "ctrl+alt+p" and loaded.hotkeys["menu"] == "ctrl+alt+m"
    assert loaded.extra("gorev").enabled is False and loaded.region_enabled is False


def test_game_names_are_not_spell_corrected_and_count_as_known():
    from altyazi import textproc

    assert textproc._spell_fix("Lucia, get in the car.") == "Lucia, get in the car."
    assert textproc._spell_fix("Grassrivers is a swamp.") == "Grassrivers is a swamp."
    assert textproc.all_words_known("Jason, meet me in Port Gellhorn.")


def test_game_names_are_kept_untranslated_and_user_entries_win(tmp_path):
    from altyazi import glossary

    path = tmp_path / "sozluk.txt"
    path.write_text("Roxy = Roksi\n", encoding="utf-8")
    entries = glossary.load(path)
    assert entries["Leonida Keys"] == "Leonida Keys" and entries["Lucia Caminos"] == "Lucia Caminos"
    assert entries["Roxy"] == "Roksi"


def test_game_name_words_come_from_bundled_name_files():
    from altyazi import textproc

    words = textproc._game_words()
    assert {"lucia", "gellhorn", "grassrivers", "dre'quan"} <= words


def test_spanish_lines_are_detected_and_not_spell_corrected_into_english():
    from altyazi import textproc

    for line in ("Que pasa, mija?", "Come on, hermano, vamos!", "¿Donde esta el dinero, pendejo?"):
        assert textproc.has_spanish(line), line
    for line in ("I'm looking for a woman.", "Lucia, get in the car.", "No, son. Me too."):
        assert not textproc.has_spanish(line), line
    assert textproc._spell_fix("hola pendejo, callate") == "hola pendejo, callate"
    assert textproc.normalize(["Oye, mija, ven aqui."]) == "Oye, mija, ven aqui."
    assert textproc.all_words_known("Gracias, hermano.")
    # İspanyolca -lar/-ler fiilleri Türkçe ek sanılıp satır atlanmaz.
    assert not textproc.looks_turkish("Tienes que hablar con ellos para trabajar.")


def test_draft_hold_follows_expected_final_latency_and_the_cap():
    from altyazi.hybrid import draft_hold_s

    assert draft_hold_s("Yeah.", 1000) < 0.6                  # kısa satır: kısa bekleme
    assert draft_hold_s("x" * 300, 1000) == 1.0               # uzun satır: üst sınır
    assert draft_hold_s("Hello there, friend.", 0) == 0.0     # 0: taslak hemen
    assert draft_hold_s("Hello there, friend.", 400) <= 0.4


def test_sms_notification_region_is_registered():
    from altyazi import hotkeys
    from altyazi.config import EXTRA_REGION_KEYS, EXTRA_REGION_LABELS, PHONE_LIKE

    assert "bildirim" in EXTRA_REGION_KEYS.values() and "bildirim" in PHONE_LIKE
    assert EXTRA_REGION_LABELS["bildirim"] == "Mesaj bildirimi"
    assert hotkeys.DEFAULT_HOTKEYS["pick-bildirim"] == "ctrl+alt+b"
    assert not hotkeys.conflicts(hotkeys.merged(None))


def test_notification_and_task_regions_become_transparent_text_once():
    from altyazi.config import CONFIG_VERSION, _from_dict, default_display_mode

    old = {"config_version": 2, "extra_regions": [
        {"name": "bildirim", "region": {"x": 0.6, "y": 0.7, "w": 0.3, "h": 0.2}, "display_mode": "cover"},
        {"name": "gorev", "region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.1}, "display_mode": "cover"},
        {"name": "telefon", "region": {"x": 0.6, "y": 0.1, "w": 0.3, "h": 0.8}, "display_mode": "cover"}]}
    modes = {e.name: e.display_mode for e in _from_dict(old).extra_regions}
    assert modes == {"bildirim": "adaptive", "gorev": "adaptive", "telefon": "cover"}
    assert default_display_mode("gorev") == "adaptive" and CONFIG_VERSION == 4
    # Sürüm 3: bildirim için kullanıcının sonradan seçtiği "cover" korunur, görev bir kez geçer
    modes = {e.name: e.display_mode for e in _from_dict({**old, "config_version": 3}).extra_regions}
    assert modes["bildirim"] == "cover" and modes["gorev"] == "adaptive"
    # Yeni sürümde kullanıcının seçimi korunur
    modes = {e.name: e.display_mode for e in _from_dict({**old, "config_version": 4}).extra_regions}
    assert modes == {"bildirim": "cover", "gorev": "cover", "telefon": "cover"}


def test_word_by_word_revealed_line_is_shown_again_when_complete():
    from altyazi.textproc import grows

    assert grows("Jason: You mean Shirley? With the dog?", "Jason: You mean Shirley? Witl")
    assert not grows("Jason: You mean Shirley? With", "Jason: You mean Shirley? Witl")  # düzeltme, uzama değil
    assert not grows("Lucia: Hey there", "Jason: Hey")
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.7, min_hold=0.04)
    events = [s.feed(t, i * 0.15, True) for i, t in enumerate([
        "Jason: You mean Shirley? Witl", "Jason: You mean Shirley? With", "Jason: You mean Shirley? With the",
        "Jason: You mean Shirley? With the dog?", "Jason: You mean Shirley? With the dog?",
        "Jason: You mean Shirley? With the dog?"])]
    shown = [e.text for e in events if isinstance(e, Show)]
    assert shown[-1] == "Jason: You mean Shirley? With the dog?" and len(shown) <= 2
    # Noktalamasız (yarım olabilecek) satır tek okumayla hızlı kabul edilmez
    s = Stabilizer(stable_reads=2, similarity=0.88, clear_after=0.7)
    assert s.feed("Lucia: Apparently, her daughter n", 0.0, True) is None
    assert isinstance(s.feed("Lucia: Okay.", 1.0, True), Show)


def test_task_region_widens_to_the_right_up_to_a_limit():
    from altyazi.config import MAX_TASK_WIDTH, Region, widened

    r = widened(Region(0.02, 0.3, 0.2, 0.2))
    assert r == Region(0.02, 0.3, 0.2 * 1.35, 0.2)
    assert widened(Region(0.02, 0.3, MAX_TASK_WIDTH, 0.2)) is None
    assert widened(Region(0.9, 0.3, 0.1, 0.2)) is None  # pencerenin sağ kenarında
