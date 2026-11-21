"""Yoksayılan ifadeler (textproc.ignore_phrases, reader.read_text)."""

import json

import numpy as np

from altyazi import config as config_mod
from altyazi import reader
from altyazi.config import Config
from altyazi.textproc import ignore_phrases


def test_exact_line_is_dropped_case_and_space_insensitive():
    assert ignore_phrases("go   to FRANKLIN'S house.", ["Go to Franklin's house"]) == ""
    assert ignore_phrases("What the hell did you do?", ["Go to Franklin's house"]) == "What the hell did you do?"


def test_phrase_inside_a_line_is_removed_and_rest_kept():
    phrases = ["Press * to *"]
    assert ignore_phrases("Get in. Press E to enter the car. We need to go now.", phrases) == \
        "Get in. We need to go now."
    assert ignore_phrases("You lost the cops. Go to Franklin's house.", ["go to franklin's house"]) == \
        "You lost the cops."


def test_implausible_remainder_drops_the_line():
    assert ignore_phrases("Press [E] to enter the vehicle.", ["Press * to *"]) == ""
    assert ignore_phrases("Hold X to sprint ok", ["Hold * to sprint"]) == ""  # "ok" altyazı değil


def test_wildcard_and_word_boundaries():
    assert ignore_phrases("Pressed Eagle feathers are nice.", ["press e"]) == "Pressed Eagle feathers are nice."
    # `*` cümle sonunu aşmaz: sonraki cümle (asıl altyazı) kalır.
    assert ignore_phrases("Press X to call. Hey, it's me.", ["Press * to *"]) == "Hey, it's me."
    # Boş ve yalnızca `*` olan ifadeler hiçbir şeyi silmez.
    assert ignore_phrases("Hey, it's me.", ["", "  ", "*", "**"]) == "Hey, it's me."


def _gray_with_text():
    gray = np.full((60, 400), 30, dtype=np.uint8)
    gray[20:40, 50:350:5] = 255
    return gray


def test_reader_applies_the_list_and_keeps_the_box_for_change_detection():
    class Engine:
        last_confidence, last_span, last_box = 0.99, (0.0, 300.0), (2, 3, 300, 30)

        def recognize(self, image):
            return ["Press [E] to enter the vehicle."]

    engine = Engine()
    cfg = Config(ignore_phrases=["press * to *"])
    result = reader.read_text(_gray_with_text(), engine.recognize, cfg, raw=True, engine=engine)
    assert result.text == "" and result.ignored == "Press [E] to enter the vehicle."
    # Kutusu kaldı: yazı ekranda durdukça yeniden okunmaz (her karede OCR değil).
    assert result.text_box is not None
    # Liste canlı: cfg değişince sonraki okuma etkilenir.
    cfg.ignore_phrases = []
    result = reader.read_text(_gray_with_text(), engine.recognize, cfg, raw=True, engine=engine)
    assert result.text == "Press [E] to enter the vehicle." and result.ignored == ""


def test_ignore_phrases_round_trip_through_config(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config(ignore_phrases=["Press * to *"])
    config_mod.save(cfg, path)
    assert json.loads(path.read_text(encoding="utf-8"))["ignore_phrases"] == ["Press * to *"]
    assert config_mod.load(path).ignore_phrases == ["Press * to *"]
    assert Config().ignore_phrases == [] and Config().ignore_phrases is not Config().ignore_phrases


def test_hand_edited_single_string_is_one_phrase(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 2, "ignore_phrases": "Press * to *"}), encoding="utf-8")
    assert config_mod.load(path).ignore_phrases == ["Press * to *"]


def test_turkish_screen_text_is_not_translated_but_english_with_names_is():
    from altyazi import textproc

    for line in ("Ipuclari kartta llerlemenizi takipedin Yeni ipucu Var Oyun Yardimi Kupalar Kullanim Kolayli",
                 "Hedef pence re: chiaki-ng Yuk: sister islemci %26, program %6 ekran kart %14"):
        assert textproc.looks_turkish(line), line
    for line in ("Geralt of Rivia. Witcher with tenure.", "Novigrad, Oxenfurt, Velen.", "Q Ara",
                 "Ve Boobie Ike said Var Var.", "Leaders, ladders, readers and traders.", "Gather Ingredients"):
        assert not textproc.looks_turkish(line), line


def test_button_prompts_and_taskbar_clock_are_not_subtitles():
    from altyazi import textproc

    for junk in ("SCROLL SEND BACK", "SELECT BACK", "EXIT", "SCROLL SEND 44 TUR 19.11.2026 02:30"):
        assert textproc.strip_ui_chrome(junk) == "", junk
    assert textproc.strip_ui_chrome("SCROLL SET DESTINATION TUR 19.11.2026 02:30") == "SCROLL SET DESTINATION"
    for line in ("Back.", "Back off, man.", "Open the door.", "Hold on, call me back"):
        assert textproc.strip_ui_chrome(line) == line, line


def test_trailing_button_prompts_are_cut_only_when_uppercase_hud_words():
    from altyazi import textproc

    assert textproc.strip_ui_chrome("PHOTO STUDIO SELECT BACK") == "PHOTO STUDIO"
    assert textproc.strip_ui_chrome("REMINDERS SELECT EXIT") == "REMINDERS"
    assert textproc.strip_ui_chrome("BACK", strict=True) == "" and textproc.strip_ui_chrome("BACK") == "BACK"
    assert textproc.strip_ui_chrome("4 TUR 02:50 19.11.2026") == ""
    for line in ("Press select to exit", "I'll call you back", "Send it back", "Meet me at 10:30 on 19.11.2026."):
        assert textproc.strip_ui_chrome(line) == line, line
