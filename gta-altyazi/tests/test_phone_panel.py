"""Telefon paneli (Tk): artımlı güncelleme, tek seferlik çeviri değişimi, yerleşim ve otomatik gizleme.
Ekran yoksa (DISPLAY) atlanır; Linux'ta Xvfb altında çalışır."""

import pytest

tk = pytest.importorskip("tkinter")

from altyazi.config import Config  # noqa: E402
from altyazi.geometry import Rect  # noqa: E402


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("ekran yok")
    r.withdraw()
    yield r
    r.destroy()


def make(root, **kw):
    from altyazi.phone_panel import PhonePanel

    cfg = Config()
    panel = PhonePanel(root, cfg, **kw)
    root.update()
    return cfg, panel


def test_cards_are_reused_and_replaced_once_when_translation_arrives(root):
    cfg, panel = make(root)
    panel.set_region(Rect(700, 100, 300, 500))
    panel.show_blocks([("a", "cops closed the pier", None), ("b", "who saw that lol", "kim gördü ahahah")])
    root.update()
    assert panel.shown and panel.screen_rect is not None
    a, b = panel.cards["a"]["label"], panel.cards["b"]["label"]
    assert a.cget("text") == "cops closed the pier …" and a.cget("foreground") != "#ffffff"
    assert b.cget("text") == "kim gördü ahahah" and b.cget("foreground") == "#ffffff"
    assert panel.created == 2
    # Aynı tarama tekrar gelir: yeni kart yok, metin değişmez
    panel.show_blocks([("a", "cops closed the pier", None), ("b", "who saw that lol", "kim gördü ahahah")])
    assert panel.created == 2 and panel.cards["a"]["label"] is a
    panel.set_translation("a", "polis iskeleyi yine kapattı")
    assert a.cget("text") == "polis iskeleyi yine kapattı" and a.cget("foreground") == "#ffffff"
    # Kaydırma: b gitti, c geldi, a kaldı -> yalnızca c için kart, sıra ekran sırası
    panel.show_blocks([("c", "new phone who dis", None), ("a", "cops closed the pier", "polis iskeleyi yine kapattı")])
    root.update()
    assert panel.created == 3 and "b" not in panel.cards and panel.cards["a"]["label"] is a
    assert panel._order == ["c", "a"]
    ys = [panel.cards[k]["frame"].winfo_y() for k in ("c", "a")]
    assert ys[0] < ys[1]
    assert "2 yazı" in panel.status_lbl.cget("text") and "1 çevriliyor" in panel.status_lbl.cget("text")


def test_docks_right_of_phone_or_left_when_no_space_and_remembers_moves(root):
    moved = []
    cfg, panel = make(root, on_moved=lambda: moved.append(1))
    sw = root.winfo_screenwidth()
    panel.set_region(Rect(100, 50, 300, 500))
    assert panel.docked_geometry().endswith(f"+{100 + 300 + panel.GAP}+50")
    panel.set_region(Rect(sw - 320, 50, 300, 500))
    assert panel.docked_geometry().endswith(f"+{sw - 320 - panel.GAP - panel.WIDTH}+50")
    cfg.phone_panel_geometry = "+30+40"
    panel.show_blocks([("a", "hello there", None)])
    root.update()
    assert panel.win.winfo_rootx() == 30 and panel.win.winfo_rooty() == 40
    panel.redock()
    root.update()
    assert cfg.phone_panel_geometry == "" and moved and panel.win.winfo_rootx() == sw - 320 - panel.GAP - panel.WIDTH


def test_auto_hides_when_empty_dismiss_and_font(root, monkeypatch):
    from altyazi import phone_panel

    monkeypatch.setattr(phone_panel, "HIDE_AFTER_S", 0.05)
    cfg, panel = make(root)
    panel.show_blocks([("a", "hello there", None)])
    assert not panel.shown  # bölge bilinmeden açılmaz
    panel.set_region(Rect(100, 50, 300, 500))
    panel.show_blocks([("a", "hello there", None)])
    assert panel.shown
    panel.show_blocks([])
    assert panel.shown  # hemen değil
    root.after(150, root.quit)
    root.mainloop()
    assert not panel.shown and panel.screen_rect is None
    # "✕": aynı içerik gelirse kapalı kalır, farklı içerik gelince açılır
    panel.show_blocks([("a", "hello there", None)])
    panel.dismiss()
    panel.show_blocks([("a", "hello there", None)])
    assert not panel.shown
    panel.show_blocks([("a", "hello there", None), ("z", "new post", None)])
    assert panel.shown
    cfg.phone_font_size = 22
    panel.apply_style()
    assert panel.font.cget("size") == 22
