"""Telefon okuma modunun yan paneli: telefon bölgesinin yanında, ekrandaki her gönderi/haber/mesaj
için bir kart (ekran sırasıyla). Çeviri gelene kadar İngilizcesi gri ve "…" ile görünür, çeviri
gelince bir kez Türkçeyle değişir (taslak yok, yazı titremez).

Normal (tıklanabilir) bir penceredir; başlığından sürüklenir, yeri kaydedilir (çift tık: telefonun
yanına geri yerleşir). Menü gibi ekran yakalamada görünür; okuma bölgesine taşarsa o kısım OCR'dan
önce karartılır (app.own_rects). Güncellenirken oyunun odağını almaz: pencere bir kez açılır,
sonra gizleme/gösterme yalnızca konum ve saydamlıkla yapılır (Windows'ta etkinleştirmeyen
pencere biçimi, focus_force yok)."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable

from .geometry import Rect
from .menu import ACCENT, DARK_BG, FONT, GRAY
from .phone import HIDE_AFTER_S

CARD_BG = "#2b2b2b"
HEADER_BG = "#141414"
WHITE = "#ffffff"
PENDING = " …"
OFFSCREEN = "+-32000+-32000"


class PhonePanel:
    WIDTH = 400
    MIN_HEIGHT = 320
    GAP = 12  # telefon bölgesiyle panel arası

    def __init__(self, root: tk.Misc, cfg, on_scan: Callable[[], None] | None = None,
                 on_moved: Callable[[], None] | None = None):
        self.cfg = cfg
        self._on_scan = on_scan
        self._on_moved = on_moved
        self.win = tk.Toplevel(root)
        self.win.title("GTA Altyazı · Telefon")
        self.win.overrideredirect(True)
        self.win.configure(background=DARK_BG, highlightthickness=1, highlightbackground="#3d3d3d")
        self.font = tkfont.Font(family=FONT, size=self._size())
        self.small = tkfont.Font(family=FONT, size=9)
        self.region: Rect | None = None
        self.screen_rect: Rect | None = None  # okuma iş parçacıkları okur (own_rects)
        self.shown = False
        self.cards: dict[str, dict] = {}
        self._order: list[str] = []
        self.created = 0  # oluşturulan kart sayısı (testler: artımlı güncelleme)
        self._hide_job = None
        self._dismissed: frozenset[str] | None = None  # "×" ile kapatılan içerik
        self._drag: tuple[int, int] | None = None
        self._build()
        self.win.geometry(f"{self.WIDTH}x{self.MIN_HEIGHT}{OFFSCREEN}")
        self.win.update_idletasks()
        self._no_activate()
        self.win.wm_attributes("-topmost", True)
        self._set_alpha(0.0)

    # --- kurulum ----------------------------------------------------------------------------

    def _size(self) -> int:
        return max(10, min(32, int(getattr(self.cfg, "phone_font_size", 15))))

    def _build(self) -> None:
        head = tk.Frame(self.win, background=HEADER_BG)
        head.pack(fill="x")
        self.title_lbl = tk.Label(head, text="Telefon", background=HEADER_BG, foreground=WHITE,
                                  font=(FONT, 11, "bold"), cursor="fleur")
        self.title_lbl.pack(side="left", padx=(10, 6), pady=6)
        self.status_lbl = tk.Label(head, text="", background=HEADER_BG, foreground=GRAY, font=self.small)
        self.status_lbl.pack(side="left", pady=6)
        close = tk.Label(head, text="✕", background=HEADER_BG, foreground=GRAY, font=(FONT, 11), cursor="hand2")
        close.pack(side="right", padx=(4, 10))
        close.bind("<Button-1>", lambda _e: self.dismiss())
        if self._on_scan is not None:
            scan = tk.Label(head, text="Şimdi çevir", background=HEADER_BG, foreground=ACCENT, font=self.small,
                            cursor="hand2")
            scan.pack(side="right", padx=4)
            scan.bind("<Button-1>", lambda _e: self._on_scan())
        for w in (head, self.title_lbl, self.status_lbl):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
            w.bind("<ButtonRelease-1>", self._drag_end)
            w.bind("<Double-Button-1>", lambda _e: self.redock())

        body = tk.Frame(self.win, background=DARK_BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background=DARK_BG, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, background=DARK_BG)
        self._inner_item = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.bar.pack(side="right", fill="y", pady=8)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._inner_item, width=e.width - 8))
        self.empty_lbl = tk.Label(self.inner, text="Telefon ekranında yazı bekleniyor…", background=DARK_BG,
                                  foreground=GRAY, font=self.small)
        self.empty_lbl.pack(anchor="w", pady=4)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.win.bind_all(seq, self._wheel, add="+")

    def _no_activate(self) -> None:
        """Windows: tıklanınca/gösterilince oyunun odağını almayan pencere (WS_EX_NOACTIVATE)."""
        try:
            from . import win32

            win32.make_no_activate(int(self.win.wm_frame(), 16))
        except Exception:
            pass  # Windows dışı (testler) ya da eski sistem: normal pencere

    def _set_alpha(self, value: float) -> None:
        try:
            self.win.wm_attributes("-alpha", value)
        except tk.TclError:
            pass

    def _wheel(self, event) -> None:
        if not self.shown or not str(event.widget).startswith(str(self.win)):
            return
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-1 if (getattr(event, "delta", 0) > 0 or event.num == 4) else 1, "units")

    # --- konum ------------------------------------------------------------------------------

    def _height(self) -> int:
        h = self.region.height if self.region is not None else self.MIN_HEIGHT
        return max(self.MIN_HEIGHT, min(h, self.win.winfo_screenheight() - 40))

    def docked_geometry(self) -> str:
        """Telefon bölgesinin sağı (yer varsa), değilse solu; üst kenarlar hizalı."""
        h = self._height()
        if self.region is None:
            return f"{self.WIDTH}x{h}+40+40"
        r = self.region
        screen_w = self.win.winfo_screenwidth()
        x = r.right + self.GAP
        if x + self.WIDTH > screen_w:
            x = max(0, r.left - self.GAP - self.WIDTH)
        return f"{self.WIDTH}x{h}+{x}+{max(0, r.top)}"

    def _geometry(self) -> str:
        saved = str(getattr(self.cfg, "phone_panel_geometry", "") or "")
        if saved.startswith("+") or saved.startswith("-"):
            return f"{self.WIDTH}x{self._height()}{saved}"
        return self.docked_geometry()

    def _place(self) -> None:
        self.win.geometry(self._geometry())
        self.win.update_idletasks()
        self._update_rect()

    def _update_rect(self) -> None:
        if not self.shown:
            self.screen_rect = None
            return
        try:
            self.screen_rect = Rect(self.win.winfo_rootx(), self.win.winfo_rooty(),
                                    self.win.winfo_width(), self.win.winfo_height())
        except tk.TclError:
            self.screen_rect = None

    def _drag_start(self, e) -> None:
        self._drag = (e.x_root - self.win.winfo_rootx(), e.y_root - self.win.winfo_rooty())

    def _drag_move(self, e) -> None:
        if self._drag is not None:
            self.win.geometry(f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _drag_end(self, _e) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.cfg.phone_panel_geometry = f"+{self.win.winfo_rootx()}+{self.win.winfo_rooty()}"
        self._update_rect()
        if self._on_moved is not None:
            self._on_moved()

    def redock(self) -> None:
        """Kaydedilen yeri unut: telefon bölgesinin yanına geri yerleş."""
        self.cfg.phone_panel_geometry = ""
        if self.shown:
            self._place()
        if self._on_moved is not None:
            self._on_moved()

    def set_region(self, rect: Rect | None) -> None:
        if rect == self.region:
            return
        self.region = rect
        if rect is None:
            self.clear()
            self.hide()
        elif self.shown:
            self._place()

    # --- görünürlük -------------------------------------------------------------------------

    def show(self) -> None:
        """Gösterir; pencere etkinleştirilmez (deiconify/focus yok: konum + saydamlık)."""
        if not self.shown:
            self.shown = True
            self._place()
            self._set_alpha(1.0)  # her zaman üstte (topmost); lift/focus pencereyi etkinleştirebilirdi
        self._update_rect()

    def hide(self) -> None:
        self._cancel_hide()
        if self.shown:
            self.shown = False
            self._set_alpha(0.0)
            self.win.geometry(OFFSCREEN)
        self.screen_rect = None

    def dismiss(self) -> None:
        """"✕": bu içerik için gizle; ekranda farklı içerik görünce yeniden açılır."""
        self._dismissed = frozenset(self._order)
        self.hide()

    def undismiss(self) -> None:
        """"Şimdi çevir": kapatılan panel yeniden açılabilsin."""
        self._dismissed = None
        if self._order and self.region is not None:
            self.show()

    def _cancel_hide(self) -> None:
        if self._hide_job is not None:
            try:
                self.win.after_cancel(self._hide_job)
            except tk.TclError:
                pass
            self._hide_job = None

    def _hide_if_empty(self) -> None:
        self._hide_job = None
        if not self._order:
            self.hide()

    # --- içerik -----------------------------------------------------------------------------

    def _new_card(self, key: str) -> dict:
        frame = tk.Frame(self.inner, background=CARD_BG, padx=10, pady=8)
        label = tk.Label(frame, text="", background=CARD_BG, foreground=GRAY, font=self.font,
                         justify="left", anchor="w", wraplength=self.WIDTH - 60)
        label.pack(fill="x")
        self.created += 1
        return {"frame": frame, "label": label, "en": "", "tr": None}

    def _paint(self, card: dict) -> None:
        if card["tr"] is not None:
            card["label"].configure(text=card["tr"], foreground=WHITE)
        else:
            card["label"].configure(text=card["en"] + PENDING, foreground=GRAY)

    def show_blocks(self, rows: list[tuple[str, str, str | None]]) -> None:
        """Ekrandaki bloklar (anahtar, İngilizce, Türkçe ya da None) ekran sırasıyla. Var olan
        kartlar yeniden kullanılır; yalnızca yeni bloklar için kart oluşturulur."""
        order = [key for key, _en, _tr in rows]
        for key, en, tr in rows:
            card = self.cards.get(key)
            if card is None:
                card = self.cards[key] = self._new_card(key)
                card["en"], card["tr"] = en, tr
                self._paint(card)
            elif tr is not None and card["tr"] != tr:
                card["tr"] = tr
                self._paint(card)
        for key in [k for k in self.cards if k not in set(order)]:
            self.cards.pop(key)["frame"].destroy()
        if order != self._order:
            for card in self.cards.values():
                card["frame"].pack_forget()
            for key in order:
                self.cards[key]["frame"].pack(fill="x", pady=(0, 8))
            if order and not self._order:
                self.canvas.yview_moveto(0.0)
            self._order = order
        if order:
            self.empty_lbl.pack_forget()
        elif not self.empty_lbl.winfo_manager():
            self.empty_lbl.pack(anchor="w", pady=4)
        self._update_status()
        if order:
            self._cancel_hide()
            if self._dismissed is not None and frozenset(order) != self._dismissed:
                self._dismissed = None
            if self._dismissed is None and self.region is not None:
                self.show()
        elif self.shown and self._hide_job is None:
            self._hide_job = self.win.after(int(HIDE_AFTER_S * 1000), self._hide_if_empty)

    def set_translation(self, key: str, translated: str) -> None:
        card = self.cards.get(key)
        if card is not None and card["tr"] != translated:
            card["tr"] = translated
            self._paint(card)
            self._update_status()

    def clear(self) -> None:
        self.show_blocks([])

    def _update_status(self) -> None:
        pending = sum(1 for c in self.cards.values() if c["tr"] is None)
        text = f"{len(self.cards)} yazı" + (f" · {pending} çevriliyor" if pending else "") if self.cards else ""
        self.status_lbl.configure(text=text)

    def apply_style(self) -> None:
        self.font.configure(size=self._size())
