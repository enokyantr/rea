"""Ayar menüsü (Ctrl+Alt+M): durum, görünüm, çeviri, OCR, kısayollar, dublaj ve kayıt.

Windows 11 görünümü (Sun Valley teması, sv-ttk; kurulu değilse standart tema). Menü yalnızca
`controller` üzerinden çalışır (app.run_gui sağlar). Anında uygulanan ayarlar (görünüm, kısayollar,
çeviri/dublaj anahtarları) hemen kaydedilir; motor değiştiren ayarlar "Kaydet ve yeniden başlat" ile."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Protocol

from . import hotkeys as hk

GREEN, RED, GRAY, ORANGE, ACCENT = "#3fb950", "#f85149", "#8b949e", "#d29922", "#58a6ff"
FONT = "Segoe UI"
DARK_BG = "#1c1c1c"  # Sun Valley koyu tema zemini


class Controller(Protocol):
    cfg: object

    def action(self, name: str) -> None: ...
    def save(self) -> None: ...
    def restart(self) -> None: ...
    def paused(self) -> bool: ...
    def region_rows(self) -> list[dict]: ...
    def engine_rows(self) -> list[tuple[str, str]]: ...
    def last_line(self) -> tuple[str, str, str]: ...
    def set_region_enabled(self, name: str, enabled: bool) -> None: ...
    def remove_region(self, name: str) -> None: ...
    def preset_region(self, name: str) -> None: ...
    def apply_style(self) -> None: ...
    def hotkey_rows(self) -> list[tuple[str, str, str, str]]: ...
    def set_hotkey(self, action: str, combo: str) -> str | None: ...
    def free_hotkeys(self) -> list[str]: ...
    def tts_voices(self) -> list[str]: ...
    def set_tts(self, enabled: bool) -> str | None: ...
    def dub_status(self) -> list[tuple[str, str]]: ...
    def characters(self) -> list: ...
    def character_gender(self, char_id: int, gender: str) -> None: ...
    def character_remove(self, char_id: int) -> None: ...
    def characters_clear(self) -> None: ...
    def character_sample(self, char_id: int) -> str | None: ...
    def open_folder(self, which: str) -> None: ...
    def desktop_shortcut(self) -> str: ...


def _apply_theme(root: tk.Misc) -> bool:
    try:
        import sv_ttk
    except ImportError:
        return False
    try:
        sv_ttk.set_theme("dark", root)
    except TypeError:  # eski sv-ttk: kök parametresi yok
        sv_ttk.set_theme("dark")
    style = ttk.Style(root)
    # Tema pencere zeminini ve etiket arka planını boş bırakıyor (sistemin açık grisi görünüyordu).
    style.configure(".", background=DARK_BG)
    style.configure("TLabel", background=DARK_BG, foreground="#e6e6e6")
    style.configure("Title.TLabel", font=(FONT, 16, "bold"))
    style.configure("Section.TLabel", font=(FONT, 11, "bold"))
    style.configure("Muted.TLabel", foreground=GRAY)
    style.configure("Big.Switch.TCheckbutton", font=(FONT, 11, "bold"))
    return True


class ControlPanel:
    REFRESH_MS = 500

    def __init__(self, root: tk.Misc, ctl: Controller):
        self.ctl = ctl
        self.cfg = ctl.cfg
        self.themed = _apply_theme(root)
        self.win = tk.Toplevel(root)
        self.win.title("GTA Altyazı")
        self.win.geometry("940x640")
        self.win.minsize(760, 520)
        self.win.protocol("WM_DELETE_WINDOW", self.hide)
        if self.themed:
            self.win.configure(background=DARK_BG)
        self._capture_action: str | None = None
        self._switch = "Switch.TCheckbutton" if self.themed else "TCheckbutton"
        self._big_switch = "Big.Switch.TCheckbutton" if self.themed else "TCheckbutton"
        self._accent = "Accent.TButton" if self.themed else "TButton"
        self._build()
        self.win.bind("<KeyPress>", self._on_key, add="+")
        self._refresh()

    # --- görünürlük -------------------------------------------------------------------------

    @property
    def visible(self) -> bool:
        return self.win.state() != "withdrawn"

    def show(self) -> None:
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def hide(self) -> None:
        self._capture_action = None
        self.win.withdraw()

    def toggle(self) -> None:
        self.hide() if self.visible else self.show()

    # --- kurulum ----------------------------------------------------------------------------

    def _build(self) -> None:
        outer = ttk.Frame(self.win, padding=(16, 12, 16, 8))
        outer.pack(fill="both", expand=True)

        head = ttk.Frame(outer)
        head.pack(fill="x")
        ttk.Label(head, text="GTA Altyazı", style="Title.TLabel").pack(side="left")
        self.translate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(head, text="Çeviri", style=self._big_switch, variable=self.translate_var,
                        command=self._toggle_translation).pack(side="left", padx=(28, 8))
        self.dub_var = tk.BooleanVar(value=bool(self.cfg.tts_enabled))
        ttk.Checkbutton(head, text="Dublaj", style=self._big_switch, variable=self.dub_var,
                        command=self._toggle_dub).pack(side="left", padx=8)
        ttk.Button(head, text="Çık", command=lambda: self.ctl.action("quit")).pack(side="right")
        ttk.Button(head, text="Gizle", command=self.hide).pack(side="right", padx=6)
        ttk.Button(head, text="Kaydet ve yeniden başlat", style=self._accent,
                   command=self.ctl.restart).pack(side="right")
        self.state_lbl = ttk.Label(outer, text="", style="Muted.TLabel")
        self.state_lbl.pack(fill="x", pady=(4, 8))

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)
        for title, build in (("Durum", self._tab_status), ("Görünüm", self._tab_look),
                             ("Çeviri", self._tab_translation), ("OCR ve hız", self._tab_ocr),
                             ("Dublaj", self._tab_dub), ("Kısayollar", self._tab_hotkeys),
                             ("Kayıt", self._tab_log)):
            frame = ttk.Frame(nb, padding=14)
            nb.add(frame, text=f"  {title}  ")
            build(frame)

        self.note_lbl = ttk.Label(outer, text="", foreground=ORANGE)
        self.note_lbl.pack(fill="x", pady=(6, 0))

    def _toggle_translation(self) -> None:
        if self.translate_var.get() == self.ctl.paused():
            self.ctl.action("pause")

    def _toggle_dub(self) -> None:
        error = self.ctl.set_tts(self.dub_var.get())
        if error:
            self.dub_var.set(False)
            self.note_lbl.configure(text=error, foreground=RED)
        else:
            self.note_lbl.configure(text="")

    @staticmethod
    def _card(parent, title: str) -> ttk.Frame:
        box = ttk.LabelFrame(parent, text=f" {title} ", padding=(12, 8))
        box.pack(fill="x", pady=(0, 10))
        return box

    # Durum ------------------------------------------------------------------------------------

    def _tab_status(self, f: ttk.Frame) -> None:
        card = self._card(f, "Okuma bölgeleri")
        self.region_widgets: dict[str, dict] = {}
        for i, row in enumerate(self.ctl.region_rows()):
            name = row["name"]
            ttk.Label(card, text=row["label"], width=9, style="Section.TLabel").grid(row=i, column=0, sticky="w",
                                                                                   pady=4)
            var = tk.BooleanVar(value=row["enabled"])
            ttk.Checkbutton(card, style=self._switch, variable=var,
                            command=lambda n=name, v=var: self.ctl.set_region_enabled(n, v.get())).grid(
                row=i, column=1, padx=(0, 8))
            status = ttk.Label(card, text="", width=22)
            status.grid(row=i, column=2, sticky="w")
            ttk.Button(card, text="Bölge seç", command=lambda n=name: self.ctl.action(
                "pick" if n == "altyazi" else f"pick-{n}")).grid(row=i, column=3, padx=3)
            ttk.Button(card, text="Hazır konum", command=lambda n=name: self.ctl.preset_region(n)).grid(
                row=i, column=4, padx=3)
            if name != "altyazi":
                ttk.Button(card, text="Kaldır", command=lambda n=name: self.ctl.remove_region(n)).grid(
                    row=i, column=5, padx=3)
            self.region_widgets[name] = {"status": status, "var": var}
        ttk.Label(card, text="Hazır konumlar GTA V düzenine göre tahminidir; GTA 6'da \"Bölge seç\" ile fareyle "
                             "seçin.", style="Muted.TLabel").grid(row=9, column=0, columnspan=6, sticky="w",
                                                                  pady=(6, 0))

        card = self._card(f, "Son satır")
        self.last_lbl = ttk.Label(card, text="", justify="left", wraplength=840, font=(FONT, 11))
        self.last_lbl.pack(anchor="w")
        self.last_info = ttk.Label(card, text="", style="Muted.TLabel")
        self.last_info.pack(anchor="w")

        card = self._card(f, "Motorlar")
        self.engine_lbl = ttk.Label(card, text="", justify="left")
        self.engine_lbl.pack(anchor="w")
        row = ttk.Frame(card)
        row.pack(anchor="w", pady=(8, 0))
        self.shortcut_lbl = ttk.Label(row, text="", style="Muted.TLabel")
        ttk.Button(row, text="Masaüstüne kısayol ekle",
                   command=lambda: self.shortcut_lbl.configure(text=self.ctl.desktop_shortcut())).pack(side="left")
        self.shortcut_lbl.pack(side="left", padx=10)

    # Görünüm ----------------------------------------------------------------------------------

    def _tab_look(self, f: ttk.Frame) -> None:
        card = self._card(f, "Türkçe yazı (anında uygulanır)")
        r = 0
        r = self._spin(card, r, "Yazı boyutu", "font_size", 12, 72, 1, live=True)
        r = self._spin(card, r, "Konum: İngilizcenin ne kadar üstünde (piksel)", "overlay_gap_px", -300, 600, 4,
                       live=True)
        r = self._spin(card, r, "Kontur kalınlığı", "outline_px", 0, 6, 1, live=True)
        families = sorted({"Segoe UI", "Arial", "Calibri", "Tahoma", "Verdana", "Bahnschrift", self.cfg.font_family})
        r = self._combo(card, r, "Yazı tipi", "font_family", families, live=True)
        r = self._check(card, r, "İngilizceyi de göster (küçük, Türkçenin üstünde)", "show_original", live=True)
        card = self._card(f, "Gösterim (yeniden başlatınca)")
        r = 0
        r = self._check(card, r, "Yayında (OBS) ve ekran görüntüsünde görünsün", "capture_visible", restart=True)
        r = self._combo(card, r, "Gösterim", "display_mode", ["adaptive", "cover"], restart=True,
                        hint="adaptive: İngilizcenin üstünde · cover: İngilizceyi örten kutu")
        r = self._spin(card, r, "Son satırlar paneli: satır sayısı", "history_size", 3, 30, 1)

    # Çeviri -----------------------------------------------------------------------------------

    def _tab_translation(self, f: ttk.Frame) -> None:
        card = self._card(f, "Son çevirmen (yeniden başlatınca)")
        r = 0
        r = self._combo(card, r, "Son çevirmen", "final_translator", ["ollama", "deepl"], restart=True,
                        hint="ollama: ücretsiz, sansürsüz · deepl: anahtar gerekir")
        r = self._entry(card, r, "Yerel dil modeli (Ollama)", "ollama_model", restart=True)
        r = self._check(card, r, "Dil modeli yalnızca küfür/argo/sözlük satırlarında (önerilen)",
                        "ollama_only_profanity", restart=True)
        r = self._check(card, r, "Yarım cümlelerde önceki satırı bağlam olarak gönder",
                        "ollama_context_continuation", restart=True)
        r = self._check(card, r, "Sansürsüz çeviri (küfür yumuşatılmaz)", "uncensored", restart=True)
        r = self._entry(card, r, "DeepL anahtarı", "deepl_auth_key", restart=True, secret=True)
        ttk.Button(f, text="Sözlüğü aç (isim/terim karşılıkları)",
                   command=lambda: self.ctl.open_folder("glossary")).pack(anchor="w")

    # OCR --------------------------------------------------------------------------------------

    def _tab_ocr(self, f: ttk.Frame) -> None:
        card = self._card(f, "Okuma")
        r = 0
        r = self._combo(card, r, "OCR motoru", "ocr_engine", ["auto", "windows"], restart=True,
                        hint="auto: RapidOCR (önerilen)")
        r = self._combo(card, r, "OCR nerede çalışsın", "ocr_device", ["auto", "cpu"], restart=True,
                        hint="auto: ekran kartı · cpu: dil modeli yavaşlarsa")
        r = self._scale(card, r, "Hızlı kabul güveni (yüksek = az hata, biraz geç)", "fast_accept_confidence",
                        0.80, 1.00)
        r = self._check(card, r, "Yalnızca yazı değişince yeniden oku (önerilen)", "change_detection")
        r = self._spin(card, r, "Altyazı gidince Türkçe kaç ms sonra silinsin", "clear_after_ms", 200, 3000, 100,
                       restart=True)

    # Dublaj -----------------------------------------------------------------------------------

    def _tab_dub(self, f: ttk.Frame) -> None:
        card = self._card(f, "Dublaj")
        r = 0
        r = self._combo(card, r, "Ses", "dub_engine", ["edge", "windows"], dub=True,
                        hint="edge: doğal kadın/erkek ses (internet) · windows: çevrimdışı")
        r = self._check(card, r, "Konuşanı sesinden tanı (kadın/erkek, aynı karaktere aynı ses)",
                        "dub_detect_speaker", dub=True, switch=True)
        r = self._scale(card, r, "Konuşma hızı", "tts_rate", 0.6, 2.0)
        r = self._scale(card, r, "Ses düzeyi", "tts_volume", 0.0, 1.0)
        voices = self.ctl.tts_voices()
        r = self._combo(card, r, "Windows sesi", "tts_voice", [""] + voices, dub=True,
                        hint="boş: ilk Türkçe ses" if voices else "Türkçe Windows sesi yok (edge önerilir)")
        self.dub_status_lbl = ttk.Label(card, text="", justify="left", style="Muted.TLabel")
        self.dub_status_lbl.grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))

        card = self._card(f, "Karakter hafızası")
        cols = ("karakter", "cinsiyet", "perde", "satir")
        self.char_tree = ttk.Treeview(card, columns=cols, show="headings", height=6, selectmode="browse")
        for col, title, width in zip(cols, ("Karakter", "Cinsiyet", "Perde", "Satır"), (240, 120, 100, 80)):
            self.char_tree.heading(col, text=title)
            self.char_tree.column(col, width=width, anchor="w")
        self.char_tree.pack(fill="x")
        buttons = ttk.Frame(card)
        buttons.pack(fill="x", pady=(8, 0))
        for text, cmd in (("Kadın yap", lambda: self._char("kadin")), ("Erkek yap", lambda: self._char("erkek")),
                          ("Sesini dinle", self._char_sample), ("Sil", self._char_remove),
                          ("Hafızayı sıfırla", self._char_clear)):
            ttk.Button(buttons, text=text, command=cmd).pack(side="left", padx=(0, 6))
        self.char_hint = ttk.Label(card, text="Karakterler oyun sesinden otomatik tanınır; yanlış cinsiyeti buradan "
                                              "düzeltebilirsiniz.", style="Muted.TLabel")
        self.char_hint.pack(anchor="w", pady=(6, 0))
        self._chars_shown: list = []

    def _selected_char(self) -> int | None:
        sel = self.char_tree.selection()
        return int(sel[0]) if sel else None

    def _char(self, gender: str) -> None:
        cid = self._selected_char()
        if cid is not None:
            self.ctl.character_gender(cid, gender)
            self._fill_characters(force=True)

    def _char_sample(self) -> None:
        cid = self._selected_char()
        if cid is not None:
            error = self.ctl.character_sample(cid)
            self.char_hint.configure(text=error or "Örnek okunuyor…", foreground=RED if error else GRAY)

    def _char_remove(self) -> None:
        cid = self._selected_char()
        if cid is not None:
            self.ctl.character_remove(cid)
            self._fill_characters(force=True)

    def _char_clear(self) -> None:
        self.ctl.characters_clear()
        self._fill_characters(force=True)

    def _fill_characters(self, force: bool = False) -> None:
        chars = self.ctl.characters()
        key = [(c.id, c.gender, round(c.f0), c.lines, c.name) for c in chars]
        if not force and key == self._chars_shown:
            return
        self._chars_shown = key
        selected = self._selected_char()
        self.char_tree.delete(*self.char_tree.get_children())
        for c in chars:
            self.char_tree.insert("", "end", iid=str(c.id), values=(
                c.label, ("Kadın" if c.gender == "kadin" else "Erkek") + (" (elle)" if c.gender_locked else ""),
                f"{c.f0:.0f} Hz", c.lines))
        if selected is not None and self.char_tree.exists(str(selected)):
            self.char_tree.selection_set(str(selected))

    # Kısayollar -------------------------------------------------------------------------------

    def _tab_hotkeys(self, f: ttk.Frame) -> None:
        cols = ("islem", "kisayol", "durum")
        self.hk_tree = ttk.Treeview(f, columns=cols, show="headings", height=len(hk.ACTIONS),
                                    selectmode="browse")
        for col, title, width in zip(cols, ("İşlem", "Kısayol", "Durum"), (320, 170, 250)):
            self.hk_tree.heading(col, text=title)
            self.hk_tree.column(col, width=width, anchor="w")
        self.hk_tree.pack(fill="both", expand=True)
        self.hk_tree.bind("<Double-1>", lambda e: self._start_capture())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=8)
        ttk.Button(row, text="Değiştir", style=self._accent, command=self._start_capture).pack(side="left")
        ttk.Button(row, text="Kaldır", command=self._clear_hotkey).pack(side="left", padx=6)
        ttk.Button(row, text="Varsayılanlar", command=self._default_hotkeys).pack(side="left")
        ttk.Button(row, text="Boştaki kısayolları göster", command=self._show_free).pack(side="right")
        self.hk_hint = ttk.Label(f, text="Bir işlem seçip \"Değiştir\"e basın, ardından yeni tuş birleşimine basın "
                                         "(Ctrl ve/veya Alt ile). Esc: vazgeç.", style="Muted.TLabel")
        self.hk_hint.pack(anchor="w")
        self.free_lbl = ttk.Label(f, text="", wraplength=860, justify="left")
        self.free_lbl.pack(anchor="w", pady=(4, 0))
        self._fill_hotkeys()

    def _fill_hotkeys(self) -> None:
        selected = self.hk_tree.selection()
        self.hk_tree.delete(*self.hk_tree.get_children())
        for action, label, combo, state in self.ctl.hotkey_rows():
            self.hk_tree.insert("", "end", iid=action, values=(label, hk.display(combo), state))
        if selected and self.hk_tree.exists(selected[0]):
            self.hk_tree.selection_set(selected[0])

    def _selected_action(self) -> str | None:
        sel = self.hk_tree.selection()
        return sel[0] if sel else None

    def _start_capture(self) -> None:
        action = self._selected_action()
        if action is None:
            self.hk_hint.configure(text="Önce listeden bir işlem seçin.", foreground=RED)
            return
        self._capture_action = action
        self.hk_hint.configure(text=f"\"{hk.ACTION_LABELS[action]}\" için yeni tuş birleşimine basın… (Esc: vazgeç)",
                               foreground=ORANGE)
        self.win.focus_force()

    def _on_key(self, event) -> None:
        if self._capture_action is None:
            return
        if event.keysym == "Escape":
            self._capture_action = None
            self.hk_hint.configure(text="Vazgeçildi.", foreground=GRAY)
            return
        combo = hk.from_tk_event(event.keysym, event.state, getattr(event, "keycode", None))
        if combo is None:
            return  # yalnızca Ctrl/Alt'a basıldı; asıl tuşu bekle
        action, self._capture_action = self._capture_action, None
        error = self.ctl.set_hotkey(action, combo)
        if error:
            self.hk_hint.configure(text=error, foreground=RED)
        else:
            self.hk_hint.configure(text=f"{hk.ACTION_LABELS[action]}: {hk.display(hk.normalize(combo))}",
                                   foreground=GREEN)
        self._fill_hotkeys()

    def _clear_hotkey(self) -> None:
        action = self._selected_action()
        if action:
            self.ctl.set_hotkey(action, "")
            self._fill_hotkeys()

    def _default_hotkeys(self) -> None:
        for action, combo in hk.DEFAULT_HOTKEYS.items():
            self.ctl.set_hotkey(action, combo)
        self._fill_hotkeys()

    def _show_free(self) -> None:
        free = self.ctl.free_hotkeys()
        self.free_lbl.configure(text=("Boşta: " + ", ".join(hk.display(c) for c in free)) if free
                                else "Denenen birleşimlerin hiçbiri boşta değil.")

    # Kayıt ------------------------------------------------------------------------------------

    def _tab_log(self, f: ttk.Frame) -> None:
        card = self._card(f, "Oturum kaydı")
        r = self._check(card, 0, "Görülen satırları (İngilizce + Türkçe) dosyaya yaz", "session_log", restart=True,
                        switch=True)
        ttk.Label(card, text="Kaçırdığınız sahneleri okumak ve kötü çevirileri toplu bildirmek için.",
                  style="Muted.TLabel").grid(row=r, column=0, columnspan=3, sticky="w")
        ttk.Button(card, text="Kayıt klasörünü aç", command=lambda: self.ctl.open_folder("log")).grid(
            row=r + 1, column=0, sticky="w", pady=(8, 0))

    # --- alan yardımcıları ------------------------------------------------------------------

    def _changed(self, live: bool, restart: bool, dub: bool = False) -> None:
        self.ctl.save()
        if live:
            self.ctl.apply_style()
        if dub and self.cfg.tts_enabled:  # dublaj ayarı: dublajı yeni ayarla yeniden başlat
            self.ctl.set_tts(False)
            error = self.ctl.set_tts(True)
            if error:
                self.dub_var.set(False)
                self.note_lbl.configure(text=error, foreground=RED)
        if restart:
            self.note_lbl.configure(text="Bazı değişiklikler \"Kaydet ve yeniden başlat\" ile geçerli olur.",
                                    foreground=ORANGE)

    def _label(self, f, r, text, hint=None):
        ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", pady=4)
        if hint:
            ttk.Label(f, text=hint, style="Muted.TLabel", wraplength=300, justify="left").grid(
                row=r, column=2, sticky="w", padx=10)

    def _spin(self, f, r, text, key, lo, hi, step, live=False, restart=False, hint=None):
        self._label(f, r, text, hint)
        var = tk.StringVar(value=str(getattr(self.cfg, key)))

        def commit(*_):
            try:
                value = int(float(var.get()))
            except ValueError:
                return
            value = max(lo, min(hi, value))
            if value != getattr(self.cfg, key):
                setattr(self.cfg, key, value)
                self._changed(live, restart)

        sb = ttk.Spinbox(f, from_=lo, to=hi, increment=step, textvariable=var, width=8, command=commit)
        sb.grid(row=r, column=1, sticky="w", padx=8)
        sb.bind("<Return>", commit)
        sb.bind("<FocusOut>", commit)
        setattr(self, f"_var_{key}", var)
        return r + 1

    def _scale(self, f, r, text, key, lo, hi):
        self._label(f, r, text)
        var = tk.DoubleVar(value=float(getattr(self.cfg, key)))
        value_lbl = ttk.Label(f, text=f"{var.get():.2f}", width=5)

        def moved(_=None):
            value_lbl.configure(text=f"{var.get():.2f}")

        def commit(_=None):
            value = round(var.get(), 2)
            if value != getattr(self.cfg, key):
                setattr(self.cfg, key, value)
                self._changed(False, False)

        sc = ttk.Scale(f, from_=lo, to=hi, variable=var, command=moved, length=220)
        sc.grid(row=r, column=1, sticky="w", padx=8)
        sc.bind("<ButtonRelease-1>", commit)
        value_lbl.grid(row=r, column=2, sticky="w", padx=10)
        return r + 1

    def _check(self, f, r, text, key, live=False, restart=False, dub=False, switch=False):
        var = tk.BooleanVar(value=bool(getattr(self.cfg, key)))

        def commit():
            setattr(self.cfg, key, var.get())
            self._changed(live, restart, dub)

        ttk.Checkbutton(f, text=text, variable=var, command=commit,
                        style=self._switch if switch else "TCheckbutton").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=3)
        setattr(self, f"_var_{key}", var)
        return r + 1

    def _combo(self, f, r, text, key, values, live=False, restart=False, hint=None, dub=False):
        self._label(f, r, text, hint)
        var = tk.StringVar(value=str(getattr(self.cfg, key)))

        def commit(_=None):
            if var.get() != getattr(self.cfg, key):
                setattr(self.cfg, key, var.get())
                self._changed(live, restart, dub)

        cb = ttk.Combobox(f, textvariable=var, values=values, width=24, state="readonly")
        cb.grid(row=r, column=1, sticky="w", padx=8)
        cb.bind("<<ComboboxSelected>>", commit)
        return r + 1

    def _entry(self, f, r, text, key, restart=False, secret=False):
        self._label(f, r, text)
        var = tk.StringVar(value=str(getattr(self.cfg, key)))

        def commit(_=None):
            if var.get().strip() != getattr(self.cfg, key):
                setattr(self.cfg, key, var.get().strip())
                self._changed(False, restart)

        e = ttk.Entry(f, textvariable=var, width=30, show="•" if secret else "")
        e.grid(row=r, column=1, sticky="w", padx=8)
        e.bind("<Return>", commit)
        e.bind("<FocusOut>", commit)
        return r + 1

    # --- canlı durum -----------------------------------------------------------------------

    def refresh_style_vars(self) -> None:
        """Kısayolla yazı boyutu/konumu değişti: menüdeki değerler de güncellensin."""
        for key in ("font_size", "overlay_gap_px"):
            var = getattr(self, f"_var_{key}", None)
            if var is not None:
                var.set(str(getattr(self.cfg, key)))
        var = getattr(self, "_var_show_original", None)
        if var is not None:
            var.set(bool(self.cfg.show_original))

    def _refresh(self) -> None:
        try:
            if self.visible:
                paused = self.ctl.paused()
                if self.translate_var.get() == paused:
                    self.translate_var.set(not paused)
                if self.dub_var.get() != bool(self.cfg.tts_enabled):
                    self.dub_var.set(bool(self.cfg.tts_enabled))
                self.state_lbl.configure(
                    text=("Çeviri duraklatıldı" if paused else "Çeviri çalışıyor")
                    + ("  ·  Dublaj açık" if self.cfg.tts_enabled else "")
                    + f"  ·  Menü: {hk.display(self.cfg.hotkeys.get('menu', ''))}",
                    foreground=ORANGE if paused else GREEN)
                for row in self.ctl.region_rows():
                    w = self.region_widgets.get(row["name"])
                    if w:
                        w["status"].configure(text=row["status"], foreground=row["color"])
                        if w["var"].get() != row["enabled"]:
                            w["var"].set(row["enabled"])
                self.engine_lbl.configure(text="\n".join(f"{k}: {v}" for k, v in self.ctl.engine_rows()))
                en, tr, info = self.ctl.last_line()
                self.last_lbl.configure(text=f"{en}\n→ {tr}" if en else "Henüz satır yok.")
                self.last_info.configure(text=info)
                self.dub_status_lbl.configure(text="\n".join(f"{k}: {v}" for k, v in self.ctl.dub_status()))
                self._fill_characters()
        finally:
            self.win.after(self.REFRESH_MS, self._refresh)
