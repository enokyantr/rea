"""Ayar menüsü (Ctrl+Alt+M): durum, görünüm, çeviri, OCR, kısayollar, ses ve kayıt.

Menü yalnızca `controller` üzerinden çalışır (app.run_gui sağlar); Windows'a, çeviri motorlarına ve
OCR'a doğrudan dokunmaz. Anında uygulanan ayarlar (görünüm, kısayollar, aç/kapat) hemen kaydedilir;
motor değiştiren ayarlar (çevirmen, model, OCR motoru) "Kaydet ve yeniden başlat" ile uygulanır."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Protocol

from . import hotkeys as hk

GREEN, RED, GRAY, ORANGE = "#1f8b4c", "#c0392b", "#7f8c8d", "#d68910"


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
    def open_folder(self, which: str) -> None: ...
    def desktop_shortcut(self) -> str: ...


class ControlPanel:
    REFRESH_MS = 500

    def __init__(self, root: tk.Misc, ctl: Controller):
        self.ctl = ctl
        self.cfg = ctl.cfg
        self.win = tk.Toplevel(root)
        self.win.title("GTA Altyazı — Menü")
        self.win.geometry("880x600")
        self.win.minsize(640, 480)
        self.win.protocol("WM_DELETE_WINDOW", self.hide)
        self._capture_action: str | None = None
        self._restart_needed = False
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
        top = ttk.Frame(self.win, padding=(10, 8))
        top.pack(fill="x")
        self.pause_btn = ttk.Button(top, text="Duraklat", width=14, command=lambda: self.ctl.action("pause"))
        self.pause_btn.pack(side="left")
        self.state_lbl = tk.Label(top, text="", font=("Segoe UI", 11, "bold"))
        self.state_lbl.pack(side="left", padx=12)
        ttk.Button(top, text="Programdan çık", command=lambda: self.ctl.action("quit")).pack(side="right")
        self.restart_btn = ttk.Button(top, text="Kaydet ve yeniden başlat", command=self.ctl.restart)
        self.restart_btn.pack(side="right", padx=6)
        ttk.Button(top, text="Menüyü gizle", command=self.hide).pack(side="right")

        nb = ttk.Notebook(self.win)
        nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        for title, build in (("Durum", self._tab_status), ("Görünüm", self._tab_look),
                             ("Çeviri", self._tab_translation), ("OCR ve hız", self._tab_ocr),
                             ("Kısayollar", self._tab_hotkeys), ("Ses ve kayıt", self._tab_audio)):
            frame = ttk.Frame(nb, padding=10)
            nb.add(frame, text=title)
            build(frame)

        self.note_lbl = tk.Label(self.win, text="", fg=ORANGE, anchor="w")
        self.note_lbl.pack(fill="x", padx=12, pady=(0, 6))

    # Durum ------------------------------------------------------------------------------------

    def _tab_status(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="Okuma bölgeleri", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.region_widgets: dict[str, dict] = {}
        for i, row in enumerate(self.ctl.region_rows(), start=1):
            name = row["name"]
            ttk.Label(f, text=row["label"], width=10).grid(row=i, column=0, sticky="w", pady=3)
            status = tk.Label(f, text="", anchor="w", width=24)
            status.grid(row=i, column=1, sticky="w")
            var = tk.BooleanVar(value=row["enabled"])
            ttk.Checkbutton(f, text="Açık", variable=var,
                            command=lambda n=name, v=var: self.ctl.set_region_enabled(n, v.get())).grid(
                row=i, column=2, padx=4)
            ttk.Button(f, text="Bölge seç", command=lambda n=name: self.ctl.action(
                "pick" if n == "altyazi" else f"pick-{n}")).grid(row=i, column=3, padx=2)
            ttk.Button(f, text="Hazır konum", command=lambda n=name: self.ctl.preset_region(n)).grid(
                row=i, column=4, padx=2)
            if name != "altyazi":
                ttk.Button(f, text="Kaldır", command=lambda n=name: self.ctl.remove_region(n)).grid(
                    row=i, column=5, padx=2)
            self.region_widgets[name] = {"status": status, "var": var}
        base = len(self.region_widgets) + 2
        ttk.Separator(f).grid(row=base, column=0, columnspan=6, sticky="ew", pady=8)
        ttk.Label(f, text="Motorlar", font=("Segoe UI", 10, "bold")).grid(row=base + 1, column=0, sticky="w")
        self.engine_lbl = ttk.Label(f, text="", justify="left")
        self.engine_lbl.grid(row=base + 2, column=0, columnspan=6, sticky="w")
        ttk.Separator(f).grid(row=base + 3, column=0, columnspan=6, sticky="ew", pady=8)
        ttk.Label(f, text="Son satır", font=("Segoe UI", 10, "bold")).grid(row=base + 4, column=0, sticky="w")
        self.last_lbl = ttk.Label(f, text="", justify="left", wraplength=820)
        self.last_lbl.grid(row=base + 5, column=0, columnspan=6, sticky="w")
        ttk.Label(f, text="Hazır konumlar GTA V düzenine göre tahminidir; GTA 6'da \"Bölge seç\" ile fareyle seçin.",
                  foreground=GRAY).grid(row=base + 6, column=0, columnspan=6, sticky="w", pady=(10, 0))
        shortcut_lbl = ttk.Label(f, text="", foreground=GRAY)
        ttk.Button(f, text="Masaüstüne kısayol ekle",
                   command=lambda: shortcut_lbl.configure(text=self.ctl.desktop_shortcut())).grid(
            row=base + 7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        shortcut_lbl.grid(row=base + 7, column=2, columnspan=4, sticky="w", pady=(10, 0))

    # Görünüm ----------------------------------------------------------------------------------

    def _tab_look(self, f: ttk.Frame) -> None:
        cfg = self.cfg
        r = 0
        r = self._spin(f, r, "Yazı boyutu", "font_size", 12, 72, 1, live=True)
        r = self._spin(f, r, "Konum (İngilizcenin ne kadar üstünde, piksel)", "overlay_gap_px", -300, 600, 4,
                       live=True)
        r = self._spin(f, r, "Kontur kalınlığı", "outline_px", 0, 6, 1, live=True)
        families = sorted({"Segoe UI", "Arial", "Calibri", "Tahoma", "Verdana", "Bahnschrift", cfg.font_family})
        r = self._combo(f, r, "Yazı tipi", "font_family", families, live=True)
        r = self._check(f, r, "İngilizceyi de göster (küçük, Türkçenin üstünde)", "show_original", live=True)
        r = self._check(f, r, "Yayında (OBS) ve ekran görüntüsünde görünsün", "capture_visible", restart=True)
        r = self._combo(f, r, "Gösterim", "display_mode", ["adaptive", "cover"], restart=True,
                        hint="adaptive: İngilizcenin üstünde, renk sahneye göre · cover: İngilizceyi örten kutu")
        r = self._spin(f, r, "Son satırlar paneli: satır sayısı", "history_size", 3, 30, 1)

    # Çeviri -----------------------------------------------------------------------------------

    def _tab_translation(self, f: ttk.Frame) -> None:
        r = 0
        r = self._combo(f, r, "Son çevirmen", "final_translator", ["ollama", "deepl"], restart=True,
                        hint="ollama: ücretsiz, sansürsüz, ekran kartında · deepl: internet + anahtar")
        r = self._entry(f, r, "Yerel dil modeli (Ollama)", "ollama_model", restart=True)
        r = self._check(f, r, "Dil modeli yalnızca küfür/argo/sözlük satırlarında (önerilen)",
                        "ollama_only_profanity", restart=True)
        r = self._check(f, r, "Yarım cümlelerde önceki satırı bağlam olarak gönder", "ollama_context_continuation",
                        restart=True)
        r = self._check(f, r, "Sansürsüz çeviri (küfür yumuşatılmaz)", "uncensored", restart=True)
        r = self._entry(f, r, "DeepL anahtarı", "deepl_auth_key", restart=True, secret=True)
        ttk.Button(f, text="Sözlüğü aç (isim/terim karşılıkları)",
                   command=lambda: self.ctl.open_folder("glossary")).grid(row=r, column=0, sticky="w", pady=8)

    # OCR --------------------------------------------------------------------------------------

    def _tab_ocr(self, f: ttk.Frame) -> None:
        r = 0
        r = self._combo(f, r, "OCR motoru", "ocr_engine", ["auto", "windows"], restart=True,
                        hint="auto: RapidOCR (önerilen) · windows: Windows OCR")
        r = self._combo(f, r, "OCR nerede çalışsın", "ocr_device", ["auto", "cpu"], restart=True,
                        hint="auto: ekran kartı (DirectML) · cpu: dil modeli yavaşlarsa")
        r = self._scale(f, r, "Hızlı kabul güveni (yüksek = daha az hata, biraz daha geç)",
                        "fast_accept_confidence", 0.80, 1.00)
        r = self._check(f, r, "Yalnızca yazı değişince yeniden oku (önerilen)", "change_detection")
        r = self._spin(f, r, "Altyazı gidince Türkçe kaç ms sonra silinsin", "clear_after_ms", 200, 3000, 100,
                       restart=True)

    # Kısayollar -------------------------------------------------------------------------------

    def _tab_hotkeys(self, f: ttk.Frame) -> None:
        cols = ("islem", "kisayol", "durum")
        self.hk_tree = ttk.Treeview(f, columns=cols, show="headings", height=len(hk.ACTIONS),
                                    selectmode="browse")
        for col, title, width in zip(cols, ("İşlem", "Kısayol", "Durum"), (280, 150, 230)):
            self.hk_tree.heading(col, text=title)
            self.hk_tree.column(col, width=width, anchor="w")
        self.hk_tree.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.hk_tree.bind("<Double-1>", lambda e: self._start_capture())
        f.rowconfigure(0, weight=1)
        f.columnconfigure(3, weight=1)
        ttk.Button(f, text="Değiştir", command=self._start_capture).grid(row=1, column=0, sticky="w", pady=6)
        ttk.Button(f, text="Kaldır", command=self._clear_hotkey).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(f, text="Varsayılanlar", command=self._default_hotkeys).grid(row=1, column=2, sticky="w")
        self.hk_hint = ttk.Label(f, text="Bir işlem seçip \"Değiştir\"e basın, ardından yeni tuş birleşimine basın "
                                         "(Ctrl ve/veya Alt ile). Esc: vazgeç.", foreground=GRAY)
        self.hk_hint.grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Button(f, text="Boştaki kısayolları göster", command=self._show_free).grid(row=3, column=0, sticky="w",
                                                                                         pady=(10, 2))
        self.free_lbl = ttk.Label(f, text="", wraplength=820, justify="left")
        self.free_lbl.grid(row=4, column=0, columnspan=4, sticky="w")
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
        self.hk_hint.configure(text=f"\"{hk.ACTION_LABELS[action]}\" için yeni tuş birleşimine basın... (Esc: vazgeç)",
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

    # Ses ve kayıt -----------------------------------------------------------------------------

    def _tab_audio(self, f: ttk.Frame) -> None:
        r = 0
        self.tts_var = tk.BooleanVar(value=self.cfg.tts_enabled)
        ttk.Checkbutton(f, text="Sesli okuma (dublaj): Türkçe satırı sesli oku", variable=self.tts_var,
                        command=self._toggle_tts).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        voices = self.ctl.tts_voices()
        r = self._combo(f, r, "Ses", "tts_voice", [""] + voices, restart=True,
                        hint="boş: ilk Türkçe ses" if voices else
                        "Türkçe ses yok: Ayarlar > Saat ve dil > Konuşma > Ses ekle > Türkçe")
        r = self._scale(f, r, "Konuşma hızı", "tts_rate", 0.5, 2.5)
        r = self._scale(f, r, "Ses düzeyi", "tts_volume", 0.0, 1.0)
        self.tts_lbl = ttk.Label(f, text="", foreground=GRAY)
        self.tts_lbl.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=10)
        r += 1
        r = self._check(f, r, "Oturum kaydı: görülen satırları (İngilizce + Türkçe) dosyaya yaz", "session_log",
                        restart=True)
        ttk.Button(f, text="Kayıt klasörünü aç", command=lambda: self.ctl.open_folder("log")).grid(
            row=r, column=0, sticky="w")

    def _toggle_tts(self) -> None:
        error = self.ctl.set_tts(self.tts_var.get())
        self.tts_lbl.configure(text=error or ("Sesli okuma açık" if self.tts_var.get() else "Sesli okuma kapalı"),
                               foreground=RED if error else GRAY)
        if error:
            self.tts_var.set(False)

    # --- alan yardımcıları ------------------------------------------------------------------

    def _changed(self, live: bool, restart: bool) -> None:
        self.ctl.save()
        if live:
            self.ctl.apply_style()
        if restart:
            self._restart_needed = True
            self.note_lbl.configure(text="Bazı değişiklikler \"Kaydet ve yeniden başlat\" ile geçerli olur.")

    def _label(self, f, r, text, hint=None):
        ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", pady=3)
        if hint:
            ttk.Label(f, text=hint, foreground=GRAY, wraplength=280, justify="left").grid(
                row=r, column=2, sticky="w", padx=8)

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
        sb.grid(row=r, column=1, sticky="w", padx=6)
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

        sc = ttk.Scale(f, from_=lo, to=hi, variable=var, command=moved, length=200)
        sc.grid(row=r, column=1, sticky="w", padx=6)
        sc.bind("<ButtonRelease-1>", commit)
        value_lbl.grid(row=r, column=2, sticky="w")
        return r + 1

    def _check(self, f, r, text, key, live=False, restart=False):
        var = tk.BooleanVar(value=bool(getattr(self.cfg, key)))

        def commit():
            setattr(self.cfg, key, var.get())
            self._changed(live, restart)

        ttk.Checkbutton(f, text=text, variable=var, command=commit).grid(row=r, column=0, columnspan=3, sticky="w",
                                                                         pady=2)
        setattr(self, f"_var_{key}", var)
        return r + 1

    def _combo(self, f, r, text, key, values, live=False, restart=False, hint=None):
        self._label(f, r, text, hint)
        var = tk.StringVar(value=str(getattr(self.cfg, key)))

        def commit(_=None):
            if var.get() != getattr(self.cfg, key):
                setattr(self.cfg, key, var.get())
                self._changed(live, restart)

        cb = ttk.Combobox(f, textvariable=var, values=values, width=22, state="readonly")
        cb.grid(row=r, column=1, sticky="w", padx=6)
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
        e.grid(row=r, column=1, sticky="w", padx=6)
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
                self.pause_btn.configure(text="Devam ettir" if paused else "Duraklat")
                self.state_lbl.configure(text="Duraklatıldı" if paused else "Çeviri çalışıyor",
                                         fg=ORANGE if paused else GREEN)
                for row in self.ctl.region_rows():
                    w = self.region_widgets.get(row["name"])
                    if w:
                        w["status"].configure(text=row["status"], fg=row["color"])
                        if w["var"].get() != row["enabled"]:
                            w["var"].set(row["enabled"])
                self.engine_lbl.configure(text="\n".join(f"{k}: {v}" for k, v in self.ctl.engine_rows()))
                en, tr, info = self.ctl.last_line()
                self.last_lbl.configure(text=f"EN: {en}\nTR: {tr}\n{info}" if en else "Henüz satır yok.")
                if self.cfg.tts_enabled != self.tts_var.get():
                    self.tts_var.set(self.cfg.tts_enabled)
        finally:
            self.win.after(self.REFRESH_MS, self._refresh)
