"""Uygulama: yakalama → OCR → kararlılık → DeepL → overlay (+ anlık yazı rengi)."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import config as config_mod
from . import contrast, imageproc, textproc, win32
from .config import EXTRA_REGION_KEYS, EXTRA_REGION_LABELS, Config, Region
from .geometry import Rect, intersect, region_to_screen, screen_to_region, union
from .ocr import OcrUnavailable, WindowsOcr
from .hybrid import DRAFT, FINAL, HybridTranslator

# Görüntü aynı kalsa bile bu kadar karede bir OCR tekrarlanır; altyazı belirirken (fade-in)
# yapılmış eksik bir okumada takılı kalmamak için.
MAX_REUSE = 4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def describe_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "QuotaExceededException":
        return "DeepL aylık karakter kotası doldu."
    if name == "AuthorizationException":
        return "DeepL anahtarı geçersiz (config.json → deepl_auth_key)."
    if name == "ConnectionException":
        return "DeepL'e bağlanılamadı (internet?)."
    return f"Çeviri hatası: {exc}"


MAIN = "altyazi"


class Pipeline(threading.Thread):
    """Bir ekran bölgesini okur ve çevirir. Ana altyazı ve her ek bölge (görev, telefon) için bir tane."""

    def __init__(self, cfg: Config, name: str, get_region: Callable[[], Region | None], display_mode: str,
                 translator: HybridTranslator, ui: queue.Queue, debug: bool,
                 sample_rect: Callable[[], Rect | None] = lambda: None):
        super().__init__(daemon=True, name=f"pipeline-{name}")
        self.cfg = cfg
        self.region_name = name  # (Thread.name ile karışmasın)
        self.get_region = get_region
        self.display_mode = display_mode
        self.translator = translator
        self.ui = ui
        self.debug = debug
        self.sample_rect = sample_rect
        self.paused = threading.Event()
        self.stopped = threading.Event()
        # Arayüz duraklat/devam ettiğinde: ekrandaki altyazı yeniden okunup gösterilsin.
        self.reset = threading.Event()
        # Çeviri başarısız oldu: aynı satır tekrar okununca yeniden dene.
        self._retry = threading.Event()
        self.seq = 0
        self.base = win32.primary_screen()
        self._previous_source: str | None = None
        self._retry_not_before = 0.0
        self._tag = "" if name == MAIN else f"[{EXTRA_REGION_LABELS.get(name, name)}] "

    def _on_show(self, event: textproc.Show) -> None:
        self.seq += 1
        seq = self.seq
        context, self._previous_source = self._previous_source, event.text
        log(f"{self._tag}EN: {event.text}")

        def on_result(stage: str, translated: str, took: float) -> None:
            if self.debug:
                log(f"{self._tag}{'taslak' if stage == DRAFT else 'çeviri'} {1000 * took:.0f} ms: {translated}")
            self.ui.put(("show", self.region_name, seq, stage, translated, event.text, event.first_seen))

        def on_error(exc: Exception, stage: str, draft_shown: bool) -> None:
            if stage == DRAFT:
                log(f"Yerel çeviri hatası: {exc}")
                return
            message = describe_error(exc)
            log(self._tag + message + (" (yerel çeviri ekranda kaldı)" if draft_shown else ""))
            if not draft_shown:
                self.ui.put(("error", self.region_name, message))
            if seq == self.seq:
                self._retry_not_before = time.monotonic() + 2.0  # çevrimdışıyken istek yağdırma
                self._retry.set()

        self.translator.submit(event.text, context, lambda: seq == self.seq, on_result, on_error)

    def _read_text(self, ocr: WindowsOcr, gray: np.ndarray) -> str:
        cfg = self.cfg
        bbox = imageproc.bright_bbox(gray, cfg.bright_threshold)
        if bbox is None:
            if cfg.skip_without_bright_text:
                return ""
        else:
            top, bottom, left, right = bbox
            gray = gray[top:bottom, left:right]
        img = imageproc.upscale(imageproc.for_ocr(gray, cfg.preprocess, cfg.bright_threshold), cfg.upscale)
        text = textproc.normalize(ocr.recognize(img))
        return text if textproc.is_plausible(text, cfg.min_text_chars) else ""

    def run(self) -> None:
        import mss

        try:
            ocr = WindowsOcr(self.cfg.ocr_language)
        except OcrUnavailable as exc:
            self.ui.put(("fatal", str(exc)))
            return
        except Exception as exc:  # pywinrt kurulu değil vb.
            self.ui.put(("fatal", f"Windows OCR başlatılamadı: {exc}"))
            return

        cfg = self.cfg
        stab = textproc.Stabilizer(cfg.stable_reads, cfg.similarity, cfg.clear_after_ms / 1000)
        picker = contrast.ColorPicker(cfg.outline_px) if self.display_mode != "cover" else None
        last_sig: np.ndarray | None = None
        last_text = ""
        reused = 0
        last_region: Rect | None = None
        last_base_check = 0.0
        last_window_title: str | None = None
        screen = win32.virtual_screen()
        interval = cfg.capture_interval_ms / 1000
        confirm_delay = min(interval, cfg.confirm_delay_ms / 1000)

        with mss.mss() as sct:
            while not self.stopped.is_set():
                tick = time.monotonic()
                if self.reset.is_set():
                    self.reset.clear()
                    self.seq += 1  # bekleyen/gelmekte olan çeviriler geçersiz
                    stab.reset()
                    last_sig, last_text, reused = None, "", 0
                if self._retry.is_set() and tick >= self._retry_not_before:
                    self._retry.clear()
                    stab.forget_shown()
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue

                spec = self.get_region()
                if spec is None:  # bu ek bölge seçilmemiş/kaldırılmış
                    if last_region is not None:
                        last_region = None
                        self.seq += 1
                        stab.reset()
                        self.ui.put(("region", self.region_name, None))
                    time.sleep(0.2)
                    continue

                if tick - last_base_check >= 1.0:
                    last_base_check = tick
                    screen = win32.virtual_screen()
                    found = win32.find_window(cfg.window_titles)
                    title = found[0] if found else None
                    if title != last_window_title and self.region_name == MAIN:
                        log(f"Hedef pencere: {title}" if title else "Hedef pencere bulunamadı, ana ekran kullanılıyor")
                    last_window_title = title
                    self.base = found[1] if found else win32.primary_screen()

                region = region_to_screen(spec, self.base)
                if region != last_region:
                    last_region = region
                    last_sig = None
                    self.ui.put(("region", self.region_name, region))

                # Bölge ile Türkçe yazının alanı tek seferde yakalanır.
                sample = self.sample_rect() if picker else None
                zone = intersect(union(region, sample) if sample else region, screen)
                if zone is None:
                    time.sleep(0.2)
                    continue
                try:
                    frame = np.asarray(sct.grab(
                        {"left": zone.left, "top": zone.top, "width": zone.width, "height": zone.height}))
                except Exception as exc:
                    log(f"Ekran yakalanamadı: {exc}")
                    time.sleep(0.5)
                    continue

                def crop(r: Rect) -> np.ndarray | None:
                    part = intersect(r, zone)
                    if part is None:
                        return None
                    return frame[part.top - zone.top:part.bottom - zone.top,
                                 part.left - zone.left:part.right - zone.left]

                # 1) Yazı rengi: Türkçenin arkasındaki sahneye göre, her karede.
                if picker and sample:
                    bg = crop(sample)
                    stats = contrast.background_stats(bg) if bg is not None else None
                    if stats is not None:
                        palette = picker.update(stats)
                        if palette is not None:
                            self.ui.put(("palette", self.region_name, palette))

                # 2) Metin.
                sub = crop(region)
                if sub is None:
                    time.sleep(interval)
                    continue
                gray = imageproc.to_gray(sub)
                sig = imageproc.signature(gray, cfg.bright_threshold)
                if imageproc.same_signature(last_sig, sig) and reused < MAX_REUSE:
                    text = last_text
                    reused += 1
                else:
                    reused = 0
                    t0 = time.perf_counter()
                    text = self._read_text(ocr, gray)
                    if self.debug and text != last_text:
                        log(f"{self._tag}OCR {1000 * (time.perf_counter() - t0):.0f} ms: {text!r}")
                    last_sig, last_text = sig, text

                event = stab.feed(text, tick)
                if isinstance(event, textproc.Show):
                    self._on_show(event)
                elif isinstance(event, textproc.Clear):
                    self.seq += 1
                    self.ui.put(("clear", self.region_name))

                # Doğrulanmayı bekleyen satır varsa hemen tekrar oku; yoksa normal kare aralığı.
                wait = confirm_delay if stab.pending else interval
                elapsed = time.monotonic() - tick
                if elapsed < wait:
                    time.sleep(wait - elapsed)

        self.translator.shutdown()


def run_gui(cfg: Config, cfg_path: Path, make_translator: Callable[[], HybridTranslator], debug: bool) -> None:
    import tkinter as tk
    from tkinter import messagebox

    from .overlay import REMOVE, HistoryPanel, Overlay, RegionPicker

    root = tk.Tk()
    root.withdraw()
    ui: queue.Queue = queue.Queue()

    overlays: dict[str, Overlay] = {MAIN: Overlay(root, cfg, cfg.display_mode)}
    pipelines: dict[str, Pipeline] = {}
    main_overlay = overlays[MAIN]
    if not main_overlay.capture_excluded:
        log("UYARI: Overlay ekran yakalamadan gizlenemedi (Windows 10 2004 veya üstü gerekir). "
            "OCR kendi Türkçe yazımızı/kutumuzu okuyabilir.")
    pipelines[MAIN] = Pipeline(cfg, MAIN, lambda: cfg.region, cfg.display_mode, make_translator(), ui, debug,
                               sample_rect=lambda: main_overlay.sample_rect)
    for name in EXTRA_REGION_KEYS.values():
        def get_region(name=name):
            extra = cfg.extra(name)
            return extra.region if extra else None
        mode = cfg.extra(name).display_mode if cfg.extra(name) else "cover"
        overlays[name] = Overlay(root, cfg, mode)
        pipelines[name] = Pipeline(cfg, name, get_region, mode, make_translator(), ui, debug,
                                   sample_rect=lambda ov=overlays[name]: ov.sample_rect)
    history = HistoryPanel(root, cfg)
    entries: list[dict] = []  # son satırlar: {"key": (bölge, seq), "label", "en", "tr"}
    state = {"welcome": True, "picking": False, "paused": False}
    final_seq = {name: -1 for name in pipelines}

    def all_pipelines():
        return pipelines.values()

    def quit_app() -> None:
        for p in all_pipelines():
            p.stopped.set()
        hotkeys.stop()
        root.destroy()

    def toggle_pause() -> None:
        state["paused"] = not state["paused"]
        for p in all_pipelines():
            p.reset.set()
            (p.paused.set if state["paused"] else p.paused.clear)()
        for ov in overlays.values():
            if state["paused"]:
                ov.clear()
                ov.hide()
            else:
                ov.unhide()
        log("Duraklatıldı (Ctrl+Alt+S ile devam)" if state["paused"] else "Devam ediliyor")

    def pick(name: str) -> None:
        if state["picking"]:
            return
        state["picking"] = True
        for p in all_pipelines():
            p.reset.set()
            p.paused.set()
        for ov in overlays.values():
            ov.clear()
            ov.hide()
        history.hide()
        label = "İngilizce altyazının" if name == MAIN else f"{EXTRA_REGION_LABELS[name]} yazısının"

        def done(result) -> None:
            state["picking"] = False
            base = pipelines[MAIN].base
            if name == MAIN and isinstance(result, Rect):
                cfg.region = screen_to_region(result, base)
                log(f"Altyazı bölgesi kaydedildi: {cfg.region}")
            elif isinstance(result, Rect):
                cfg.set_extra(name, screen_to_region(result, base))
                log(f"{EXTRA_REGION_LABELS[name]} bölgesi kaydedildi")
            elif result is REMOVE and name != MAIN:
                cfg.set_extra(name, None)
                log(f"{EXTRA_REGION_LABELS[name]} bölgesi kaldırıldı")
            config_mod.save(cfg, cfg_path)
            if not state["paused"]:
                for ov in overlays.values():
                    ov.unhide()
                for p in all_pipelines():
                    p.paused.clear()
            if history.visible_wanted:
                history.show(entries, pipelines[MAIN].base)

        RegionPicker(root, done, f"{label} çıktığı alanı fareyle seçin", allow_remove=name != MAIN)

    def toggle_original() -> None:
        cfg.show_original = not cfg.show_original
        log(f"İngilizce satır gösterimi: {'açık' if cfg.show_original else 'kapalı'}")

    def toggle_history() -> None:
        history.toggle(entries, pipelines[MAIN].base)

    def remember(name: str, seq: int, original: str, translated: str) -> None:
        key = (name, seq)
        for entry in entries:
            if entry["key"] == key:  # taslağın yerine DeepL geldi
                entry["tr"] = translated
                break
        else:
            entries.append({"key": key, "label": "" if name == MAIN else EXTRA_REGION_LABELS.get(name, name),
                            "en": original, "tr": translated})
            del entries[:-max(1, cfg.history_size)]
        history.refresh(entries, pipelines[MAIN].base)

    actions = {"pause": toggle_pause, "pick": lambda: pick(MAIN), "original": toggle_original,
               "history": toggle_history, "quit": quit_app}
    keys = {"S": "pause", "R": "pick", "O": "original", "H": "history", "Q": "quit"}
    for key, name in EXTRA_REGION_KEYS.items():
        actions[f"pick-{name}"] = lambda name=name: pick(name)
        keys[key] = f"pick-{name}"
    hotkeys = win32.HotkeyThread({key: (lambda a=a: ui.put(("action", a))) for key, a in keys.items()})

    def welcome_off() -> None:
        if state["welcome"]:
            state["welcome"] = False
            main_overlay.clear()

    def poll() -> None:
        try:
            while True:
                msg = ui.get_nowait()
                kind = msg[0]
                if kind == "region":
                    _, name, rect = msg
                    if rect is None:
                        overlays[name].clear()
                    overlays[name].set_region(rect)
                    if name == MAIN and state["welcome"]:
                        main_overlay.status("Türkçe altyazı aktif  •  Ctrl+Alt+R altyazı  •  Ctrl+Alt+G görev  •  "
                                            "Ctrl+Alt+T telefon  •  Ctrl+Alt+H geçmiş  •  Ctrl+Alt+Q çık")
                        root.after(6000, welcome_off)
                elif kind == "palette":
                    _, name, palette = msg
                    overlays[name].set_palette(palette)
                    if debug:
                        log(f"yazı rengi: {palette.fill} (kontur {palette.outline_px}px)")
                elif kind == "show":
                    _, name, seq, stage, translated, original, first_seen = msg
                    # Eski satırı ya da DeepL'den sonra gelen taslağı gösterme.
                    if seq == pipelines[name].seq and not (stage == DRAFT and final_seq[name] == seq):
                        if stage == FINAL:
                            final_seq[name] = seq
                        if name == MAIN:
                            state["welcome"] = False
                        overlays[name].show(translated, original)
                        remember(name, seq, original, translated)
                        if debug:
                            what = "taslak" if stage == DRAFT else "son çeviri"
                            log(f"ilk okumadan ekrana ({what}): {1000 * (time.monotonic() - first_seen):.0f} ms")
                elif kind == "clear":
                    name = msg[1]
                    if not (name == MAIN and state["welcome"]):
                        overlays[name].clear()
                elif kind == "error":
                    overlays[msg[1]].status(msg[2])
                elif kind == "fatal":
                    log(msg[1])
                    messagebox.showerror("GTA Altyazı", msg[1])
                    quit_app()
                    return
                elif kind == "action":
                    actions[msg[1]]()
                    if msg[1] == "quit":
                        return
        except queue.Empty:
            pass
        root.after(10, poll)

    hotkeys.start()
    for p in all_pipelines():
        p.start()
    root.after(10, poll)
    root.after(1500, lambda: hotkeys.failed and log(
        "Bu kısayollar başka bir program tarafından kullanılıyor: " + ", ".join(hotkeys.failed)))
    log("Çalışıyor. Ctrl+Alt+R: altyazı bölgesi, Ctrl+Alt+G: görev bölgesi, Ctrl+Alt+T: telefon bölgesi, "
        "Ctrl+Alt+H: son satırlar, Ctrl+Alt+S: duraklat/devam, Ctrl+Alt+O: İngilizceyi de göster, Ctrl+Alt+Q: çıkış")
    try:
        root.mainloop()
    finally:
        for p in all_pipelines():
            p.stopped.set()


def run_ocr_test(cfg: Config, image_path: Path, translator: HybridTranslator, full: bool) -> None:
    """Bir ekran görüntüsü üzerinde OCR + çeviriyi dener (ayar yapmak için)."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img)
    bgra = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    if not full:
        r = region_to_screen(cfg.region, Rect(0, 0, img.width, img.height))
        bgra = bgra[r.top:r.bottom, r.left:r.right]
        log(f"Bölge (piksel): x={r.left} y={r.top} {r.width}x{r.height}")
    gray = imageproc.to_gray(bgra)
    bbox = imageproc.bright_bbox(gray, cfg.bright_threshold)
    if bbox is not None:
        top, bottom, left, right = bbox
        gray = gray[top:bottom, left:right]
    else:
        log("Bölgede parlak yazı bulunamadı (bright_threshold çok yüksek olabilir)")
    ocr_input = imageproc.upscale(imageproc.for_ocr(gray, cfg.preprocess, cfg.bright_threshold), cfg.upscale)
    debug_path = image_path.with_name(image_path.stem + "_ocr_girdi.png")
    Image.fromarray(ocr_input).save(debug_path)
    log(f"OCR'a verilen görüntü kaydedildi: {debug_path}")

    ocr = WindowsOcr(cfg.ocr_language)
    t0 = time.perf_counter()
    lines = ocr.recognize(ocr_input)
    log(f"OCR süresi: {1000 * (time.perf_counter() - t0):.0f} ms")
    for line in lines:
        print("  satır:", line)
    text = textproc.normalize(lines)
    print("İngilizce:", text or "(metin yok)")
    for name, backend in (("Yerel", translator.draft), ("DeepL", translator.final)):
        if text and backend is not None:
            t0 = time.perf_counter()
            out = backend.translate(text, None)
            print(f"Türkçe ({name}): {out}   [{1000 * (time.perf_counter() - t0):.0f} ms]")
