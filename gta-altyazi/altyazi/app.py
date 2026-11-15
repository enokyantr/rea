"""Uygulama: yakalama → OCR → kararlılık → DeepL → overlay (+ anlık yazı rengi)."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from . import config as config_mod
from . import contrast, imageproc, textproc, win32
from .config import Config
from .geometry import Rect, intersect, region_to_screen, screen_to_region, union
from .ocr import OcrUnavailable, WindowsOcr
from .translate import CachedTranslator

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


class Pipeline(threading.Thread):
    def __init__(self, cfg: Config, translator: CachedTranslator, ui: queue.Queue, debug: bool,
                 sample_rect: Callable[[], Rect | None] = lambda: None):
        super().__init__(daemon=True, name="pipeline")
        self.cfg = cfg
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepl")
        self._previous_source: str | None = None

    def _translate(self, seq: int, text: str, context: str | None, first_seen: float) -> None:
        if seq != self.seq:  # kuyrukta beklerken yeni satır geldi: boşuna çevirme
            return
        started = time.perf_counter()
        try:
            translated = self.translator.translate(text, context)
        except Exception as exc:  # ağ/kota hataları oyunu durdurmamalı
            message = describe_error(exc)
            log(message)
            self.ui.put(("error", message))
            if seq == self.seq:
                self._retry.set()
            return
        if self.debug:
            log(f"çeviri {1000 * (time.perf_counter() - started):.0f} ms: {translated}")
        if seq == self.seq:  # bu arada yeni satır geldiyse eski çeviriyi gösterme
            self.ui.put(("show", seq, translated, text, first_seen))

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
        picker = contrast.ColorPicker(cfg.outline_px) if cfg.display_mode != "cover" else None
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
                if self._retry.is_set():
                    self._retry.clear()
                    stab.forget_shown()
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue

                if tick - last_base_check >= 1.0:
                    last_base_check = tick
                    screen = win32.virtual_screen()
                    found = win32.find_window(cfg.window_titles)
                    title = found[0] if found else None
                    if title != last_window_title:
                        log(f"Hedef pencere: {title}" if title else "Hedef pencere bulunamadı, ana ekran kullanılıyor")
                        last_window_title = title
                    self.base = found[1] if found else win32.primary_screen()

                region = region_to_screen(cfg.region, self.base)
                if region != last_region:
                    last_region = region
                    last_sig = None
                    self.ui.put(("region", region))

                # Altyazı bölgesi ile Türkçe yazının alanı tek seferde yakalanır.
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
                            self.ui.put(("palette", palette))

                # 2) Altyazı metni.
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
                        log(f"OCR {1000 * (time.perf_counter() - t0):.0f} ms: {text!r}")
                    last_sig, last_text = sig, text

                event = stab.feed(text, tick)
                if isinstance(event, textproc.Show):
                    self.seq += 1
                    context, self._previous_source = self._previous_source, event.text
                    log(f"EN: {event.text}")
                    self._executor.submit(self._translate, self.seq, event.text, context, event.first_seen)
                elif isinstance(event, textproc.Clear):
                    self.seq += 1
                    self.ui.put(("clear",))

                # Doğrulanmayı bekleyen satır varsa hemen tekrar oku; yoksa normal kare aralığı.
                wait = confirm_delay if stab.pending else interval
                elapsed = time.monotonic() - tick
                if elapsed < wait:
                    time.sleep(wait - elapsed)

        self._executor.shutdown(wait=False, cancel_futures=True)


def run_gui(cfg: Config, cfg_path: Path, translator: CachedTranslator, debug: bool) -> None:
    import tkinter as tk
    from tkinter import messagebox

    from .overlay import Overlay, RegionPicker

    root = tk.Tk()
    root.withdraw()
    overlay = Overlay(root, cfg)
    if not overlay.capture_excluded:
        log("UYARI: Overlay ekran yakalamadan gizlenemedi (Windows 10 2004 veya üstü gerekir). "
            + ("'cover' modunda OCR kendi kutumuzu okuyabilir; 'adaptive' modunu kullanın."
               if cfg.display_mode == "cover" else "Renk seçimi kendi yazımızdan etkilenebilir."))
    ui: queue.Queue = queue.Queue()
    pipeline = Pipeline(cfg, translator, ui, debug, sample_rect=lambda: overlay.sample_rect)
    state = {"welcome": True, "picking": False}

    def quit_app() -> None:
        pipeline.stopped.set()
        hotkeys.stop()
        root.destroy()

    def toggle_pause() -> None:
        pipeline.reset.set()
        if pipeline.paused.is_set():
            pipeline.paused.clear()
            overlay.unhide()
            log("Devam ediliyor")
        else:
            pipeline.paused.set()
            overlay.clear()
            overlay.hide()
            log("Duraklatıldı (Ctrl+Alt+S ile devam)")

    def pick_region() -> None:
        if state["picking"]:
            return
        state["picking"] = True
        was_paused = pipeline.paused.is_set()
        pipeline.reset.set()
        pipeline.paused.set()
        overlay.clear()
        overlay.hide()

        def done(rect: Rect | None) -> None:
            state["picking"] = False
            if rect is not None:
                cfg.region = screen_to_region(rect, pipeline.base)
                config_mod.save(cfg, cfg_path)
                log(f"Yeni altyazı bölgesi kaydedildi: {cfg.region}")
            overlay.unhide()
            if not was_paused:
                pipeline.paused.clear()

        RegionPicker(root, done)

    def toggle_original() -> None:
        cfg.show_original = not cfg.show_original
        log(f"İngilizce satır gösterimi: {'açık' if cfg.show_original else 'kapalı'}")

    actions = {"pause": toggle_pause, "pick": pick_region, "original": toggle_original, "quit": quit_app}
    hotkeys = win32.HotkeyThread(
        {key: (lambda name=name: ui.put(("action", name))) for key, name in
         {"S": "pause", "R": "pick", "O": "original", "Q": "quit"}.items()}
    )

    def welcome_off() -> None:
        if state["welcome"]:
            state["welcome"] = False
            overlay.clear()

    def poll() -> None:
        try:
            while True:
                msg = ui.get_nowait()
                kind = msg[0]
                if kind == "region":
                    overlay.set_region(msg[1])
                    if state["welcome"]:
                        overlay.status("Türkçe altyazı aktif  •  Ctrl+Alt+R bölge seç  •  Ctrl+Alt+S duraklat  •  Ctrl+Alt+Q çık")
                        root.after(5000, welcome_off)
                elif kind == "palette":
                    overlay.set_palette(msg[1])
                    if debug:
                        log(f"yazı rengi: {msg[1].fill} (kontur {msg[1].outline_px}px)")
                elif kind == "show":
                    _, seq, translated, original, first_seen = msg
                    if seq == pipeline.seq:
                        state["welcome"] = False
                        overlay.show(translated, original)
                        if debug:
                            log(f"ilk okumadan ekrana: {1000 * (time.monotonic() - first_seen):.0f} ms")
                elif kind == "clear":
                    if not state["welcome"]:
                        overlay.clear()
                elif kind == "error":
                    overlay.status(msg[1])
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
    pipeline.start()
    root.after(10, poll)
    root.after(1500, lambda: hotkeys.failed and log(
        "Bu kısayollar başka bir program tarafından kullanılıyor: " + ", ".join(hotkeys.failed)))
    log("Çalışıyor. Ctrl+Alt+R: altyazı bölgesini seç, Ctrl+Alt+S: duraklat/devam, "
        "Ctrl+Alt+O: İngilizceyi de göster, Ctrl+Alt+Q: çıkış")
    try:
        root.mainloop()
    finally:
        pipeline.stopped.set()


def run_ocr_test(cfg: Config, image_path: Path, translator: CachedTranslator, full: bool) -> None:
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
    if text:
        t0 = time.perf_counter()
        print("Türkçe:   ", translator.translate(text))
        log(f"Çeviri süresi: {1000 * (time.perf_counter() - t0):.0f} ms")
