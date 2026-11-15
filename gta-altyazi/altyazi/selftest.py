"""Windows'ta gerçek bileşenleri dener: OCR, overlay, ekran yakalamadan gizleme, DeepL gecikmesi.

Kullanım:  GTA-Altyazi.exe --self-test   (veya baslat.bat --self-test)"""

from __future__ import annotations

import dataclasses
import statistics
import time

import numpy as np

from . import contrast, imageproc, textproc, win32
from .bench import BENCH_LINES
from .config import Config
from .geometry import Rect

SAMPLE = "You think this is a game, Jason? Get in the car!"


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def item(self, name: str, status: str, detail: str = "") -> None:
        if status == "HATA":
            self.failed += 1
        print(f"[{status:^9}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _render_subtitle(text: str, background=(40, 55, 70)) -> np.ndarray:
    """Oyun altyazısına benzer görüntü: koyu sahne, siyah konturlu beyaz yazı (BGRA)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default(30)
    img = Image.new("RGB", (1100, 90), background)
    ImageDraw.Draw(img).text((30, 25), text, font=font, fill=(255, 255, 255), stroke_width=2,
                             stroke_fill=(0, 0, 0))
    rgb = np.asarray(img)
    bgra = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    return bgra


def _check_ocr(cfg: Config, report: Report) -> None:
    from .ocr import OcrUnavailable, WindowsOcr

    try:
        ocr = WindowsOcr(cfg.ocr_language)
    except OcrUnavailable as exc:
        report.item("Windows OCR", "HATA", str(exc).splitlines()[0])
        return
    except Exception as exc:
        report.item("Windows OCR", "HATA", f"{type(exc).__name__}: {exc}")
        return
    # Koyu sahne ve açık (gökyüzü) sahne: beyaz yazı + siyah kontur her ikisinde de okunmalı.
    for scene, background in (("koyu sahne", (40, 55, 70)), ("açık sahne", (214, 226, 238))):
        gray = imageproc.to_gray(_render_subtitle(SAMPLE, background))
        mask = imageproc.text_mask(gray, cfg.preprocess, cfg.bright_threshold, cfg.dark_threshold,
                                   cfg.outline_radius, cfg.outline_white_threshold)
        bbox = imageproc.bright_bbox(mask)
        if bbox is not None:
            top, bottom, left, right = bbox
            gray, mask = gray[top:bottom, left:right], mask[top:bottom, left:right]
        img = imageproc.upscale(imageproc.for_ocr(gray, mask, cfg.preprocess), cfg.upscale)
        times, text = [], ""
        for _ in range(5):
            t0 = time.perf_counter()
            text = textproc.normalize(ocr.recognize(img))
            times.append(1000 * (time.perf_counter() - t0))
        ok = textproc.similar(text, SAMPLE, 0.9)
        report.item(f"Windows OCR ({scene})", "TAMAM" if ok else "HATA",
                    f"{statistics.median(times):.0f} ms, okunan: {text!r}")


def _check_overlay(cfg: Config, report: Report) -> None:
    import tkinter as tk

    import mss

    from .overlay import HistoryPanel, Overlay

    root = tk.Tk()
    root.withdraw()
    area = Rect(200, 200, 700, 240)
    # Arkada bembeyaz bir pencere (karlı sahne yerine).
    backdrop = tk.Toplevel(root)
    backdrop.overrideredirect(True)
    backdrop.configure(bg="#FFFFFF")
    backdrop.geometry(f"{area.width}x{area.height}+{area.left}+{area.top}")
    backdrop.wm_attributes("-topmost", True)

    test_cfg = dataclasses.replace(cfg, display_mode="cover", cover_color="#000000")
    overlay = Overlay(root, test_cfg, "cover")
    region = Rect(area.left + 50, area.top + 120, area.width - 100, 80)
    overlay.set_region(region)
    overlay.show("Arabaya bin, hemen!")
    for _ in range(10):
        root.update()
        time.sleep(0.05)
    report.item("Overlay penceresi", "TAMAM", "oluşturuldu, tıklamalar geçirgen")

    with mss.mss() as sct:
        shot = np.asarray(sct.grab({"left": region.left, "top": region.top,
                                    "width": region.width, "height": region.height}))
    gray = imageproc.to_gray(shot)
    mean = float(gray.mean())
    if not overlay.capture_excluded:
        report.item("Ekran yakalamadan gizleme", "HATA",
                    "Windows reddetti (Windows 10 2004+ gerekir); OCR kendi yazımızı görebilir")
    elif mean > 200:
        report.item("Ekran yakalamadan gizleme", "TAMAM",
                    "yakalamada overlay yok, arkadaki beyaz pencere görünüyor")
    elif mean < 5:
        report.item("Ekran yakalamadan gizleme", "BELİRSİZ",
                    "yakalama tamamen siyah (masaüstü oturumu yok mu?)")
    else:
        report.item("Ekran yakalamadan gizleme", "HATA", f"yakalamada overlay görünüyor (ort. parlaklık {mean:.0f})")

    # Arkadaki beyaz sahneye göre renk seçimi.
    stats = contrast.background_stats(shot)
    picker = contrast.ColorPicker(cfg.outline_px)
    for _ in range(3):
        picker.update(stats)
    if mean > 200:
        ok = picker.palette.fill == contrast.BLACK
        report.item("Uyarlanabilir renk (beyaz sahne)", "TAMAM" if ok else "HATA", f"seçilen yazı rengi {picker.palette.fill}")

    # Son satırlar paneli de yakalamada görünmemeli (aksi hâlde OCR onu okuyabilir).
    overlay.clear()
    panel = HistoryPanel(root, cfg)
    panel.show([{"label": "", "en": "Get in the car!", "tr": "Arabaya bin!"}], Rect(area.left - 20, area.top - 20, 0, 0))
    for _ in range(10):
        root.update()
        time.sleep(0.05)
    with mss.mss() as sct:
        shot = np.asarray(sct.grab({"left": area.left + 10, "top": area.top + 10, "width": 300, "height": 60}))
    pmean = float(imageproc.to_gray(shot).mean())
    if mean < 5:
        pass  # masaüstü yakalanamıyor; yukarıda BELİRSİZ raporlandı
    else:
        report.item("Geçmiş paneli yakalamada gizli", "TAMAM" if pmean > 200 else "HATA",
                    f"ort. parlaklık {pmean:.0f} (beyaz = panel görünmüyor)")
    root.destroy()


def _check_deepl(backend, report: Report) -> None:
    times, out = [], ""
    for i in range(4):
        text = f"{SAMPLE} ({i})"  # önbelleğe takılmasın
        t0 = time.perf_counter()
        try:
            out = backend.translate(text, None)
        except Exception as exc:
            report.item(getattr(backend, "name", "Son çeviri"), "HATA", f"{type(exc).__name__}: {exc}")
            return
        times.append(1000 * (time.perf_counter() - t0))
    report.item(getattr(backend, "name", "Son çeviri"), "TAMAM",
                f"ilk {times[0]:.0f} ms, sonraki ort. {statistics.mean(times[1:]):.0f} ms — {out!r}")
    line = "Shut the fuck up, asshole!"
    try:
        out = backend.translate(line, None)
    except Exception as exc:
        report.item(f"{getattr(backend, 'name', 'Son çeviri')} sansürsüz", "HATA", f"{type(exc).__name__}: {exc}")
        return
    active = getattr(backend, "instructions_active", False)
    report.item(f"{getattr(backend, 'name', 'Son çeviri')} sansürsüz", "TAMAM" if active else "BİLGİ",
                f"talimat {'kabul edildi' if active else 'DeepL tarafından desteklenmiyor'} — {line!r} → {out!r}")




def _check_local(local, report: Report) -> None:
    times, out = [], ""
    for line in BENCH_LINES[:6]:
        t0 = time.perf_counter()
        try:
            out = local.translate(line, None)
        except Exception as exc:
            report.item("Yerel çeviri", "HATA", f"{type(exc).__name__}: {exc}")
            return
        times.append(1000 * (time.perf_counter() - t0))
    ok = bool(out.strip())
    report.item("Yerel çeviri", "TAMAM" if ok else "HATA",
                f"{local.device}, medyan {statistics.median(times):.0f} ms — {BENCH_LINES[5]!r} → {out!r}")


def run(cfg: Config, backend=None, local=None) -> int:
    report = Report()
    print("GTA Altyazı öz test\n")
    windows = win32.visible_windows()
    report.item("Pencere listesi", "TAMAM", f"{len(windows)} görünür pencere")
    found = win32.find_window(cfg.window_titles)
    report.item("Remote Play penceresi", "TAMAM" if found else "YOK",
                found[0] if found else "açık değil (oyun sırasında açık olmalı)")
    _check_ocr(cfg, report)
    try:
        _check_overlay(cfg, report)
    except Exception as exc:
        report.item("Overlay", "HATA", f"{type(exc).__name__}: {exc}")
    if local is not None:
        _check_local(local, report)
    else:
        report.item("Yerel çeviri", "YOK", "model bulunamadı veya başlatılamadı (yukarıdaki nedene bakın)")
    if backend is not None:
        _check_deepl(backend, report)
    else:
        report.item("Son çeviri", "ATLANDI", "son çevirmen kullanılamıyor (yukarıdaki nedene bakın)")
    print("\nSonuç:", "BAŞARILI" if report.failed == 0 else f"{report.failed} HATA")
    return 1 if report.failed else 0
