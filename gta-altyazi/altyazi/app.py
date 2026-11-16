"""Uygulama: yakalama → OCR → kararlılık → DeepL → overlay (+ anlık yazı rengi)."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import config as config_mod
from . import contrast, imageproc, reader, textproc, win32
from .config import EXTRA_REGION_KEYS, EXTRA_REGION_LABELS, Config, Region
from .geometry import Rect, intersect, region_to_screen, screen_to_region, union
from . import ocr as ocr_mod
from .ocr import OcrUnavailable, WindowsOcr
from .hybrid import DRAFT, FINAL, HybridTranslator

# Görüntü aynı kalsa bile bu kadar karede bir OCR tekrarlanır; altyazı belirirken (fade-in)
# yapılmış eksik bir okumada takılı kalmamak için.
MAX_REUSE = 4
# Taslak çeviri sürerken OCR en fazla bu kadar bekler (sonraki altyazı en az ~1 sn ekranda kalır).
DRAFT_YIELD_S = 0.4
# Bu kadar sn diyalog yoksa yeni sahne: sohbet geçmişi sıfırlanır.
DIALOGUE_GAP_S = 45.0
# Yazı kutusu haritasında bu oranda hücre değişirse satır değişmiş sayılır (sıkıştırma gürültüsü ~%1).
BOX_TOLERANCE = 0.03


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
    if name == "OllamaUnavailable":
        return str(exc)
    if name in ("ConnectionError", "ConnectTimeout"):
        return "Yerel LLM'e (Ollama) bağlanılamadı — Ollama açık mı?"
    if name in ("ReadTimeout", "Timeout"):
        return "Yerel LLM zamanında yanıt vermedi (ollama_timeout_s artırılabilir)."
    return f"Çeviri hatası: {exc}"


MAIN = "altyazi"


class Pipeline(threading.Thread):
    """Bir ekran bölgesini okur ve çevirir. Ana altyazı ve her ek bölge (görev, telefon) için bir tane."""

    def __init__(self, cfg: Config, name: str, get_region: Callable[[], Region | None], display_mode: str,
                 translator: HybridTranslator, ui: queue.Queue, debug: bool,
                 sample_rect: Callable[[], Rect | None] = lambda: None,
                 own_rects: Callable[[], list[Rect]] = lambda: []):
        super().__init__(daemon=True, name=f"pipeline-{name}")
        self.cfg = cfg
        self.region_name = name  # (Thread.name ile karışmasın)
        self.get_region = get_region
        self.display_mode = display_mode
        self.translator = translator
        self.ui = ui
        self.debug = debug
        self.sample_rect = sample_rect
        # Yakalamada görünen kendi overlay'lerimiz: okuma bölgesine taşarsa OCR'dan önce karartılır.
        self.own_rects = own_rects
        self.paused = threading.Event()
        self.stopped = threading.Event()
        # Arayüz duraklat/devam ettiğinde: ekrandaki altyazı yeniden okunup gösterilsin.
        self.reset = threading.Event()
        # Çeviri başarısız oldu: aynı satır tekrar okununca yeniden dene.
        self._retry = threading.Event()
        self.seq = 0
        self.base = win32.primary_screen()
        self._previous_source: str | None = None
        self._dialogue: list[tuple[str, str, float]] = []  # (İngilizce, Türkçe, zaman)
        self._retry_not_before = 0.0
        self._tag = "" if name == MAIN else f"[{EXTRA_REGION_LABELS.get(name, name)}] "
        self.dump_dir: Path | None = None
        self._dumped = 0
        self._center_x: int | None = None
        from .perf import STATIC

        self.load = STATIC  # yük izleyici (app bağlar): ekran yakalama aralığı, OCR arası en az süre
        # Menü için: son kare okuma zamanı ve OCR motorunun adı.
        self.last_read_at = 0.0
        self.engine_name = ""
        self.last_visible_at = 0.0  # ana altyazı en son ne zaman ekrandaydı (sokak dinleme için)

    def _on_show(self, event: textproc.Show) -> None:
        self.seq += 1
        seq = self.seq
        context = self._dialogue_context(event.first_seen)
        self._previous_source = event.text
        drafts: dict[str, str] = {}
        center_x = self._center_x  # İngilizce satırın ekrandaki yatay ortası (bilinmiyorsa None)
        log(f"{self._tag}EN: {event.text}")

        def on_result(stage: str, translated: str, took: float) -> None:
            if stage == FINAL:
                self._remember_line(event.text, translated)
            else:
                drafts["tr"] = translated
            if self.debug:
                log(f"{self._tag}{'taslak' if stage == DRAFT else 'çeviri'} {1000 * took:.0f} ms: {translated}")
            self.ui.put(("show", self.region_name, seq, stage, translated, event.text, event.first_seen,
                         center_x))

        def on_error(exc: Exception, stage: str, draft_shown: bool) -> None:
            if stage == DRAFT:
                log(f"Yerel çeviri hatası: {exc}")
                return
            message = describe_error(exc)
            log(self._tag + message + (" (yerel çeviri ekranda kaldı)" if draft_shown else ""))
            if draft_shown:
                # Taslak bu satırın kesin metni: aynı satır yeniden çevrilmez (yazı ve dublaj sonradan
                # farklı bir çeviriyle değişmesin).
                if "tr" in drafts:
                    self._remember_line(event.text, drafts["tr"])
                self.ui.put(("draft-final", self.region_name, seq))
                return
            self.ui.put(("error", self.region_name, message))
            if seq == self.seq:
                self._retry_not_before = time.monotonic() + 2.0  # çevrimdışıyken istek yağdırma
                self._retry.set()

        self.translator.submit(event.text, context, lambda: seq == self.seq, on_result, on_error)

    def _remember_line(self, english: str, turkish: str) -> None:
        """Sohbet geçmişi: kesinleşen satırlar (çeviri tutarlılığı için sonraki satırlara bağlam)."""
        if self._dialogue and self._dialogue[-1][0] == english:
            self._dialogue[-1] = (english, turkish, time.monotonic())
        else:
            self._dialogue.append((english, turkish, time.monotonic()))
        del self._dialogue[:-max(1, self.cfg.dialogue_lines)]

    def _dialogue_context(self, now: float) -> textproc.Dialogue | None:
        """Son satırlar; DIALOGUE_GAP_S'den uzun sessizlik yeni sahne sayılır, geçmiş sıfırlanır."""
        if self._dialogue and now - self._dialogue[-1][2] > DIALOGUE_GAP_S:
            self._dialogue.clear()
        if not self.cfg.context_translation or not self._dialogue:
            return textproc.Dialogue(self._previous_source) if self._previous_source else None
        history = [(en, tr) for en, tr, _ in self._dialogue]
        return textproc.Dialogue(self._previous_source or history[-1][0], history)

    def _read_text(self, ocr, gray: np.ndarray) -> reader.ReadResult:
        result = reader.read_text(gray, ocr.recognize, self.cfg, ocr.raw_input, ocr)
        if self.debug and len(result.candidates) > 1:
            log(f"{self._tag}seçenekler: " + " | ".join(f"{n} {q:.0f} {t!r}" for n, t, q in result.candidates)
                + f" → {result.variant}")
        if self.dump_dir is not None and result.crop is not None and self._dumped < 300:
            self._dump(result)
        return result

    def _dump(self, result: reader.ReadResult) -> None:
        """--dump: OCR'a giren gerçek görüntüleri ve sonuçları kaydeder (ayar yapmak için)."""
        from PIL import Image

        self._dumped += 1
        stem = self.dump_dir / f"{self.region_name}_{self._dumped:03d}"
        Image.fromarray(result.crop).save(stem.with_suffix(".png"))
        lines = [f"{n}\t{q:.0f}\t{t}" for n, t, q in result.candidates] + [f"seçilen\t{result.variant}\t{result.text}"]
        stem.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        import mss

        # Seçilmemiş ek bölge (görev/telefon) için OCR motoru kurulmaz: açılış hızlanır, ekran
        # kartı belleği boşa gitmez. Bölge seçilince kurulur.
        while self.get_region() is None and not self.stopped.is_set():
            time.sleep(0.2)
        if self.stopped.is_set():
            return
        try:
            ocr = ocr_mod.create(self.cfg)
        except OcrUnavailable as exc:
            self.ui.put(("fatal", str(exc)))
            return
        except Exception as exc:  # pywinrt / onnxruntime kurulu değil vb.
            self.ui.put(("fatal", f"OCR başlatılamadı: {exc}"))
            return
        self.engine_name = ocr.name
        if self.region_name == MAIN:
            log(f"OCR motoru: {ocr.name}")
        textproc.normalize(["warm up"])  # sözlük ilk satırda değil şimdi yüklensin (~0,1 sn)

        cfg = self.cfg
        self._center_x = None
        ocr_on_gpu = bool(getattr(ocr, "on_gpu", False))
        busy_since: float | None = None
        stab = textproc.Stabilizer(cfg.stable_reads, cfg.similarity, cfg.clear_after_ms / 1000,
                                   cfg.confirm_hold_ms / 1000)
        picker = contrast.ColorPicker(cfg.outline_px) if self.display_mode != "cover" else None
        last_sig: np.ndarray | None = None
        last_text = ""
        reused = 0
        last_box: tuple[int, int, int, int] | None = None
        last_ocr_at = 0.0
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
                    last_sig, last_text, reused, last_box = None, "", 0, None
                if self._retry.is_set() and tick >= self._retry_not_before:
                    self._retry.clear()
                    stab.forget_shown()
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue
                # Yerel taslak çeviri işlemcide çalışırken OCR da işlemcideyse onu yavaşlatıyordu
                # (2 kelime 380 ms). Çeviri bitene kadar (en fazla DRAFT_YIELD_S) okuma yapma.
                if not ocr_on_gpu and self.translator.draft_running():
                    if busy_since is None:
                        busy_since = tick
                    if tick - busy_since < DRAFT_YIELD_S:
                        time.sleep(0.005)
                        continue
                else:
                    busy_since = None

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
                    last_sig, last_box = None, None
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
                self.last_read_at = tick
                sub = crop(region)
                if sub is None:
                    time.sleep(interval)
                    continue
                gray = imageproc.to_gray(sub)
                origin = intersect(region, zone)  # `sub` ekran kenarında kırpılmış olabilir
                for own in self.own_rects():
                    hit = intersect(own, origin)
                    if hit is not None:  # kendi Türkçe yazımızı okuma (yayında görünür modda)
                        gray[hit.top - origin.top:hit.bottom - origin.top,
                             hit.left - origin.left:hit.right - origin.left] = 0
                # Değişim tespiti: yazı varken yalnızca yazının kutusuna bakılır (arkadaki video oynasa
                # da altyazı aynıysa OCR çalışmaz; ekran kartı dil modeline kalır). Yazı yokken tüm bölge.
                if last_box is not None:
                    sig = imageproc.box_signature(gray, last_box, cfg.bright_threshold)
                    unchanged = (sig is not None and imageproc.same_signature(last_sig, sig, BOX_TOLERANCE)
                                 and tick - last_ocr_at < cfg.reread_max_ms / 1000)
                else:
                    sig = imageproc.signature(gray >= cfg.bright_threshold)
                    unchanged = imageproc.same_signature(last_sig, sig) and reused < MAX_REUSE
                # Bilgisayar yüklüyken (yayın vb.) iki OCR arasında en az bu kadar beklenir.
                if not unchanged and last_sig is not None and tick - last_ocr_at < self.load.min_ocr_gap_s:
                    unchanged = True
                if unchanged:
                    text = last_text
                    if text:
                        self.last_visible_at = tick
                    confident = False  # yeni okuma değil: hızlı kabul sayılmaz
                    reused += 1
                else:
                    reused = 0
                    last_ocr_at = tick
                    t0 = time.perf_counter()
                    result = self._read_text(ocr, gray)
                    text = result.text
                    if text:
                        self.last_visible_at = tick
                    confident = (bool(text) and (result.confidence or 0.0) >= cfg.fast_accept_confidence
                                 and textproc.all_words_known(text))
                    if result.center_x is not None and text:
                        self._center_x = round(origin.left + result.center_x)
                    if self.debug and text != last_text:
                        conf = "" if result.confidence is None else f" (güven {result.confidence:.2f})"
                        log(f"{self._tag}OCR {1000 * (time.perf_counter() - t0):.0f} ms{conf}: {text!r}")
                    last_text = text
                    last_box = result.text_box if cfg.change_detection else None
                    last_sig = (imageproc.box_signature(gray, last_box, cfg.bright_threshold)
                                if last_box is not None else imageproc.signature(gray >= cfg.bright_threshold))
                    if last_sig is None:
                        last_box = None
                        last_sig = imageproc.signature(gray >= cfg.bright_threshold)

                event = stab.feed(text, tick, confident)
                if isinstance(event, textproc.Show):
                    self._on_show(event)
                elif isinstance(event, textproc.Clear):
                    self.seq += 1
                    self.ui.put(("clear", self.region_name))

                # Doğrulanmayı bekleyen satır varsa hemen tekrar oku; yoksa normal kare aralığı.
                interval = self.load.capture_interval_s
                wait = confirm_delay if stab.pending else interval
                elapsed = time.monotonic() - tick
                if elapsed < wait:
                    time.sleep(wait - elapsed)

        self.translator.shutdown()


# Dublaj açıkken taslak çeviri, son çeviri için en fazla bu kadar bekletilir (yazı = ses).
DUB_DRAFT_HOLD_S = 1.5

# "Hazır konum" düğmesi: GTA V düzenine göre tahmini bölgeler (hedef pencereye göre oran).
PRESETS = {
    MAIN: Region(0.15, 0.80, 0.70, 0.15),
    "gorev": Region(0.25, 0.88, 0.50, 0.08),
    "telefon": Region(0.70, 0.45, 0.27, 0.45),
}
REGION_ORDER = [MAIN, *EXTRA_REGION_KEYS.values()]
STREET = "sokak"
REGION_LABELS = {MAIN: "Altyazı", **EXTRA_REGION_LABELS, STREET: "Sokak"}


def run_gui(cfg: Config, cfg_path: Path, make_translator: Callable[[], HybridTranslator], debug: bool,
            dump_dir: Path | None = None, engine_info: list[tuple[str, str]] | None = None,
            instance=None) -> bool:
    """Overlay'ler, okuma iş parçacıkları, kısayollar ve menü. Dönüş: yeniden başlatılsın mı."""
    import os
    import tkinter as tk
    from tkinter import messagebox

    from . import hotkeys as hk
    from .menu import ControlPanel
    from .overlay import REMOVE, HistoryPanel, Overlay, RegionPicker
    from .sessionlog import SessionLog

    root = tk.Tk()
    root.withdraw()
    ui: queue.Queue = queue.Queue()
    # Program ikinci kez başlatılırsa yeni kopya açılmaz; bu kopyanın menüsü öne gelir.
    win32.watch_show_requests(instance, lambda: ui.put(("action", "menu-show")))
    # Ayrıntılı kayıt (süreler): --debug ya da menüdeki ayar; menüden program durmadan açılıp kapanır.
    detail = {"on": bool(debug or cfg.detailed_log)}

    overlays: dict[str, Overlay] = {MAIN: Overlay(root, cfg, cfg.display_mode)}
    pipelines: dict[str, Pipeline] = {}
    main_overlay = overlays[MAIN]
    if main_overlay.capture_visible:
        log("Türkçe altyazı yayında (OBS) ve ekran görüntülerinde görünür.")
    elif not main_overlay.capture_excluded:
        log("UYARI: Overlay ekran yakalamadan gizlenemedi (Windows 10 2004 veya üstü gerekir). "
            "OCR kendi Türkçe yazımızı/kutumuzu okuyabilir.")
    else:
        log("Türkçe altyazı yayında/ekran görüntüsünde görünmez ("
            + ("\"cover\" modu okunan yazının üstünü örttüğü için" if cfg.display_mode == "cover"
               else "capture_visible: false") + ").")

    panel = None  # menü (aşağıda kurulur); own_rects okur

    def own_rects() -> list[Rect]:
        """Ekran yakalamada görünen kendi pencerelerimiz: okuma bölgesine taşarsa OCR'dan önce karartılır."""
        rects = [ov.screen_rect for ov in overlays.values() if ov.capture_visible and ov.screen_rect]
        menu_rect = panel.screen_rect if panel is not None else None
        if menu_rect is not None:
            rects.append(menu_rect)
        return rects

    pipelines[MAIN] = Pipeline(cfg, MAIN, lambda: cfg.region if cfg.region_enabled else None, cfg.display_mode,
                               make_translator(), ui, detail["on"],
                               sample_rect=lambda: main_overlay.sample_rect, own_rects=own_rects)
    for name in EXTRA_REGION_KEYS.values():
        def get_region(name=name):
            extra = cfg.extra(name)
            return extra.region if extra and extra.enabled else None
        mode = cfg.extra(name).display_mode if cfg.extra(name) else "cover"
        overlays[name] = Overlay(root, cfg, mode)
        pipelines[name] = Pipeline(cfg, name, get_region, mode, make_translator(), ui, detail["on"],
                                   sample_rect=lambda ov=overlays[name]: ov.sample_rect, own_rects=own_rects)
    from .street import StreetListener

    overlays[STREET] = Overlay(root, cfg, "adaptive")
    pipelines[STREET] = StreetListener(cfg, STREET, lambda: tts["recorder"], make_translator(), ui,
                                       lambda a, b: street_blocked(a, b), log)
    from .perf import LoadMonitor

    load = LoadMonitor(cfg, log)
    for p in pipelines.values():
        p.load = load
    history = HistoryPanel(root, cfg)
    entries: list[dict] = []  # son satırlar: {"key": (bölge, seq), "label", "en", "tr"}
    state = {"welcome": True, "picking": False, "paused": False, "restart": False, "save_pending": False}
    final_seq = {name: -1 for name in pipelines}
    last_line = {"en": "", "tr": "", "info": ""}
    session = None
    if cfg.session_log:
        try:
            folder = Path(cfg.session_log_dir)
            session = SessionLog(folder if folder.is_absolute() else cfg_path.parent / folder)
            log(f"Oturum kaydı: {session.path}")
        except OSError as exc:
            log(f"Oturum kaydı açılamadı: {exc}")
    tts = {"speaker": None, "recorder": None}
    from .voiceid import CharacterBook

    char_path = Path(cfg.characters_file)
    book = CharacterBook(char_path if char_path.is_absolute() else cfg_path.parent / char_path)
    line_started: dict[str, int] = {}  # bölge -> dublajı kesilen son satır (seq)

    def all_pipelines():
        return pipelines.values()

    # --- ayarlar -------------------------------------------------------------------------------

    def save_soon() -> None:
        """Art arda değişiklikleri (kısayolla büyütme vb.) tek yazmada topla."""
        if not state["save_pending"]:
            state["save_pending"] = True

            def flush():
                state["save_pending"] = False
                config_mod.save(cfg, cfg_path)
            root.after(400, flush)

    def apply_style() -> None:
        for ov in overlays.values():
            ov.apply_style()
        history.refresh(entries, pipelines[MAIN].base)
        if panel is not None:
            panel.refresh_style_vars()

    # --- kısayollar ----------------------------------------------------------------------------

    hk_state: dict = {"thread": None}

    def apply_hotkeys() -> None:
        old = hk_state["thread"]
        if old is not None:
            old.stop()
            old.join(timeout=1)
        bindings = []
        for action, combo in cfg.hotkeys.items():
            if not combo:
                continue
            try:
                mods, vk = hk.parse(combo)
            except hk.HotkeyError:
                continue
            bindings.append((action, mods, vk, lambda a=action: ui.put(("action", a))))
        thread = win32.HotkeyThread(bindings)
        thread.start()
        thread.ready.wait(1.0)
        hk_state["thread"] = thread
        failed = [hk.display(cfg.hotkeys[a]) for a in thread.failed]
        if failed:
            log("Bu kısayollar başka bir program tarafından kullanılıyor: " + ", ".join(failed)
                + " (menüden değiştirebilirsiniz)")

    def hotkey_rows() -> list[tuple[str, str, str, str]]:
        status = hk_state["thread"].status if hk_state["thread"] else {}
        clashes = hk.conflicts(cfg.hotkeys)
        rows = []
        for action, label, _ in hk.ACTIONS:
            combo = cfg.hotkeys.get(action, "")
            if not combo:
                text = "— atanmamış"
            elif combo in clashes:
                text = "✖ başka bir işlemle aynı"
            elif status.get(action):
                text = "✔ çalışıyor"
            else:
                text = "✖ başka program kullanıyor"
            rows.append((action, label, combo, text))
        return rows

    def set_hotkey(action: str, combo: str) -> str | None:
        if combo:
            try:
                combo = hk.normalize(combo)
            except hk.HotkeyError as exc:
                return f"Geçersiz kısayol: {exc}"
            other = next((a for a, c in cfg.hotkeys.items() if c == combo and a != action), None)
            if other:
                return f"{hk.display(combo)} zaten \"{hk.ACTION_LABELS[other]}\" işlemine atanmış."
        cfg.hotkeys[action] = combo
        save_soon()
        apply_hotkeys()
        if combo and not hk_state["thread"].status.get(action, False):
            return f"{hk.display(combo)} kaydedildi ama başka bir program kullanıyor; başka bir tuş seçin."
        return None

    def free_hotkeys() -> list[str]:
        ours = set(cfg.hotkeys.values())
        free = []
        for combo in hk.candidates():
            if combo in ours:
                continue
            try:
                if win32.hotkey_free(*hk.parse(combo)):
                    free.append(combo)
            except Exception:
                pass
        return free

    # --- işlemler ------------------------------------------------------------------------------

    def quit_app(restart: bool = False) -> None:
        state["restart"] = restart
        for p in all_pipelines():
            p.stopped.set()
        if hk_state["thread"] is not None:
            hk_state["thread"].stop()
        if tts["speaker"] is not None:
            tts["speaker"].stop()
        if tts["recorder"] is not None:
            tts["recorder"].stop()
        load.stop()
        if state["save_pending"]:
            config_mod.save(cfg, cfg_path)
        if session is not None:
            session.close()
        root.destroy()

    def restart_app() -> None:
        config_mod.save(cfg, cfg_path)
        log("Yeniden başlatılıyor...")
        quit_app(restart=True)

    def toggle_pause() -> None:
        if state["picking"]:  # seçici açıkken duraklatma, seçim bitince durumu bozardı
            return
        state["paused"] = not state["paused"]
        if state["paused"] and tts["speaker"] is not None:
            tts["speaker"].interrupt()  # duraklatınca dublaj da sussun
        for p in all_pipelines():
            p.reset.set()
            (p.paused.set if state["paused"] else p.paused.clear)()
        for ov in overlays.values():
            if state["paused"]:
                ov.clear()
                ov.hide()
            else:
                ov.unhide()
        log("Duraklatıldı" if state["paused"] else "Devam ediliyor")

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
        menu_was_visible = panel is not None and panel.visible
        if panel is not None:
            panel.hide()
        label = "İngilizce altyazının" if name == MAIN else f"{EXTRA_REGION_LABELS[name]} yazısının"

        def done(result) -> None:
            state["picking"] = False
            base = pipelines[MAIN].base
            if name == MAIN and isinstance(result, Rect):
                cfg.region = screen_to_region(result, base)
                cfg.region_enabled = True
                log(f"Altyazı bölgesi kaydedildi: {cfg.region}")
            elif isinstance(result, Rect):
                cfg.set_extra(name, screen_to_region(result, base))
                cfg.extra(name).enabled = True
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
            if menu_was_visible:
                panel.show()

        RegionPicker(root, done, f"{label} çıktığı alanı fareyle seçin", allow_remove=name != MAIN)

    def set_region_enabled(name: str, enabled: bool) -> None:
        if name == MAIN:
            cfg.region_enabled = enabled
        elif cfg.extra(name) is not None:
            cfg.extra(name).enabled = enabled
        elif enabled:  # seçilmemiş bölgeyi açmak: önce bölge seçilmeli
            pick(name)
            return
        if not enabled:
            overlays[name].clear()
        log(f"{REGION_LABELS[name]} okuma {'açık' if enabled else 'kapalı'}")
        save_soon()

    def toggle_region(name: str) -> None:
        extra = cfg.extra(name)
        set_region_enabled(name, not (extra.enabled if extra else False))

    def remove_region(name: str) -> None:
        if name != MAIN and cfg.extra(name) is not None:
            cfg.set_extra(name, None)
            overlays[name].clear()
            log(f"{REGION_LABELS[name]} bölgesi kaldırıldı")
            save_soon()

    def preset_region(name: str) -> None:
        preset = PRESETS[name]
        if name == MAIN:
            cfg.region = Region(**vars(preset))
            cfg.region_enabled = True
        else:
            cfg.set_extra(name, Region(**vars(preset)))
            cfg.extra(name).enabled = True
        log(f"{REGION_LABELS[name]} bölgesi hazır konuma alındı")
        save_soon()

    def toggle_original() -> None:
        cfg.show_original = not cfg.show_original
        log(f"İngilizce satır gösterimi: {'açık' if cfg.show_original else 'kapalı'}")
        save_soon()
        apply_style()

    def toggle_history() -> None:
        history.toggle(entries, pipelines[MAIN].base)

    def change_font(step: int) -> None:
        cfg.font_size = max(12, min(72, cfg.font_size + step))
        save_soon()
        apply_style()

    def move_text(step: int) -> None:
        cfg.overlay_gap_px = max(-300, min(600, cfg.overlay_gap_px + step))
        save_soon()
        apply_style()

    def recorder_needed() -> bool:
        return (tts["speaker"] is not None and cfg.dub_detect_speaker) or cfg.street_enabled

    def ensure_recorder():
        """Oyun sesi kaydedicisi: dublaj (konuşan tanıma) ve sokak konuşmaları ortak kullanır."""
        from .audio import LoopbackRecorder

        if tts["recorder"] is None:
            recorder = LoopbackRecorder(log=log)
            recorder.start()
            recorder.ready.wait(3)
            if recorder.error:
                log(recorder.error)
            tts["recorder"] = recorder
        return tts["recorder"]

    def release_recorder() -> None:
        if tts["recorder"] is not None and not recorder_needed():
            tts["recorder"].stop()
            tts["recorder"] = None

    def set_tts(enabled: bool) -> str | None:
        """Dublajı program durmadan aç/kapat. Hata varsa açıklaması döner (dublaj kapalı kalır)."""
        from .tts import Speaker

        speaker = tts["speaker"]
        if enabled and speaker is None:
            recorder = ensure_recorder() if cfg.dub_detect_speaker else None
            speaker = Speaker(cfg, book, recorder, log)
            speaker.debug = detail["on"]  # ayrıntılı kayıt: her satırın dublaj gecikmesi loga
            speaker.start()
            speaker.ready.wait(5)
            if speaker.error:
                cfg.tts_enabled = False
                save_soon()
                release_recorder()
                log(speaker.error)
                return speaker.error
            tts["speaker"] = speaker
            log(f"Dublaj açık: {speaker.voice_in_use}"
                + (", konuşan sesinden tanınıyor" if cfg.dub_detect_speaker else ""))
        elif not enabled:
            if speaker is not None:
                speaker.stop()
                tts["speaker"] = None
            release_recorder()
            log("Dublaj kapalı")
        cfg.tts_enabled = enabled
        save_soon()
        return None

    def set_street(enabled: bool) -> str | None:
        """Sokak konuşmalarını (altyazısız NPC'ler) program durmadan aç/kapat."""
        import importlib.util

        if enabled and importlib.util.find_spec("faster_whisper") is None:
            cfg.street_enabled = False
            return "Konuşma tanıma (faster-whisper) kurulu değil: guncelle.bat çalıştırın."
        if enabled:
            recorder = ensure_recorder()
            if recorder.error:
                cfg.street_enabled = False
                release_recorder()
                return recorder.error
            base = pipelines[MAIN].base
            overlays[STREET].set_region(Rect(base.left + base.width // 5, base.top + int(base.height * 0.12),
                                             base.width * 3 // 5, 1))
        cfg.street_enabled = enabled
        if not enabled:
            overlays[STREET].clear()
            release_recorder()
        log(f"Sokak konuşmaları {'açık' if enabled else 'kapalı'}")
        save_soon()
        return None

    def street_blocked(start: float, end: float) -> bool:
        """Parça sırasında ekranda altyazı vardı (zaten çevriliyor) ya da kendi dublajımız çalıyordu."""
        if pipelines[MAIN].last_visible_at >= start - 0.3:
            return True
        speaker = tts["speaker"]
        return speaker is not None and speaker.busy_until > start

    def character_sample(char_id: int) -> str | None:
        speaker = tts["speaker"]
        if speaker is None:
            return "Örnek için önce dublajı açın."
        ch = next((c for c in book.snapshot() if c.id == char_id), None)
        if ch is not None:
            speaker.sample(ch.gender, ch.pitch_pct, ch.rate_pct, ch.voice)
        return None

    def dub_status() -> list[tuple[str, str]]:
        speaker, recorder = tts["speaker"], tts["recorder"]
        if speaker is None:
            return [("Durum", "kapalı")]
        rows = [("Ses", speaker.voice_in_use)]
        if speaker.last_engine:
            rows.append(("Son satırı okuyan", speaker.last_engine
                         + (f"  ·  Edge'e geçiş: {speaker.fallbacks}" if speaker.fallbacks else "")))
        if cfg.dub_detect_speaker:
            rows.append(("Oyun sesi", recorder.error if recorder and recorder.error else
                         f"dinleniyor ({recorder.device})" if recorder else "kapalı"))
            rows.append(("Son konuşan", speaker.last_character or "henüz yok"))
        return rows

    def desktop_shortcut() -> str:
        """Masaüstüne "GTA Altyazı" kısayolu (baslat.bat). OneDrive masaüstü dahil."""
        import subprocess

        folder = cfg_path.parent.resolve()
        script = (
            "$d=[Environment]::GetFolderPath('Desktop');"
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'GTA Altyazı.lnk'));"
            f"$s.TargetPath='{folder / 'baslat.bat'}';$s.WorkingDirectory='{folder}';"
            "$s.Description='GTA Türkçe altyazı';$s.Save()"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           check=True, capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Kısayol oluşturulamadı: {exc}"
        return "Masaüstüne \"GTA Altyazı\" kısayolu eklendi."

    def set_detailed(on: bool) -> None:
        detail["on"] = on
        cfg.detailed_log = on
        for pipeline in pipelines.values():
            pipeline.debug = on
        if tts["speaker"] is not None:
            tts["speaker"].debug = on
        save_soon()
        log(f"Ayrıntılı kayıt {'açık' if on else 'kapalı'}")

    def save_log() -> str:
        from .logbuf import BUFFER

        folder = Path(cfg.session_log_dir)
        folder = folder if folder.is_absolute() else cfg_path.parent / folder
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"sorun_kaydi_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            path.write_text(BUFFER.text() + "\n", encoding="utf-8")
        except OSError as exc:
            return f"Kaydedilemedi: {exc}"
        return f"Kaydedildi: {path}"

    def open_folder(which: str) -> None:
        if which == "log":
            folder = Path(cfg.session_log_dir)
            target = folder if folder.is_absolute() else cfg_path.parent / folder
            target.mkdir(parents=True, exist_ok=True)
        else:
            target = Path(cfg.glossary_file)
            target = target if target.is_absolute() else cfg_path.parent / target
        try:
            os.startfile(target)  # type: ignore[attr-defined]  # yalnızca Windows
        except (AttributeError, OSError) as exc:
            log(f"Açılamadı: {target} ({exc})")

    def remember(name: str, seq: int, original: str, translated: str) -> None:
        key = (name, seq)
        for entry in entries:
            if entry["key"] == key:  # taslağın yerine son çeviri geldi
                entry["tr"] = translated
                break
        else:
            entries.append({"key": key, "label": "" if name == MAIN else EXTRA_REGION_LABELS.get(name, name),
                            "en": original, "tr": translated})
            del entries[:-max(1, cfg.history_size)]
        history.refresh(entries, pipelines[MAIN].base)

    actions = {
        "menu": lambda: panel.toggle() if panel is not None else None,
        "menu-show": lambda: panel.show() if panel is not None else None,
        "pause": toggle_pause, "pick": lambda: pick(MAIN), "original": toggle_original,
        "history": toggle_history, "quit": quit_app,
        "font-up": lambda: change_font(2), "font-down": lambda: change_font(-2),
        "move-up": lambda: move_text(8), "move-down": lambda: move_text(-8),
        "tts": lambda: set_tts(not cfg.tts_enabled),
        "street": lambda: set_street(not cfg.street_enabled),
    }
    for name in EXTRA_REGION_KEYS.values():
        actions[f"pick-{name}"] = lambda name=name: pick(name)
        actions[f"toggle-{name}"] = lambda name=name: toggle_region(name)

    # --- menünün okuduğu durum ----------------------------------------------------------------

    class Controller:
        def __init__(self):
            self.cfg = cfg

        def action(self, name: str) -> None:
            actions[name]()

        def save(self) -> None:
            save_soon()

        def restart(self) -> None:
            restart_app()

        def paused(self) -> bool:
            return state["paused"]

        def region_rows(self) -> list[dict]:
            rows = []
            now = time.monotonic()
            for name in REGION_ORDER:
                p = pipelines[name]
                if name == MAIN:
                    selected, enabled = True, cfg.region_enabled
                else:
                    extra = cfg.extra(name)
                    selected, enabled = extra is not None, bool(extra and extra.enabled)
                if not selected:
                    status, color = "Bölge seçilmedi", "#7f8c8d"
                elif not enabled:
                    status, color = "Kapalı", "#7f8c8d"
                elif state["paused"] or state["picking"]:
                    status, color = "Duraklatıldı", "#d68910"
                elif now - p.last_read_at < 1.5:
                    status, color = "Çalışıyor ✔", "#1f8b4c"
                else:
                    status, color = "Başlatılıyor…", "#d68910"
                rows.append({"name": name, "label": REGION_LABELS[name], "enabled": enabled,
                             "selected": selected, "status": status, "color": color})
            return rows

        def engine_rows(self) -> list[tuple[str, str]]:
            rows = [("Yük", load.describe()),
                    ("OCR", pipelines[MAIN].engine_name or "henüz başlatılmadı")]
            rows += list(engine_info or [])
            speaker = tts["speaker"]
            rows.append(("Dublaj", f"açık ({speaker.voice_in_use})" if speaker else "kapalı"))
            street = pipelines[STREET]
            rows.append(("Sokak konuşmaları", (f"açık · {street.engine_name or 'yükleniyor'} · çevrilen {street.heard}, "
                                               f"altyazı/dublaj yüzünden atlanan {street.skipped}")
                         if cfg.street_enabled else "kapalı"))
            rows.append(("Oturum kaydı", str(session.path) if session else "kapalı"))
            return rows

        def last_line(self) -> tuple[str, str, str]:
            return last_line["en"], last_line["tr"], last_line["info"]

        def set_region_enabled(self, name: str, enabled: bool) -> None:
            set_region_enabled(name, enabled)

        def remove_region(self, name: str) -> None:
            remove_region(name)

        def preset_region(self, name: str) -> None:
            preset_region(name)

        def apply_style(self) -> None:
            apply_style()

        def hotkey_rows(self):
            return hotkey_rows()

        def set_hotkey(self, action: str, combo: str) -> str | None:
            return set_hotkey(action, combo)

        def free_hotkeys(self) -> list[str]:
            return free_hotkeys()

        def tts_voices(self) -> list[str]:
            from .tts import turkish_voices
            return turkish_voices()

        def set_tts(self, enabled: bool) -> str | None:
            return set_tts(enabled)

        def open_folder(self, which: str) -> None:
            open_folder(which)

        def log_since(self, seen: int) -> tuple[int, list[str]]:
            from .logbuf import BUFFER

            return BUFFER.since(seen)

        def log_text(self) -> str:
            from .logbuf import BUFFER

            return BUFFER.text()

        def set_detailed(self, on: bool) -> None:
            set_detailed(on)

        def save_log(self) -> str:
            return save_log()

        def desktop_shortcut(self) -> str:
            return desktop_shortcut()

        def dub_status(self) -> list[tuple[str, str]]:
            return dub_status()

        def set_street(self, enabled: bool) -> str | None:
            return set_street(enabled)

        def characters(self) -> list:
            return book.snapshot()

        def character_gender(self, char_id: int, gender: str) -> None:
            book.set_gender(char_id, gender)

        def character_remove(self, char_id: int) -> None:
            book.remove(char_id)

        def characters_clear(self) -> None:
            book.clear()

        def character_sample(self, char_id: int) -> str | None:
            return character_sample(char_id)

    panel = ControlPanel(root, Controller())
    # Menü ekran görüntüsünde ve yayında görünür (gizlenmez); bir okuma bölgesinin üstüne gelirse
    # o kısım OCR'dan önce karartılır (own_rects), menü yazıları çevrilmez.
    if not cfg.show_menu_on_start:
        panel.hide()

    held_draft: dict[str, tuple] = {}  # bölge -> dublaj için bekletilen taslak mesajı

    def dub_waits(name: str) -> bool:
        return name == MAIN and tts["speaker"] is not None and cfg.dub_wait_final

    def display(name: str, seq: int, stage: str, translated: str, original: str, first_seen: float,
                center_x) -> None:
        if name == MAIN:
            state["welcome"] = False
        overlays[name].show(f"Sokak: {translated}" if name == STREET else translated, original, center_x)
        remember(name, seq, original, translated)
        if name == STREET:  # sokak satırı birkaç saniye sonra kendiliğinden silinir
            ms = int(1000 * min(7.0, 2.5 + 0.06 * len(translated)))
            root.after(ms, lambda n=name, q=seq: overlays[n].clear() if pipelines[n].seq == q else None)
        speaker = tts["speaker"]
        if speaker is not None and name == MAIN and line_started.get(name) != seq:
            line_started[name] = seq
            speaker.prepare((name, seq), first_seen)  # konuşan tanıma çeviriyle paralel başlasın
        took = 1000 * (time.monotonic() - first_seen)
        if name == MAIN:
            last_line.update(en=original, tr=translated,
                             info=f"{'taslak' if stage == DRAFT else 'son çeviri'}, ilk okumadan ekrana {took:.0f} ms")
        if stage == FINAL:
            finalize(name, seq, translated, original, first_seen)
        if detail["on"]:
            log(f"ilk okumadan ekrana ({'taslak' if stage == DRAFT else 'son çeviri'}): {took:.0f} ms")

    def finalize(name: str, seq: int, translated: str, original: str, first_seen) -> None:
        """Satırın kesin Türkçesi: ekrandaki metnin aynısı okunur ve kaydedilir; sonra gelen hiçbir
        çeviri bu satırın yazısını ya da sesini değiştirmez."""
        final_seq[name] = seq
        if session is not None:
            session.write("" if name == MAIN else REGION_LABELS[name], original, translated)
        speaker = tts["speaker"]
        if speaker is not None and (name == MAIN or (name == STREET and cfg.street_dub)):
            speaker.say(translated, onset=first_seen, original=original, key=(name, seq))

    def release_draft(name: str, seq: int) -> bool:
        """Bekletilen taslağı kesin metin olarak göster (son çeviri gelmedi). Bekleyen yoksa False."""
        msg = held_draft.get(name)
        if msg is None or msg[2] != seq:
            return False
        del held_draft[name]
        if seq != pipelines[name].seq or final_seq[name] == seq or state["paused"] or state["picking"]:
            return True
        _, _, _, _, translated, original, first_seen, center_x = msg
        display(name, seq, FINAL, translated, original, first_seen, center_x)
        return True

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
                        menu_key = hk.display(cfg.hotkeys.get("menu", ""))
                        main_overlay.status(f"Türkçe altyazı aktif  •  {menu_key}: menü ve ayarlar")
                        root.after(5000, welcome_off)
                elif kind == "palette":
                    _, name, palette = msg
                    overlays[name].set_palette(palette)
                    if detail["on"]:
                        log(f"yazı rengi: {palette.fill} (kontur {palette.outline_px}px)")
                elif kind in ("show", "error") and (state["paused"] or state["picking"]):
                    # Duraklatma/seçim sırasında yolda olan çeviri: gizli pencereye yazılıp devamda
                    # bayat satır olarak geri gelmesin (seq artışı iş parçacığında biraz sonra olur).
                    continue
                elif kind == "show":
                    _, name, seq, stage, translated, original, first_seen, center_x = msg
                    # Eski satırı ya da kesinleşmiş satıra sonradan gelen çeviriyi (taslak ya da geç
                    # kalan son çeviri) gösterme: yazı ve dublaj o satır için değişmez.
                    if seq != pipelines[name].seq or final_seq[name] == seq:
                        continue
                    if stage == DRAFT and dub_waits(name):
                        # Dublaj açık: yazı ve ses aynı çeviriyi göstersin. Taslak, son çeviri gelene
                        # kadar (en fazla DUB_DRAFT_HOLD_S) bekletilir; gelmezse taslak kesinleşir.
                        held_draft[name] = msg
                        root.after(int(DUB_DRAFT_HOLD_S * 1000), lambda n=name, q=seq: release_draft(n, q))
                        continue
                    if stage == FINAL:
                        held_draft.pop(name, None)
                    display(name, seq, stage, translated, original, first_seen, center_x)
                elif kind == "draft-final":
                    # Son çeviri başarısız oldu, taslak bu satırın kesin metni: bekletiliyorsa hemen
                    # göster; ekrandaysa dublaj ve kayıt için kesinleştir.
                    _, name, seq = msg
                    if not release_draft(name, seq) and seq == pipelines[name].seq and final_seq[name] != seq:
                        shown = next((e for e in entries if e["key"] == (name, seq)), None)
                        if shown is not None:
                            finalize(name, seq, shown["tr"], shown["en"], None)
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
        except tk.TclError:  # pencere kapanırken
            return
        root.after(10, poll)

    apply_hotkeys()
    load.start()
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        log(f"OCR görüntüleri kaydediliyor: {dump_dir} (bölge başına en fazla 300)")
    for p in all_pipelines():
        p.dump_dir = dump_dir
        p.start()
    if cfg.tts_enabled:
        root.after(100, lambda: set_tts(True))
    if cfg.street_enabled:
        root.after(200, lambda: set_street(True) and None)
    root.after(10, poll)
    log(f"Çalışıyor. Menü ve tüm ayarlar: {hk.display(cfg.hotkeys.get('menu', ''))}")
    try:
        root.mainloop()
    finally:
        for p in all_pipelines():
            p.stopped.set()
        if session is not None:
            session.close()
    return state["restart"]


def run_ocr_compare(cfg: Config, folder: Path) -> None:
    """--dump ile kaydedilen gerçek kırpıntılarda Windows OCR'ın her ön işleme seçeneğini ve RapidOCR'ı dener.
    Doğru metin bilinmediği için puan (reading_score) ve "anlamlı satır" sayısı karşılaştırılır."""
    from PIL import Image

    images = sorted(p for p in folder.glob("*.png") if not p.stem.endswith("_ocr_girdi"))
    if not images:
        log(f"{folder} içinde .png yok. Önce --debug --dump ile çalıştırın.")
        return
    engines = []
    for make in (lambda: WindowsOcr(cfg.ocr_language), ocr_mod.RapidOcr):
        try:
            engines.append(make())
        except Exception as exc:
            log(f"OCR motoru kullanılamıyor: {exc}")
    totals: dict[str, list[float]] = {}
    for path in images:
        gray = np.asarray(Image.open(path).convert("L"))
        print(path.name)
        jobs = []
        for engine in engines:
            if engine.raw_input:
                jobs.append(("rapidocr", gray, 1.0, engine))
            else:
                jobs += [(name, image, cfg.upscale, engine) for name, image in reader.variants(gray, cfg, "all")]
        for name, image, factor, engine in jobs:
            t0 = time.perf_counter()
            text = textproc.normalize(engine.recognize(imageproc.upscale(image, factor)))
            ms = 1000 * (time.perf_counter() - t0)
            score = textproc.reading_score(text)
            ok = textproc.is_plausible(text, cfg.min_text_chars)
            t = totals.setdefault(name, [0.0, 0, 0.0])
            t[0] += score
            t[1] += ok
            t[2] += ms
            print(f"  {name:8} {score:4.0f} {'✓' if ok else ' '} {text}")
    print(f"\n{len(images)} görüntü. Seçenek / toplam puan / anlamlı satır / ort. OCR ms:")
    for name, (score, ok, ms) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:8} {score:7.0f} {int(ok):5d} {ms / len(images):6.1f}")
    print("Hangi seçeneğin hangi satırı doğru okuduğuna yukarıdaki listeden bakın.")


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
    ocr = ocr_mod.create(cfg)
    log(f"OCR motoru: {ocr.name}")
    t0 = time.perf_counter()
    result = reader.read_text(gray, ocr.recognize, cfg, ocr.raw_input)
    log(f"OCR süresi (tüm seçenekler): {1000 * (time.perf_counter() - t0):.0f} ms")
    if result.crop is not None:
        debug_path = image_path.with_name(image_path.stem + "_ocr_girdi.png")
        Image.fromarray(result.crop).save(debug_path)
        log(f"OCR'a verilen kırpılmış görüntü: {debug_path}")
    for name, cand, quality in result.candidates:
        print(f"  {name:7} {quality:4.0f}  {cand}")
    text = result.text
    print("İngilizce:", text or "(metin yok)", f"[{result.variant}]" if result.variant else "")
    for name, backend in (("Yerel", translator.draft), ("DeepL", translator.final)):
        if text and backend is not None:
            t0 = time.perf_counter()
            out = backend.translate(text, None)
            print(f"Türkçe ({name}): {out}   [{1000 * (time.perf_counter() - t0):.0f} ms]")
