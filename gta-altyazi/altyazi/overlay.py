"""Her zaman üstte, tıklanamaz, şeffaf Türkçe altyazı penceresi (tkinter) ve bölge seçici."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

from . import win32
from .config import Config
from .geometry import Rect

KEY_COLOR = "#010203"  # şeffaf yapılan renk; başka hiçbir yerde kullanılmamalı


class Overlay:
    def __init__(self, root: tk.Tk, cfg: Config):
        self.cfg = cfg
        self.win = tk.Toplevel(root)
        self.win.title("GTA Altyazi Overlay")
        self.win.overrideredirect(True)
        self.win.configure(bg=KEY_COLOR)
        self.win.wm_attributes("-topmost", True)
        self.win.wm_attributes("-transparentcolor", KEY_COLOR)
        self.canvas = tk.Canvas(self.win, bg=KEY_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.font = tkfont.Font(family=cfg.font_family, size=cfg.font_size, weight="bold")
        self.small_font = tkfont.Font(family=cfg.font_family, size=max(10, cfg.font_size * 6 // 10))
        self.region: Rect | None = None
        self._state: tuple[str, str, str] | None = None  # (tr, en, status)
        self.win.geometry("1x1+0+0")
        self.win.update_idletasks()
        win32.make_click_through(win32.user32.GetParent(self.win.winfo_id()))
        self._keep_on_top()

    def _keep_on_top(self) -> None:
        # Remote Play tam ekrana geçtiğinde overlay'i tekrar en üste al.
        self.win.wm_attributes("-topmost", True)
        self.win.lift()
        self.win.after(2000, self._keep_on_top)

    def set_region(self, region: Rect) -> None:
        if region != self.region:
            self.region = region
            self._redraw()

    def show(self, translated: str, original: str = "") -> None:
        self._state = (translated, original, "")
        self._redraw()

    def status(self, message: str) -> None:
        self._state = ("", "", message)
        self._redraw()

    def clear(self) -> None:
        self._state = None
        self._redraw()

    def hide(self) -> None:
        self.win.withdraw()

    def unhide(self) -> None:
        self.win.deiconify()
        self.win.update_idletasks()
        win32.make_click_through(win32.user32.GetParent(self.win.winfo_id()))
        self._redraw()

    def _text(self, x: int, y: int, text: str, font, width: int, fill: str) -> int:
        """Konturlu yazı çizer, yüksekliğini döndürür."""
        o = self.cfg.outline_px
        for dx in range(-o, o + 1):
            for dy in range(-o, o + 1):
                if dx or dy:
                    self.canvas.create_text(
                        x + dx, y + dy, text=text, font=font, width=width,
                        fill=self.cfg.outline_color, anchor="n", justify="center",
                    )
        item = self.canvas.create_text(
            x, y, text=text, font=font, width=width, fill=fill, anchor="n", justify="center"
        )
        x0, y0, x1, y1 = self.canvas.bbox(item)
        return y1 - y0

    def _redraw(self) -> None:
        self.canvas.delete("all")
        region = self.region
        if region is None or self._state is None:
            self.win.geometry(f"1x1+{-10}+{-10}")
            return
        translated, original, status = self._state
        pad = self.cfg.cover_padding_px
        width = region.width + 2 * pad
        wrap = max(50, width - 2 * pad - 2 * self.cfg.outline_px)
        cx = width // 2

        # Önce yüksekliği ölçmek için çiz, sonra pencereyi ona göre boyutlandır.
        y = pad
        if status:
            y += self._text(cx, y, status, self.small_font, wrap, "#FFD166")
        if self.cfg.show_original and original:
            y += self._text(cx, y, original, self.small_font, wrap, "#BBBBBB") + 2
        if translated:
            y += self._text(cx, y, translated, self.font, wrap, self.cfg.text_color)
        height = max(y + pad, region.height + 2 * pad)

        # Yazıyı kutunun dikey ortasına taşı.
        content_h = y - pad
        offset = (height - 2 * pad - content_h) // 2
        if offset > 0:
            self.canvas.move("all", 0, offset)
        if self.cfg.cover_original:
            rect = self.canvas.create_rectangle(
                0, 0, width, height, fill=self.cfg.cover_color, outline=""
            )
            self.canvas.tag_lower(rect)

        # Alt kenar altyazı bölgesinin alt kenarına hizalı; uzun çeviri yukarı doğru büyür.
        left = region.left - pad
        top = region.bottom + pad - height
        self.canvas.configure(width=width, height=height)
        self.win.geometry(f"{width}x{height}+{left}+{top}")


class RegionPicker:
    """Ekranın üstünde yarı saydam katman: fareyle sürükleyerek altyazı bölgesini seç."""

    def __init__(self, root: tk.Tk, on_done: Callable[[Rect | None], None]):
        self._on_done = on_done
        screen = win32.virtual_screen()
        self._origin = (screen.left, screen.top)
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{screen.width}x{screen.height}+{screen.left}+{screen.top}")
        self.win.wm_attributes("-topmost", True)
        self.win.wm_attributes("-alpha", 0.35)
        self.win.configure(bg="black", cursor="crosshair")
        self.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            screen.width // 2, 60, fill="white", font=("Segoe UI", 20, "bold"),
            text="İngilizce altyazının çıktığı alanı fareyle sürükleyerek seçin  •  İptal: Esc",
        )
        self._start: tuple[int, int] | None = None
        self._rect_id: int | None = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.win.bind("<Escape>", lambda _e: self._finish(None))
        self.win.focus_force()

    def _press(self, e) -> None:
        self._start = (e.x, e.y)
        self._rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#FF3B30", width=3)

    def _drag(self, e) -> None:
        if self._start and self._rect_id:
            self.canvas.coords(self._rect_id, *self._start, e.x, e.y)

    def _release(self, e) -> None:
        if not self._start:
            return
        x0, y0 = self._start
        left, top = min(x0, e.x), min(y0, e.y)
        w, h = abs(e.x - x0), abs(e.y - y0)
        if w < 20 or h < 10:
            self._finish(None)
            return
        self._finish(Rect(left + self._origin[0], top + self._origin[1], w, h))

    def _finish(self, rect: Rect | None) -> None:
        self.win.destroy()
        self._on_done(rect)
