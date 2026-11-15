"""Windows API yardımcıları (ctypes): DPI, pencere bulma, tıklanamaz overlay, global kısayollar."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

from .geometry import Rect

user32 = ctypes.WinDLL("user32", use_last_error=True)


def enable_dpi_awareness() -> None:
    """Ekran ölçeklendirmesi (%125, %150) varken koordinatların fiziksel piksel olması için."""
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


def primary_screen() -> Rect:
    return Rect(0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))


def virtual_screen() -> Rect:
    return Rect(
        user32.GetSystemMetrics(76),
        user32.GetSystemMetrics(77),
        user32.GetSystemMetrics(78),
        user32.GetSystemMetrics(79),
    )


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _client_rect(hwnd: int) -> Rect | None:
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width, height = rc.right - rc.left, rc.bottom - rc.top
    if width <= 0 or height <= 0:
        return None
    return Rect(origin.x, origin.y, width, height)


def visible_windows() -> list[tuple[int, str, Rect]]:
    found: list[tuple[int, str, Rect]] = []

    def callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            title = _window_title(hwnd)
            if title:
                rect = _client_rect(hwnd)
                if rect is not None:
                    found.append((hwnd, title, rect))
        return True

    user32.EnumWindows(_WNDENUMPROC(callback), 0)
    return found


def find_window(title_parts: list[str]) -> tuple[str, Rect] | None:
    """Başlığında verilen ifadelerden biri geçen en büyük görünür pencere."""
    needles = [p.casefold() for p in title_parts if p]
    best: tuple[str, Rect] | None = None
    for _hwnd, title, rect in visible_windows():
        folded = title.casefold()
        if any(n in folded for n in needles):
            if best is None or rect.width * rect.height > best[1].width * best[1].height:
                best = (title, rect)
    return best


GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008


def make_click_through(hwnd: int) -> None:
    """Fare tıklamaları overlay'den geçip alttaki pencereye (Remote Play) gider."""
    get = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_ = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    get.restype = ctypes.c_ssize_t
    set_.restype = ctypes.c_ssize_t
    set_.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    style = get(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST
    set_(hwnd, GWL_EXSTYLE, style)


WDA_EXCLUDEFROMCAPTURE = 0x00000011


def exclude_from_capture(hwnd: int) -> bool:
    """Overlay ekranda görünür ama ekran yakalamada görünmez olur (Windows 10 2004+).

    Böylece OCR kendi Türkçe yazımızı/kutumuzu okumaz ve renk seçimi gerçek oyun
    görüntüsüne göre yapılır."""
    try:
        if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return False
        value = wintypes.DWORD()
        return bool(user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(value))) and value.value == WDA_EXCLUDEFROMCAPTURE
    except AttributeError:
        return False


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class HotkeyThread(threading.Thread):
    """Ctrl+Alt+<tuş> global kısayolları. Oyun penceresi odaktayken de çalışır."""

    def __init__(self, keys: dict[str, Callable[[], None]]):
        super().__init__(daemon=True, name="hotkeys")
        self._keys = list(keys.items())
        self._thread_id: int | None = None
        self.failed: list[str] = []

    def run(self) -> None:
        self._thread_id = ctypes.WinDLL("kernel32").GetCurrentThreadId()
        for idx, (key, _cb) in enumerate(self._keys, start=1):
            if not user32.RegisterHotKey(None, idx, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, ord(key.upper())):
                self.failed.append(f"Ctrl+Alt+{key.upper()}")
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and 1 <= msg.wParam <= len(self._keys):
                self._keys[msg.wParam - 1][1]()
        for idx in range(1, len(self._keys) + 1):
            user32.UnregisterHotKey(None, idx)

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
