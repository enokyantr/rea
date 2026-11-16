"""Windows API yardımcıları (ctypes): DPI, pencere bulma, tıklanamaz overlay, global kısayollar."""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Callable

from .geometry import Rect, pick_window

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


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def visible_windows() -> list[tuple[int, str, Rect]]:
    found: list[tuple[int, str, Rect]] = []
    own_pid = os.getpid()

    def callback(hwnd, _lparam):
        # Kendi pencerelerimizin başlığını okumak Tk ana iş parçacığını beklerdi; zaten hedef değiller.
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and _window_pid(hwnd) != own_pid:
            title = _window_title(hwnd)
            if title:
                rect = _client_rect(hwnd)
                if rect is not None:
                    found.append((hwnd, title, rect))
        return True

    user32.EnumWindows(_WNDENUMPROC(callback), 0)
    return found


def find_window(title_parts: list[str]) -> tuple[str, Rect] | None:
    return pick_window(title_parts, [(title, rect) for _hwnd, title, rect in visible_windows()])



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


ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


def raise_priority() -> bool:
    """Oyun, Remote Play ve OBS açıkken işlemci önce altyazıya ayrılsın (gecikme). "Yüksek" değil:
    oyunu/yayını aç bırakmaz."""
    try:
        kernel32 = ctypes.windll.kernel32
        return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
    except (AttributeError, OSError):
        return False


WDA_NONE = 0x00000000


def include_in_capture(hwnd: int) -> bool:
    """Overlay ekran yakalamada (OBS, ekran görüntüsü) da görünür (varsayılan Windows davranışı)."""
    try:
        return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_NONE))
    except AttributeError:
        return True


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class HotkeyThread(threading.Thread):
    """Global kısayollar (oyun penceresi odaktayken de çalışır).

    `bindings`: [(etiket, değiştiriciler, sanal tuş kodu, geri çağırma)]. Kayıt sonucu `status`
    sözlüğünde: etiket -> True (çalışıyor) / False (başka program kullanıyor)."""

    def __init__(self, bindings: list[tuple[str, int, int, Callable[[], None]]]):
        super().__init__(daemon=True, name="hotkeys")
        self._bindings = list(bindings)
        self._thread_id: int | None = None
        self.status: dict[str, bool] = {}
        self.ready = threading.Event()

    @property
    def failed(self) -> list[str]:
        return [label for label, ok in self.status.items() if not ok]

    def run(self) -> None:
        self._thread_id = ctypes.WinDLL("kernel32").GetCurrentThreadId()
        for idx, (label, mods, vk, _cb) in enumerate(self._bindings, start=1):
            self.status[label] = bool(user32.RegisterHotKey(None, idx, mods | MOD_NOREPEAT, vk))
        self.ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and 1 <= msg.wParam <= len(self._bindings):
                self._bindings[msg.wParam - 1][3]()
        for idx in range(1, len(self._bindings) + 1):
            user32.UnregisterHotKey(None, idx)

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)


def hotkey_free(mods: int, vk: int) -> bool:
    """Bu kısayol şu an boşta mı (hiçbir program, bu program da dahil, kaydetmemiş)?"""
    probe_id = 0xBFFE
    if not user32.RegisterHotKey(None, probe_id, mods | MOD_NOREPEAT, vk):
        return False
    user32.UnregisterHotKey(None, probe_id)
    return True


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CreateEventW.restype = wintypes.HANDLE
kernel32.OpenEventW.restype = wintypes.HANDLE
kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.SetEvent.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
_ERROR_ALREADY_EXISTS = 183
_EVENT_MODIFY_STATE = 0x0002
_MUTEX_NAME = "Local\\GtaAltyaziTekKopya"
_SHOW_EVENT_NAME = "Local\\GtaAltyaziMenuGoster"


def claim_single_instance():
    """Program bu oturumda zaten açık mı? Açıksa ona "menünü göster" işareti gönderir ve None döner;
    değilse (mutex, olay) tutamaçlarını döner: süreç kapanınca Windows bunları bırakır."""
    mutex = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        event = kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, _SHOW_EVENT_NAME)
        if event:
            kernel32.SetEvent(event)
            kernel32.CloseHandle(event)
        if mutex:
            kernel32.CloseHandle(mutex)
        return None
    event = kernel32.CreateEventW(None, False, False, _SHOW_EVENT_NAME)  # otomatik sıfırlanan
    return mutex, event


def watch_show_requests(handles, on_request: Callable[[], None]) -> None:
    """İkinci kez başlatılınca (claim_single_instance) on_request çağrılır (arka plan iş parçacığı)."""
    if not handles or not handles[1]:
        return
    event = handles[1]

    def run():
        while True:
            if kernel32.WaitForSingleObject(event, 0xFFFFFFFF) == 0:
                on_request()
    threading.Thread(target=run, name="tek-kopya", daemon=True).start()
