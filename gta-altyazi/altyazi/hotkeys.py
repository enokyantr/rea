"""Kısayol tuşları: işlemler, varsayılan atamalar, "ctrl+alt+r" biçiminin ayrıştırılması (platformdan bağımsız).

Windows'a kayıt win32.HotkeyThread'de yapılır; menü atamaları buradaki biçimle okur ve yazar."""

from __future__ import annotations

# (işlem, menüdeki açıklama, varsayılan kısayol). Boş kısayol: atanmamış.
ACTIONS: list[tuple[str, str, str]] = [
    ("menu", "Menüyü aç / gizle", "ctrl+alt+m"),
    ("pause", "Çeviriyi duraklat / devam ettir", "ctrl+alt+s"),
    ("pick", "Altyazı bölgesini seç", "ctrl+alt+r"),
    ("pick-gorev", "Görev bölgesini seç", "ctrl+alt+g"),
    ("pick-telefon", "Telefon bölgesini seç", "ctrl+alt+t"),
    ("toggle-gorev", "Görev okumayı aç / kapat", ""),
    ("toggle-telefon", "Telefon okumayı aç / kapat", ""),
    ("original", "İngilizceyi de göster", "ctrl+alt+o"),
    ("history", "Son satırlar paneli", "ctrl+alt+h"),
    ("font-up", "Yazıyı büyüt", "ctrl+alt+pageup"),
    ("font-down", "Yazıyı küçült", "ctrl+alt+pagedown"),
    ("move-up", "Yazıyı yukarı taşı", "ctrl+alt+up"),
    ("move-down", "Yazıyı aşağı taşı", "ctrl+alt+down"),
    ("tts", "Sesli okuma (dublaj) aç / kapat", "ctrl+alt+d"),
    ("quit", "Programdan çık", "ctrl+alt+q"),
]
ACTION_LABELS = {action: label for action, label, _ in ACTIONS}
DEFAULT_HOTKEYS = {action: combo for action, _, combo in ACTIONS}

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008
_MODS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_MOD_ORDER = ["ctrl", "alt", "shift", "win"]

# Tuş adı -> Windows sanal tuş kodu.
KEYS: dict[str, int] = {chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)}
KEYS.update({str(d): 0x30 + d for d in range(10)})
KEYS.update({f"f{n}": 0x6F + n for n in range(1, 25)})
KEYS.update({
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "insert": 0x2D, "delete": 0x2E,
    "plus": 0xBB, "minus": 0xBD, "comma": 0xBC, "period": 0xBE,
})
KEYS.update({f"num{d}": 0x60 + d for d in range(10)})

_DISPLAY = {"pageup": "PageUp", "pagedown": "PageDown", "up": "↑", "down": "↓", "left": "←", "right": "→",
            "space": "Boşluk", "insert": "Insert", "delete": "Delete", "home": "Home", "end": "End",
            "plus": "+", "minus": "-", "comma": ",", "period": "."}


class HotkeyError(ValueError):
    pass


def parse(combo: str) -> tuple[int, int]:
    """"ctrl+alt+r" -> (değiştirici bayrakları, sanal tuş kodu)."""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise HotkeyError("boş kısayol")
    *mods, key = parts
    flags = 0
    for mod in mods:
        if mod not in _MODS:
            raise HotkeyError(f"bilinmeyen değiştirici: {mod}")
        flags |= _MODS[mod]
    if key not in KEYS:
        raise HotkeyError(f"bilinmeyen tuş: {key}")
    if not flags & (MOD_CONTROL | MOD_ALT | MOD_WIN):
        # Tek başına harf/Shift+harf oyunda yazmayı ve kontrolleri bozardı.
        raise HotkeyError("kısayolda Ctrl, Alt ya da Win olmalı")
    return flags, KEYS[key]


def normalize(combo: str) -> str:
    """Kısayolu tek biçime getirir ("Alt+Ctrl+R" -> "ctrl+alt+r"). Geçersizse HotkeyError."""
    parse(combo)
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    *mods, key = parts
    return "+".join([m for m in _MOD_ORDER if m in mods] + [key])


def display(combo: str) -> str:
    if not combo:
        return "—"
    parts = combo.split("+")
    *mods, key = parts
    return "+".join([m.capitalize() for m in mods] + [_DISPLAY.get(key, key.upper())])


def merged(user: dict | None) -> dict[str, str]:
    """Kullanıcının atamaları + yeni sürümde eklenen işlemlerin varsayılanları. Geçersiz olanlar atılır."""
    result = dict(DEFAULT_HOTKEYS)
    for action, combo in (user or {}).items():
        if action not in result or not isinstance(combo, str):
            continue
        try:
            result[action] = normalize(combo) if combo else ""
        except HotkeyError:
            pass
    return result


def conflicts(hotkeys: dict[str, str]) -> dict[str, list[str]]:
    """Aynı kısayola atanmış birden çok işlem: kısayol -> işlemler."""
    seen: dict[str, list[str]] = {}
    for action, combo in hotkeys.items():
        if combo:
            seen.setdefault(combo, []).append(action)
    return {combo: actions for combo, actions in seen.items() if len(actions) > 1}


# Tk tuş olayı -> bizim tuş adımız (menüde "Değiştir" ile basılan tuşu yakalamak için).
_TK_KEYSYMS = {"prior": "pageup", "next": "pagedown", "up": "up", "down": "down", "left": "left",
               "right": "right", "space": "space", "insert": "insert", "delete": "delete", "home": "home",
               "end": "end", "plus": "plus", "minus": "minus", "comma": "comma", "period": "period",
               "kp_add": "plus", "kp_subtract": "minus"}
_TK_MODIFIER_KEYSYMS = {"control_l", "control_r", "alt_l", "alt_r", "shift_l", "shift_r", "win_l", "win_r",
                        "super_l", "super_r", "meta_l", "meta_r", "caps_lock"}


def from_tk_event(keysym: str, state: int, keycode: int | None = None) -> str | None:
    """Menüde basılan tuşu kısayol metnine çevirir. Yalnızca değiştirici basıldıysa None.

    Windows'ta Tk durum bitleri: Shift 0x1, Ctrl 0x4, Alt 0x20000 (sol Alt) / 0x8 (bazı sürümler)."""
    name = keysym.lower()
    if name in _TK_MODIFIER_KEYSYMS:
        return None
    if name in _TK_KEYSYMS:
        key = _TK_KEYSYMS[name]
    elif len(name) == 1 and (name.isalnum()):
        key = name
    elif name.startswith("f") and name[1:].isdigit():
        key = name
    elif name.startswith("kp_") and name[3:].isdigit():
        key = f"num{name[3:]}"
    elif keycode is not None and 0x41 <= keycode <= 0x5A:  # Ctrl+Alt ile harfler bazen farklı keysym verir
        key = chr(keycode).lower()
    else:
        return None
    mods = []
    if state & 0x4:
        mods.append("ctrl")
    if state & 0x20000 or state & 0x8:
        mods.append("alt")
    if state & 0x1:
        mods.append("shift")
    return "+".join(mods + [key])


# "Boştaki kısayollar" listesinde denenen adaylar.
def candidates() -> list[str]:
    keys = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [str(d) for d in range(10)]
    keys += [f"f{n}" for n in range(1, 13)] + ["up", "down", "left", "right", "pageup", "pagedown"]
    return [f"ctrl+alt+{k}" for k in keys]
