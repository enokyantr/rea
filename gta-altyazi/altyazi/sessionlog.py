"""Oturum kaydı: oyunda görülen her satır (İngilizce + Türkçe) bir metin dosyasına yazılır.

Kaçırılan sahneleri sonradan okumak ve kötü çevirileri toplu bildirmek için. Her oturum ayrı dosya:
kayitlar/2026-11-16_04-10.txt. Satır başına bir yazma + flush: program çökse de kayıt kaybolmaz."""

from __future__ import annotations

import time
from pathlib import Path


class SessionLog:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / time.strftime("%Y-%m-%d_%H-%M.txt")
        self._fh = self.path.open("a", encoding="utf-8")
        self._last: tuple[str, str] | None = None

    def write(self, label: str, original: str, translated: str) -> None:
        if (original, translated) == self._last:  # aynı satırın tekrarı (ör. yeniden okuma)
            return
        self._last = (original, translated)
        prefix = f"[{label}] " if label else ""
        self._fh.write(f"{time.strftime('%H:%M:%S')}  {prefix}{original}\n          {translated}\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass
