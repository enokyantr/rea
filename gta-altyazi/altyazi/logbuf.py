"""Programın konsol çıktısının son satırları (menüdeki "Kayıt" bölümü için).

Konsola yazılan her şey (log satırları, uyarılar, hata dökümleri) aynen konsola gider ve ayrıca
burada sınırlı sayıda tutulur; kullanıcı sorun anında menüden kopyalayıp gönderebilir."""

from __future__ import annotations

import sys
import threading
from collections import deque

MAX_LINES = 3000


class LogBuffer:
    def __init__(self, max_lines: int = MAX_LINES):
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self.total = 0  # şimdiye kadar eklenen satır sayısı (silinenler dahil)

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            self.total += 1

    def since(self, seen: int) -> tuple[int, list[str]]:
        """`seen` satır görülmüşse sonrakiler (tampondan düşenler hariç) ve yeni toplam."""
        with self._lock:
            new = min(self.total - seen, len(self._lines))
            lines = list(self._lines)[len(self._lines) - new:] if new > 0 else []
            return self.total, lines

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


class _Tee:
    """Akışa yazılanı aynen iletir, tamamlanan satırları tampona da ekler."""

    def __init__(self, stream, buffer: LogBuffer):
        self._stream = stream
        self._buffer = buffer
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        with self._lock:
            parts = (self._partial + text.replace("\r", "\n")).split("\n")
            self._partial = parts.pop()
        for line in parts:
            if line.strip():
                self._buffer.add(line)
        return len(text)

    def flush(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


BUFFER = LogBuffer()


def install() -> None:
    """stdout ve stderr'i tampona da yaz (bir kez)."""
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, BUFFER)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr, BUFFER)
