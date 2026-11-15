"""OCR metnini temizleme ve titremeye karşı kararlılık (platformdan bağımsız)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_SPACE = re.compile(r"\s+")
# OCR'ın kenarlarda ürettiği tek başına kalmış işaretler.
_EDGE_JUNK = re.compile(r"^[\W_]+(?=\w)|(?<=[\w.!?\"')])[^\w.!?\"')]+$")


def normalize(lines: list[str]) -> str:
    """Satırları tek cümleye birleştirir, boşlukları ve kenar çöplerini temizler."""
    text = " ".join(line.strip() for line in lines if line.strip())
    text = _SPACE.sub(" ", text).strip()
    text = _EDGE_JUNK.sub("", text).strip()
    # Satır sonunda kelime bölünmesi: "some- thing" -> "something"
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return text


def is_plausible(text: str, min_chars: int) -> bool:
    """Gerçek altyazıya benziyor mu? Harf oranı düşük gürültüyü eler."""
    if len(text) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= min_chars and letters / len(text) >= 0.55


def similar(a: str, b: str, threshold: float) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


@dataclass(frozen=True)
class Show:
    text: str


@dataclass(frozen=True)
class Clear:
    pass


class Stabilizer:
    """OCR okumalarını olaylara çevirir.

    - Aynı (benzer) metin `stable_reads` kez art arda okunursa `Show` üretir.
    - Gösterilen metne benzer okumalar yok sayılır (OCR titremesi).
    - `clear_after` saniye boyunca metin yoksa `Clear` üretir.
    """

    def __init__(self, stable_reads: int, similarity: float, clear_after: float):
        self.stable_reads = max(1, stable_reads)
        self.similarity = similarity
        self.clear_after = clear_after
        self.shown: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_text_at: float | None = None

    def feed(self, text: str, now: float) -> Show | Clear | None:
        if not text:
            self._candidate, self._candidate_count = None, 0
            if self.shown is not None and self._last_text_at is not None:
                if now - self._last_text_at >= self.clear_after:
                    self.shown = None
                    return Clear()
            return None

        self._last_text_at = now
        if self.shown is not None and similar(text, self.shown, self.similarity):
            self._candidate, self._candidate_count = None, 0
            return None

        if self._candidate is not None and similar(text, self._candidate, self.similarity):
            self._candidate_count += 1
            # En uzun okumayı tut: kısmi okumalar genelde kısa olur.
            if len(text) >= len(self._candidate):
                self._candidate = text
        else:
            self._candidate, self._candidate_count = text, 1

        if self._candidate_count >= self.stable_reads:
            self.shown = self._candidate
            self._candidate, self._candidate_count = None, 0
            return Show(self.shown)
        return None
