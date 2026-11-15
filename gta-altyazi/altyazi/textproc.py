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


# Cümle sonu (. ! ? …, ardından isteğe bağlı kapanış tırnağı), boşluk ve büyük harf/tırnakla
# başlayan yeni cümle. Yaygın unvan kısaltmalarından (Mr. Dr. …) sonra bölünmez.
_ABBREV = "".join(rf"(?<!\b{a}\.)" for a in ("Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "vs"))
_SENTENCE_END = re.compile(
    rf"(?:(?<=[.!?…])|(?<=[.!?…][\"')\]])){_ABBREV}\s+(?=[\"'(\[]?[A-ZÇĞİÖŞÜ0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Altyazı satırını cümlelere böler. Yerel çeviri modeli (Opus-MT) çok cümleli girdide
    sonraki cümleleri atabildiği için her cümle ayrı çevrilir."""
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p] or [text]


# Küfür/argo içeren satırlar: yerel taslak model bunları yumuşatıyor, son çevirmene gönderilir.
_PROFANITY = re.compile(
    r"\b(?:\w*fuck\w*|\w*shit\w*|bitch\w*|sons? of a bitch|ass|asses|\w+ass(?:es)?|ass(?:hole|hat|wipe)s?"
    r"|bastards?|dicks?|dickhead\w*|cocks?(?:sucker)?s?|puss(?:y|ies)|cunts?|piss\w*|damn\w*|goddamn\w*"
    r"|crap\w*|bullshit|hoes?|whores?|sluts?|twats?|pricks?|wankers?|bollocks|screw (?:you|this|that|off)"
    r"|wtf|stfu|bro|bruh|homie|dawg|nigga[sz]?|niggaz|ho)\b",
    re.IGNORECASE,
)
# "ass" ile biten ama küfür olmayan sık kelimeler (\w+ass yakalamasın).
_NOT_PROFANE = re.compile(r"\b(?:class|glass|grass|pass|mass|brass|bass|lass|compass|embarrass|harass|"
                          r"surpass|bypass|amass|carcass|morass|trespass|crass|overpass|underpass|"
                          r"sunglass|eyeglass|hourglass|wineglass)(?:es|ed|ing)?\b", re.IGNORECASE)


def needs_uncensored(text: str) -> bool:
    """Satırda küfür/argo var mı? (Bu satırlar sansürsüz son çevirmene gider.)"""
    cleaned = _NOT_PROFANE.sub(" ", text)
    return bool(_PROFANITY.search(cleaned))


_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_VOWELS = set("aeiouyAEIOUY")
# Tekrarlanan harfler teke indirilmiş hâlleri (hmmm -> hm, psst -> pst).
_INTERJECTIONS = {"hm", "m", "sh", "pst", "pft", "br", "gr", "tsk", "hmph", "pf", "ps"}


def _wordlike(token: str) -> bool:
    """Gerçek kelimeye benziyor mu: sesli harf içerir, sessiz harf yığını değildir."""
    if len(token) == 1:
        return token in ("I", "a", "A", "O", "o")
    if re.sub(r"(.)\1+", r"\1", token.lower()) in _INTERJECTIONS:
        return True  # sesli harfsiz ünlemler: hmm, shhh, psst, pfft, brr...
    if not any(ch in _VOWELS for ch in token):
        return False
    run = longest = 0
    for ch in token.lower():
        run = 0 if ch in "aeiouy'’" else run + 1
        longest = max(longest, run)
    return longest <= 4


_JUNK = re.compile(r"[•=|\\\[\]{}<>~^*_@#%$¦‹›«»„]|[A-Za-z]\d|\d[A-Za-z]")


def text_quality(text: str) -> float:
    """Okunan metnin gerçek metne benzerliği (0..1, çöp karakterler cezalandırılır). Birden çok
    ön işleme seçeneğinin OCR sonucu arasından seçim için."""
    tokens = _TOKEN.findall(text)
    if not tokens:
        return 0.0
    total = sum(len(t) for t in tokens)
    ratio = sum(len(t) for t in tokens if _wordlike(t)) / total
    return max(0.0, ratio - 0.1 * len(_JUNK.findall(text)))


def is_plausible(text: str, min_chars: int) -> bool:
    """Gerçek altyazıya benziyor mu? Harf oranı düşük gürültüyü ve OCR'ın bozuk okuduğu
    anlamsız satırları (ör. 'ntpgmfl/ Wag', '3hjc.') eler."""
    if len(text) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < min_chars or letters / len(text) < 0.55:
        return False
    tokens = _TOKEN.findall(text)
    if not tokens:
        return False
    wordlike = sum(len(t) for t in tokens if _wordlike(t))
    return wordlike / sum(len(t) for t in tokens) >= 0.7


def similar(a: str, b: str, threshold: float) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


@dataclass(frozen=True)
class Show:
    text: str
    first_seen: float  # satırın ilk okunduğu an (gecikme ölçümü için)


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
        self._candidate_since = 0.0
        self._last_text_at: float | None = None

    @property
    def pending(self) -> bool:
        """Doğrulanmayı bekleyen yeni bir satır var mı (bir sonraki okuma hemen yapılmalı)."""
        return self._candidate is not None

    def reset(self) -> None:
        """Ekranda hiçbir şey gösterilmiyormuş gibi baştan başla (duraklatma/devam sonrası)."""
        self.shown = None
        self._candidate, self._candidate_count = None, 0

    def forget_shown(self) -> None:
        """Gösterilen satır başarısız olduysa (çeviri hatası) aynı satır tekrar okununca yeniden dene."""
        self.shown = None

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
            self._candidate, self._candidate_count, self._candidate_since = text, 1, now

        if self._candidate_count >= self.stable_reads:
            self.shown = self._candidate
            self._candidate, self._candidate_count = None, 0
            return Show(self.shown, self._candidate_since)
        return None
