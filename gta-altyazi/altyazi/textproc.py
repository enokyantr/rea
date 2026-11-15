"""OCR metnini temizleme ve titremeye karşı kararlılık (platformdan bağımsız)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_SPACE = re.compile(r"\s+")
# OCR'ın kenarlarda ürettiği tek başına kalmış işaretler.
_EDGE_JUNK = re.compile(r"^[\W_]+(?=\w)|(?<=[\w.!?\"')])[^\w.!?\"')]+$")


def _fold_accents(text: str) -> str:
    """İngilizce altyazıda aksanlı harf olmaz; OCR'ın okuduğu å/é/ö... gürültüdür: a/e/o..."""
    if text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(lines: list[str]) -> str:
    """Satırları tek cümleye birleştirir, boşlukları ve kenar çöplerini temizler."""
    text = _fold_accents(" ".join(line.strip() for line in lines if line.strip()))
    text = _fix_ocr_confusions(_SPACE.sub(" ", text).strip())
    text = _EDGE_JUNK.sub("", text).strip()
    # Satır sonunda kelime bölünmesi: "some- thing" -> "something"
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return text


# Kontur maskesinde OCR'ın sistematik karışıklıkları (gerçek log): kesme işareti -> r
# ("donrt", "Irm"), l -> [ ( { ("[ike", "exect[y"). Değiştirilen biçimler gerçek İngilizce kelime
# değildir, bu yüzden düzeltme doğru okumayı bozmaz.
_NRT = re.compile(r"\b(do|does|did|is|are|was|were|have|has|had|could|would|should|must|need"
                  r"|ca|wo|ai)nrt\b", re.IGNORECASE)
_CONTRACTIONS = re.compile(
    r"\b(?:(I)r(m|ll|ve|d)|(you|they)r(re|ll|ve)|(we)r(re|ll|ve)"
    r"|(it|that|what|there|here|who|let)r(s))\b", re.IGNORECASE)
_L_LOOKALIKE = re.compile(r"(?<=[A-Za-z])[\[{(](?=[a-z])|(?<![\w])[\[{(](?=[a-z]{2})")


def _contraction(m: re.Match) -> str:
    groups = [g for g in m.groups() if g is not None]
    return f"{groups[0]}'{groups[1]}"


def _fix_ocr_confusions(text: str) -> str:
    text = _NRT.sub(r"\1n't", text)
    text = _CONTRACTIONS.sub(_contraction, text)
    # Gerçek parantez/köşeli parantez kapanışıyla gelir ("[laughs]"); kapanış yoksa harf l'dir.
    if not any(c in text for c in ")]}"):
        text = _L_LOOKALIKE.sub("l", text)
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


def reading_score(text: str) -> float:
    """Bir OCR okumasının değeri: kelimeye benzeyen harf sayısı, çöp karakter ve İngilizcede
    olmayan harfler (å, é...) cezalı. Uzun ve temiz okuma kazanır; kısmi okuma ("Go to") tam
    okumayı ("Go to Franklin's house.") yenemez."""
    tokens = _TOKEN.findall(text)
    good = sum(len(t) for t in tokens if t.isascii() and _wordlike(t))
    non_ascii = sum(1 for ch in text if ch.isalpha() and not ch.isascii())
    return good - 3 * len(_JUNK.findall(text)) - 2 * non_ascii


def _words(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text) if len(t) >= 2}


def same_line(a: str, b: str, threshold: float) -> bool:
    """İki okuma aynı altyazı satırı mı? Benzer metin ya da biri diğerinin parçası (kısmi okuma:
    "you got" ⊂ "Maybe if you got rid of that haircut you got")."""
    if similar(a, b, threshold):
        return True
    wa, wb = _words(a), _words(b)
    small, big = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    return bool(small) and len(small & big) >= max(1, round(0.8 * len(small)))


@dataclass(frozen=True)
class Show:
    text: str
    first_seen: float  # satırın ilk okunduğu an (gecikme ölçümü için)


@dataclass(frozen=True)
class Clear:
    pass


class Stabilizer:
    """OCR okumalarını olaylara çevirir.

    - Aynı satırın okuması `stable_reads` kez art arda gelirse `Show` üretir; o okumalar içinden en
      değerlisi (reading_score) gösterilir.
    - Gösterilen satırın okumaları (titreşen ya da kısmi) yeni satır sayılmaz; ama belirgin şekilde
      daha iyi bir okuma gelirse `Show` ile güncellenir (eksik okunan satır tamamlanır).
    - `clear_after` saniye boyunca metin yoksa `Clear` üretir.
    """

    # Gösterilen satırı güncellemek için yeni okuma en az bu kadar (ve %25) daha değerli olmalı:
    # her küçük farkta yeniden çevirmemek, ama eksik okunan satırı tamamlamak için.
    UPGRADE_MARGIN = 4
    UPGRADE_RATIO = 0.25

    def __init__(self, stable_reads: int, similarity: float, clear_after: float):
        self.stable_reads = max(1, stable_reads)
        self.similarity = similarity
        self.clear_after = clear_after
        self.shown: str | None = None
        self._shown_score = 0.0
        self._candidate: str | None = None
        self._candidate_score = 0.0
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
        score = reading_score(text)
        if self.shown is not None and same_line(text, self.shown, self.similarity):
            self._candidate, self._candidate_count = None, 0
            if score >= self._shown_score + max(self.UPGRADE_MARGIN, self.UPGRADE_RATIO * self._shown_score):
                self.shown, self._shown_score = text, score  # aynı satırın daha iyi okuması
                return Show(text, now)
            return None

        if self._candidate is not None and same_line(text, self._candidate, self.similarity):
            self._candidate_count += 1
            if score >= self._candidate_score:  # en değerli okumayı tut
                self._candidate, self._candidate_score = text, score
        else:
            self._candidate, self._candidate_score = text, score
            self._candidate_count, self._candidate_since = 1, now

        if self._candidate_count >= self.stable_reads:
            self.shown, self._shown_score = self._candidate, self._candidate_score
            self._candidate, self._candidate_count = None, 0
            return Show(self.shown, self._candidate_since)
        return None
