"""OCR metnini temizleme ve titremeye karşı kararlılık (platformdan bağımsız)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

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
    text = _spell_fix(_split_glued(_fix_ocr_confusions(_SPACE.sub(" ", text).strip())))
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


# Logda tekrar tekrar görülen, İngilizcede olmayan okumalar (büyük/küçük harf duyarlı).
_WORD_FIXES = {"lfm": "I'm", "Ifm": "I'm", "lrm": "I'm", "T'm": "I'm", "Ilke": "like", "Ilike": "I like"}
_WORD_FIX = re.compile(r"(?<![\w'])(" + "|".join(_WORD_FIXES) + r")(?![\w'])")


def _contraction(m: re.Match) -> str:
    groups = [g for g in m.groups() if g is not None]
    return f"{groups[0]}'{groups[1]}"


def _fix_ocr_confusions(text: str) -> str:
    text = _NRT.sub(r"\1n't", text)
    text = _CONTRACTIONS.sub(_contraction, text)
    text = _WORD_FIX.sub(lambda m: _WORD_FIXES[m.group(1)], text)
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


# Sokak argosu / ağız (küfürsüz de olsa): yerel taslak model bunları anlamıyor ("Yall telling me"
# -> "Yall anlatıyor"), dil modeline gönderilir. OCR kesme işaretini düşürebildiği için ikisi de.
_SLANG = re.compile(
    r"\b(?:y'?all|ain'?t|finna|tryna|lemme|gimme|imma|'?cause|cuz|'?bout|nah|yo|ya|da"
    r"|homeboy|homes|homey|crib|loc|locs|ese|vato|playa|hood|cuh|fam|dope|ballin|trippin|frontin"
    r"|\w{2,}in'(?!\w)|(?:call|talk|play|mess|ly|go|do|com|say|noth|someth|buzz|bugg|chill|roll|rid|"
    r"hang|look|lov|kidd|kill|runn|work|think|try|keep|hold|gett|tell|want|need|walk)in)\b",
    re.IGNORECASE,
)


def needs_llm(text: str) -> bool:
    """Satır dil modeline gitmeli mi? Küfür/argo ya da sokak ağzı (taslak model ikisinde de zayıf)."""
    return needs_uncensored(text) or bool(_SLANG.search(text))


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


# İngilizce kelime listesi (pyspellchecker, MIT; altyazı/diyalog metinlerinden sıklıklı). Sözlükte
# az geçen (< 20) kayıtlar çoğunlukla yazım hatasıdır, alınmaz. Oyun argosu ayrıca eklenir.
_MIN_FREQ = 20
_SLANG_WORDS = {"nigga", "niggas", "niggaz", "yee", "loc", "locs", "wassup", "y'all", "ain't", "finna",
                "tryna", "homie", "homies", "cuz", "dawg", "bruh", "imma", "lemme", "gimme"}


def _game_words() -> set[str]:
    """Oyunların karakter/yer adlarındaki kelimeler (altyazi/data/*_isimler.txt): yeni bir oyunun
    isim dosyası eklenince OCR o isimleri de tanır."""
    try:
        from .glossary import load_bundled

        _general, names = load_bundled()
    except Exception:
        return set()
    return {t.lower().replace("’", "'") for name in names for t in _TOKEN.findall(name)}


@lru_cache(maxsize=1)
def _word_freq() -> dict[str, int]:
    """Kelime -> sıklık. pyspellchecker kurulu değilse boş (yalnızca kelime benzerliği kullanılır)."""
    try:
        import gzip
        import json
        from importlib.resources import files

        with gzip.open(files("spellchecker") / "resources" / "en.json.gz", "rt", encoding="utf-8") as f:
            freq = json.load(f)
    except Exception:
        return {}
    words = {w: n for w, n in freq.items() if n >= _MIN_FREQ}
    words.update(dict.fromkeys(_SLANG_WORDS, 1_000_000))
    # Oyundaki isimler: satır başında "Lucia" -> "Lucid" gibi "düzeltilmesin", hızlı kabulü
    # engellemesin; yaygınlığı yazım düzeltmesine aday olacak kadar (OCR "Lucla" -> "Lucia").
    words.update(dict.fromkeys(_game_words(), 5_000))
    return words


# Yazım düzeltmesi: OCR'ın sistematik hataları. RapidOCR çoğunlukla fazladan bir iz ekler
# ("dlick", "caill", "Niggal", "fuckinig", "cdog") ya da bir harfi karıştırır ("surgeor").
# Yalnızca sözlükte OLMAYAN, en az 4 harfli, kesme işaretine bitişik olmayan kelimeler düzeltilir;
# önce harf silme, sonra harf değiştirme denenir ve aday yeterince yaygın olmalıdır.
_SPELL_TOKEN = re.compile(r"(?<![\w'’])[A-Za-z]{4,}(?![\w'’])")
_SPELL_MIN_FREQ = 1000
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _spell_candidate(word: str, freq: dict[str, int]) -> str | None:
    deletions = {word[:i] + word[i + 1:] for i in range(len(word))}
    substitutions = {word[:i] + c + word[i + 1:] for i in range(len(word)) for c in _LETTERS}
    for candidates in (deletions, substitutions):
        best = max(candidates, key=lambda c: freq.get(c, 0))
        if freq.get(best, 0) >= _SPELL_MIN_FREQ:
            return best
    return None


def _spell_fix(text: str) -> str:
    freq = _word_freq()
    if not freq:
        return text

    def fix(m: re.Match) -> str:
        token = m.group(0)
        low = token.lower()
        if low in freq:
            return token
        # Özel isimlere dokunma: yalnızca küçük harfli kelime ya da satır başındaki büyük harfli kelime.
        first = not text[:m.start()].strip(" \"'(-")
        if not (token.islower() or (first and token[0].isupper() and token[1:].islower())):
            return token
        if low in _apostrophe_less():  # "thats" -> "that's" ("that" değil)
            best = _apostrophe_less()[low]
            return best.capitalize() if token[0].isupper() else best
        best = _spell_candidate(low, freq)
        if best is None:
            return token
        return best.capitalize() if token[0].isupper() else best

    return _SPELL_TOKEN.sub(fix, text)


# Bitişik kelimeler: RapidOCR bazı karelerde boşlukları atlıyor ("maybehe'snotaracist",
# "Moneyisntanissue", "getataxrebate"). Sözlükte olmayan uzun bir parça, sözlük kelimelerine en
# olası (sık kelime = düşük maliyet) biçimde bölünür. Kesme işaretsiz yazılan kısaltmalar da
# ("isnt", "thats") gerçek biçimine döner.
_GLUED = re.compile(r"(?<![\w'’])[A-Za-z'’]{6,}(?![\w'’])")
_SEGMENT_MIN_FREQ = 200


@lru_cache(maxsize=1)
def _apostrophe_less() -> dict[str, str]:
    """"isnt" -> "isn't": kesme işaretsiz hâli kendisi kelime olmayan yaygın kısaltmalar."""
    freq = _word_freq()
    out = {}
    for word, n in freq.items():
        if "'" in word and n >= 10_000:
            bare = word.replace("'", "")
            if freq.get(bare, 0) < _MIN_FREQ:
                out[bare] = word
    return out


def _segment(chunk: str) -> list[str] | None:
    """Küçük harfli bitişik metni kelimelere böler; bölünemezse None."""
    import math

    freq = _word_freq()
    bare = _apostrophe_less()
    n = len(chunk)
    best: list[tuple[float, int]] = [(0.0, 0)] + [(math.inf, 0)] * n
    for end in range(1, n + 1):
        for start in range(max(0, end - 20), end):
            piece = chunk[start:end]
            word = bare.get(piece, piece)
            f = freq.get(word, 0)
            if len(piece) == 1 and piece not in ("a", "i"):
                continue
            if f < (10_000 if len(piece) <= 2 else _SEGMENT_MIN_FREQ):
                continue
            cost = best[start][0] + math.log(1e9 / f)
            if cost < best[end][0]:
                best[end] = (cost, start)
    if best[n][0] == math.inf:
        return None
    words, end = [], n
    while end > 0:
        start = best[end][1]
        piece = chunk[start:end]
        words.append(bare.get(piece, piece))
        end = start
    return words[::-1]


def _split_glued(text: str) -> str:
    freq = _word_freq()
    if not freq:
        return text

    def fix(m: re.Match) -> str:
        token = m.group(0)
        low = token.lower().replace("’", "'")
        if low in freq:
            return token
        # Özel isimlere dokunma (Vinewood, Yetarian): yalnızca küçük harfli parça ya da satır başındaki
        # büyük harfli parça; satır başındakinin en az 3 kelimeye bölünmesi gerekir.
        first = not text[:m.start()].strip(" \"'(-")
        if not (token.islower() or (first and token[0].isupper() and token[1:].islower())):
            return token
        words = _segment(low)
        if not words or len(words) < (2 if token.islower() else 3):
            return token
        # İki parçaya bölünme, tek harflik bir OCR hatasından da gelebilir ("surgeor" -> "surge or"
        # değil "surgeon"): o durumda yazım düzeltmesi (sonraki adım) tercih edilir.
        if len(words) == 2 and _spell_candidate(low, freq) is not None:
            return token
        if token[0].isupper():
            words[0] = words[0].capitalize()
        return " ".join("I" if w == "i" else ("I" + w[1:] if w.startswith("i'") else w) for w in words)

    return _GLUED.sub(fix, text)


def all_words_known(text: str) -> bool:
    """Her kelime sözlükte (ya da büyük harfle başlayan bir isim) mi? Satır başındaki yarım
    okumalar ("acist", "et a tax rebate") tek okumayla hızlı kabul edilmesin diye."""
    freq = _word_freq()
    if not freq:
        return True
    for t in _TOKEN.findall(text):
        if len(t) < 2 or t.lower().replace("’", "'") in freq:
            continue
        if t[0].isupper() and t[1:].islower():
            continue
        return False
    return True


def _known_words(tokens: list[str]) -> int:
    freq = _word_freq()
    return sum(len(t) for t in tokens if t.lower().replace("’", "'") in freq)


def reading_score(text: str) -> float:
    """Bir OCR okumasının değeri. Asıl ölçü: GERÇEK İngilizce kelimelerdeki harf sayısı
    ("peniss", "xactly", "sorne" sayılmaz). Ek olarak kelimeye benzeyen harfler (özel isimler,
    sözlükte olmayan argo) ve çöp karakter / İngilizcede olmayan harf cezası.

    Gerçek kayıtlarda (300 kare, 8 satır) aynı satırın okumaları arasından seçimde hatayı
    0.199'dan 0.118'e düşürdü (karakter hata oranı). Uzun ve temiz okuma kazanır; kısmi okuma
    ("Go to") tam okumayı ("Go to Franklin's house.") yenemez."""
    tokens = _TOKEN.findall(text)
    known = _known_words(tokens)
    good = sum(len(t) for t in tokens if t.isascii() and _wordlike(t))
    non_ascii = sum(1 for ch in text if ch.isalpha() and not ch.isascii())
    junk = len(_JUNK.findall(text))
    return known - 3 * junk + 0.6 * (good - 3 * junk - 2 * non_ascii)


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

    - Aynı satırın okuması `stable_reads` kez art arda gelir ve ilk okumadan bu yana en az
      `min_hold` saniye geçerse `Show` üretir; o okumalar içinden en değerlisi (reading_score)
      gösterilir. Süre şartı önemli: 8 ms arayla yapılan iki okuma çoğu zaman videonun AYNI
      karesidir; geçiş/solma karesinin bozuk okuması böylece kendini "doğrulardı".
    - Gösterilen satırın okumaları (titreşen ya da kısmi) yeni satır sayılmaz; belirgin şekilde
      daha iyi bir okuma aynı şartla doğrulanırsa `Show` ile güncellenir (eksik satır tamamlanır).
    - `clear_after` saniye boyunca metin yoksa `Clear` üretir.
    """

    # Gösterilen satırı güncellemek için yeni okuma en az bu kadar (ve %25) daha değerli olmalı:
    # her küçük farkta yeniden çevirmemek, ama eksik okunan satırı tamamlamak için.
    UPGRADE_MARGIN = 4
    UPGRADE_RATIO = 0.25

    def __init__(self, stable_reads: int, similarity: float, clear_after: float, min_hold: float = 0.0):
        self.stable_reads = max(1, stable_reads)
        self.similarity = similarity
        self.clear_after = clear_after
        self.min_hold = max(0.0, min_hold)
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

    def feed(self, text: str, now: float, confident: bool = False) -> Show | Clear | None:
        """`confident`: OCR motoru bu okumadan çok emin (RapidOCR güveni yüksek). Tek okumayla
        gösterilir; ikinci (doğrulama) okumasının ~100-350 ms'si beklenmez."""
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
            needed = self._shown_score + max(self.UPGRADE_MARGIN, self.UPGRADE_RATIO * self._shown_score)
            if score < needed:
                return None  # gösterilen satırın sıradan/kısmi okuması: bekleyen güncellemeyi bozmaz
            # Daha iyi okuma: yeni satır gibi doğrulanması gerekir (tek bozuk kare güncelleyemez).

        if self._candidate is not None and same_line(text, self._candidate, self.similarity):
            self._candidate_count += 1
            if score >= self._candidate_score:  # en değerli okumayı tut
                self._candidate, self._candidate_score = text, score
        else:
            self._candidate, self._candidate_score = text, score
            self._candidate_count, self._candidate_since = 1, now

        if confident or (self._candidate_count >= self.stable_reads
                         and now - self._candidate_since >= self.min_hold):
            self.shown, self._shown_score = self._candidate, self._candidate_score
            self._candidate, self._candidate_count = None, 0
            return Show(self.shown, self._candidate_since)
        return None


class Dialogue(str):
    """Çevirmene giden bağlam: değeri önceki İngilizce satırdır (eski arayüzle uyumlu; DeepL ve devam
    satırı denetimi bunu kullanır), `history` ise sohbetin son satırları: [(İngilizce, verdiğimiz Türkçe)].
    Yerel dil modeli bunları sohbet geçmişi olarak görür: hitap (sen/siz), lakaplar ve konu tutarlı kalır."""

    history: tuple[tuple[str, str], ...]

    def __new__(cls, previous: str, history=()):
        obj = super().__new__(cls, previous)
        obj.history = tuple(history)
        return obj


# --- Yoksayılan ifadeler ------------------------------------------------------------------------
# Kullanıcının listesi (config.ignore_phrases): sürekli görünen oyun yönlendirmeleri ("Press [E] to
# enter") çevrilmez, gösterilmez, seslendirilmez. Büyük/küçük harf ve boşluk farkı önemsizdir; `*`
# cümle sonunu (. ! ?) aşmadan herhangi bir metne uyar. Satırın tamamına uyan ifade satırı siler;
# satırın içindeki ifade satırdan çıkarılır, geriye anlamlı bir altyazı kalmazsa satır silinir.
_IGNORE_EDGE = " \t,;:-–—|/"
_SENTENCE_START = re.compile(r"(?:^|[.!?…][\"')\]]?)\s*$")


def _ignore_pattern(phrase: str) -> re.Pattern | None:
    phrase = re.sub(r"\*+", "*", " ".join(phrase.split()))
    if not phrase.replace("*", "").strip():
        return None  # boş ya da yalnızca "*": her şeyi silerdi
    pieces = [p for p in re.split(r"(\*)", phrase) if p]
    body = []
    for i, piece in enumerate(pieces):
        if piece == "*":
            # Sondaki `*` cümle sonuna kadar (noktalaması dahil), aradaki en kısa metne uyar.
            body.append(r"[^.!?]*[.!?]?" if i == len(pieces) - 1 else r"[^.!?]*?")
            continue
        words = r"\s+".join(re.escape(w) for w in piece.split())
        lead = r"\s*" if piece[0] == " " else ""
        trail = r"\s*" if piece[-1] == " " else ""
        body.append(lead + words + trail)
    # Kelime ortasında eşleşmesin ("Press E" -> "Pressed Eagle" değil). İfade kelimeyle bitiyorsa
    # ardından gelen cümle sonu noktalaması da yakalanır (tek başına cümleyse onunla silinir).
    head = r"(?<!\w)" if re.match(r"\w", phrase) else ""
    tail = r"(?!\w)(?P<end>[.!?]+)?" if re.search(r"\w$", phrase) else ""
    return re.compile(head + "".join(body) + tail, re.IGNORECASE)


@lru_cache(maxsize=8)
def _ignore_patterns(phrases: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    return tuple(p for p in map(_ignore_pattern, phrases) if p is not None)


def ignore_phrases(text: str, phrases, min_chars: int = 3) -> str:
    """`text`ten yoksayılan ifadeleri çıkarır; geriye altyazı kalmazsa "" döner. Değişmediyse aynen."""
    if not text or not phrases:
        return text
    rest = text
    for pattern in _ignore_patterns(tuple(str(p) for p in phrases)):
        def cut(m: re.Match) -> str:
            # Cümlenin ortasındaki ifade: soru/ünlem işareti cümleye aittir, kalır.
            whole = _SENTENCE_START.search(m.string[:m.start()]) is not None
            end = m.groupdict().get("end")
            return " " if whole or not end else end
        rest = pattern.sub(cut, rest)
    if rest == text:
        return text
    rest = re.sub(r"\s+([,.!?;:])", r"\1", _SPACE.sub(" ", rest)).strip(_IGNORE_EDGE)
    return rest if is_plausible(rest, min_chars) else ""
