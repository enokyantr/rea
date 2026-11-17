"""Terim sözlüğü: programla gelen sözlükler ve kullanıcının `sozluk.txt`'si (İngilizce = Türkçe).

Kaynaklar ve öncelik (düşükten yükseğe; aynı terimde sonraki kazanır):
  1. altyazi/data/oyun_sozlugu.txt — genel oyun terimleri (varsa)
  2. altyazi/data/*_isimler.txt    — oyunların karakter/yer adları (çevrilmeden kalır)
  3. kullanıcının sozluk.txt'si

Satırda hangi terimlerin geçtiğini bulmanın tek sahibi `Matcher`'dır: yerel LLM istemi, önbellek
anahtarı ve son çevirmene yönlendirme aynı eşleşmeyi kullanır."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

# Paket verisi modül dosyasına göre bulunur: `python -m altyazi` hangi klasörden çalıştırılırsa
# çalıştırılsın (PyInstaller paketinde de _internal/altyazi/data altında olmalı).
DATA_DIR = Path(__file__).resolve().parent / "data"
GENERAL_FILE = "oyun_sozlugu.txt"
NAMES_GLOB = "*_isimler.txt"

DEFAULT_TEXT = """\
# GTA Altyazı terim sözlüğü — her satır:  İngilizce = Türkçe
# Bu terimler DeepL çevirisinde her zaman bu şekilde kullanılır.
# İsimlerin çevrilmemesi için kendisiyle eşleyin (Vice City = Vice City).
# Satır başındaki # yorumdur. Dosyayı değiştirip programı yeniden başlatmanız yeterli.
# Not: Sözlük tam kelime/ifade eşleşmesiyle çalışır; tek başına çok anlamlı kelimeleri
# (ör. "shit") buraya eklemeyin, yoksa her bağlamda aynı karşılık kullanılır.
# Programla gelen genel oyun sözlüğü ve oyun adları ayrıca yüklenir; buradaki aynı terim onları geçersiz kılar.

Jason = Jason
Lucia = Lucia
Vice City = Vice City
Leonida = Leonida
Port Gellhorn = Port Gellhorn
Ambrosia = Ambrosia
Grassrivers = Grassrivers
Mount Kalaga = Mount Kalaga
Leonida Keys = Leonida Keys
"""


class GlossaryError(ValueError):
    pass


class Glossary(dict):
    """Birleşik sözlük (terim -> karşılık). `core`: genel oyun sözlüğü hariç, yalnızca oyun adları ve
    kullanıcı terimleri (DeepL'e büyük sözlük gönderilemediğinde ve taslak/son çevirmen
    yönlendirmesinde kullanılır; genel sözlük her satırı son çevirmene göndermesin)."""

    def __init__(self, entries=(), core: dict[str, str] | None = None):
        super().__init__(entries)
        self.core: dict[str, str] = dict(self) if core is None else dict(core)


def parse(text: str, source: str = "sözlük") -> dict[str, str]:
    """`İngilizce = Türkçe` satırlarını okur. Hatalı satırda dosya adı ve satır numarasıyla hata verir."""
    entries: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise GlossaryError(f"{source} satır {number}: '=' yok: {raw!r}")
        src, target = (part.strip() for part in line.split("=", 1))
        if not src or not target:
            raise GlossaryError(f"{source} satır {number}: boş taraf: {raw!r}")
        if "\t" in src or "\t" in target:
            raise GlossaryError(f"{source} satır {number}: sekme karakteri kullanılamaz")
        entries[src] = target  # aynı terim tekrar yazılırsa sonuncusu geçerli
    return entries


def _read(path: Path) -> dict[str, str]:
    return parse(path.read_text(encoding="utf-8-sig"), source=path.name)


def _stamp(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


@lru_cache(maxsize=4)
def _bundled_cached(data_dir: str, stamps: tuple) -> tuple[dict[str, str], dict[str, str]]:
    base = Path(data_dir)
    general = _read(base / GENERAL_FILE) if (base / GENERAL_FILE).is_file() else {}
    names: dict[str, str] = {}
    for path in sorted(base.glob(NAMES_GLOB)):
        names.update(_read(path))
    return general, names


def load_bundled(data_dir: Path = DATA_DIR) -> tuple[dict[str, str], dict[str, str]]:
    """Programla gelen (genel sözlük, oyun adları). Klasör/dosya yoksa boş. Dosyalar değişmedikçe
    yeniden okunmaz (sözlük açılışta birkaç kez istenir)."""
    if not data_dir.is_dir():
        return {}, {}
    files = [data_dir / GENERAL_FILE, *sorted(data_dir.glob(NAMES_GLOB))]
    stamps = tuple((p.name, _stamp(p)) for p in files)
    general, names = _bundled_cached(str(data_dir), stamps)
    return dict(general), dict(names)


def load(path: Path, data_dir: Path = DATA_DIR) -> Glossary:
    """Tüm kaynakları öncelik sırasıyla birleştirir; kullanıcı dosyası yoksa varsayılanı yazar."""
    if not path.exists():
        path.write_text(DEFAULT_TEXT, encoding="utf-8")
    general, names = load_bundled(data_dir)
    user = _read(path)
    core = {**names, **user}
    return Glossary({**general, **core}, core=core)


def fingerprint(entries: dict[str, str]) -> str:
    """Sözlük içeriğinin kısa özeti: DeepL sözlük adı ve önbellek anahtarı için."""
    blob = json.dumps(sorted(entries.items()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# --- satırda geçen terimler ----------------------------------------------------------------------

# Kelime: harf/rakam, içinde kesme işareti olabilir (Dre'Quan, y'all). Kesmeyle ayrılmış parça ayrı
# kelime sayılmaz: "Quan" terimi "Dre'Quan" içinde eşleşmez.
_TOKEN = re.compile(r"\w+(?:['’]\w+)*")
# Son kelimedeki iyelik/kısaltma eki: "Lucia's", "Jason'll" -> "Lucia" / "Jason" terimi eşleşir.
_CLITIC = re.compile(r"'(?:s|ll|re|ve|d|m)$")
_SPACE = re.compile(r"\s+")


def _norm(word: str) -> str:
    return word.casefold().replace("’", "'")


def _split(text: str) -> tuple[list[str], list[str]]:
    """(kelimeler, aradaki ayraçlar). Ayraçta boşluklar tek boşluğa indirilir ("Vice  City")."""
    words, gaps, end = [], [], None
    for m in _TOKEN.finditer(text):
        if end is not None:
            gaps.append(_SPACE.sub(" ", text[end:m.start()]))
        words.append(_norm(m.group()))
        end = m.end()
    return words, gaps


class Matcher:
    """Satırda tam kelime/ifade olarak (büyük-küçük harf duyarsız) geçen sözlük terimleri.

    Uzun ifade önce: "Leonida Keys" eşleşince içindeki "Leonida" ayrıca sayılmaz. Terimler ilk
    kelimelerine göre dizinlenir; binlerce terimde de satır başına yalnızca birkaç karşılaştırma."""

    CACHE_SIZE = 512

    def __init__(self, entries: dict[str, str]):
        self.entries = dict(entries)
        self._index: dict[str, list[tuple[list[str], list[str], str]]] = {}
        for term in self.entries:
            words, gaps = _split(term)
            if words:
                self._index.setdefault(words[0], []).append((words, gaps, term))
        for candidates in self._index.values():
            candidates.sort(key=lambda c: (len(c[0]), len(c[2])), reverse=True)
        # Aynı satır kısa sürede birkaç kez sorulur (önbellek anahtarı, istem, önceden çeviri).
        self._cache: dict[str, tuple[str, ...]] = {}
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self._index)

    def terms(self, text: str) -> tuple[str, ...]:
        """Satırda geçen terimler (sözlükteki yazımıyla), satırdaki sırayla, tekrarsız."""
        if not self._index:
            return ()
        with self._lock:
            hit = self._cache.get(text)
        if hit is not None:
            return hit
        found = self._find(text)
        with self._lock:
            if len(self._cache) >= self.CACHE_SIZE:
                self._cache.clear()
            self._cache[text] = found
        return found

    def _find(self, text: str) -> tuple[str, ...]:
        words, gaps = _split(text)
        stems = [_CLITIC.sub("", w) for w in words]
        found: list[str] = []
        i = 0
        while i < len(words):
            step = 1
            candidates = self._index.get(words[i], [])
            if stems[i] != words[i]:
                candidates = candidates + self._index.get(stems[i], [])
            for t_words, t_gaps, term in candidates:
                n = len(t_words)
                if i + n > len(words) or gaps[i:i + n - 1] != t_gaps:
                    continue
                last = i + n - 1
                if words[i:last] == t_words[:-1] and t_words[-1] in (words[last], stems[last]):
                    if term not in found:
                        found.append(term)
                    step = n
                    break
            i += step
        return tuple(found)

    def matched(self, text: str) -> dict[str, str]:
        return {term: self.entries[term] for term in self.terms(text)}

    def fingerprint(self, text: str) -> str:
        """Satıra uygulanan terimlerin özeti; terim yoksa "". İlgisiz terim eklemek değiştirmez."""
        matched = self.matched(text)
        return fingerprint(matched) if matched else ""


def needs_final(entries: dict[str, str]):
    """Sözlükte İngilizcesinden farklı karşılığı olan bir terim geçen satırlar: yerel taslak model
    sözlüğü bilmez, bu satırlar sözlüğü bilen son çevirmene (yerel LLM / DeepL) gitmeli.
    Genel oyun sözlüğü sayılmaz (yalnızca kullanıcı terimleri ve oyun adları): yaygın oyun kelimeleri
    neredeyse her satırı yavaş son çevirmene gönderirdi."""
    core = getattr(entries, "core", entries)
    matcher = Matcher({src: dst for src, dst in core.items() if src.casefold() != dst.casefold()})
    if not matcher:
        return lambda text: False
    return lambda text: bool(matcher.terms(text))
