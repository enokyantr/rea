"""Terim sözlüğü: `sozluk.txt` (İngilizce = Türkçe) okunur, DeepL sözlüğüne dönüştürülür."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_TEXT = """\
# GTA Altyazı terim sözlüğü — her satır:  İngilizce = Türkçe
# Bu terimler DeepL çevirisinde her zaman bu şekilde kullanılır.
# İsimlerin çevrilmemesi için kendisiyle eşleyin (Vice City = Vice City).
# Satır başındaki # yorumdur. Dosyayı değiştirip programı yeniden başlatmanız yeterli.
# Not: Sözlük tam kelime/ifade eşleşmesiyle çalışır; tek başına çok anlamlı kelimeleri
# (ör. "shit") buraya eklemeyin, yoksa her bağlamda aynı karşılık kullanılır.

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


# GTA VI'nın bilinen karakter ve yer adları (fragmanlar ve Rockstar'ın sitesi): her zaman çevrilmeden
# kalır. Kullanıcının sozluk.txt'sindeki aynı terim bunların önüne geçer.
GAME_NAMES = [
    "Jason", "Jason Duval", "Lucia", "Lucia Caminos", "Boobie Ike", "Brian Heder", "Lori Heder",
    "Cal Hampton", "Dre'Quan", "Dre'Quan Priest", "Raul Bautista", "Real Dimez", "Bae-Luxe", "Roxy",
    "Vice City", "Leonida", "Leonida Keys", "Grassrivers", "Port Gellhorn", "Ambrosia", "Mount Kalaga",
]


class GlossaryError(ValueError):
    pass


def parse(text: str) -> dict[str, str]:
    """`İngilizce = Türkçe` satırlarını okur. Hatalı satırda satır numarasıyla hata verir."""
    entries: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise GlossaryError(f"sözlük satır {number}: '=' yok: {raw!r}")
        source, target = (part.strip() for part in line.split("=", 1))
        if not source or not target:
            raise GlossaryError(f"sözlük satır {number}: boş taraf: {raw!r}")
        if "\t" in source or "\t" in target:
            raise GlossaryError(f"sözlük satır {number}: sekme karakteri kullanılamaz")
        entries[source] = target  # aynı terim tekrar yazılırsa sonuncusu geçerli
    return entries


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        path.write_text(DEFAULT_TEXT, encoding="utf-8")
    return {**{name: name for name in GAME_NAMES}, **parse(path.read_text(encoding="utf-8-sig"))}


def fingerprint(entries: dict[str, str]) -> str:
    """Sözlük içeriğinin kısa özeti: DeepL sözlük adı ve önbellek anahtarı için."""
    blob = json.dumps(sorted(entries.items()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def needs_final(entries: dict[str, str]):
    """Sözlükte İngilizcesinden farklı karşılığı olan bir terim geçen satırlar: yerel taslak model
    sözlüğü bilmez, bu satırlar sözlüğü bilen son çevirmene (yerel LLM / DeepL) gitmeli."""
    import re

    terms = sorted((src for src, dst in entries.items() if src.casefold() != dst.casefold()), key=len, reverse=True)
    if not terms:
        return lambda text: False
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)", re.IGNORECASE)
    return lambda text: bool(pattern.search(text))
