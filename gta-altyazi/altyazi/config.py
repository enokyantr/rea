"""Ayarlar: config.json dosyasından okunur, eksik alanlar varsayılanla doldurulur."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Region:
    """Altyazı bölgesi, hedef pencerenin iç alanına göre oran (0..1) olarak."""

    left: float = 0.15
    top: float = 0.80
    width: float = 0.70
    height: float = 0.15


@dataclass
class Config:
    # DeepL anahtarı. Boş bırakılırsa DEEPL_AUTH_KEY ortam değişkeni kullanılır.
    deepl_auth_key: str = ""
    source_lang: str = "EN"
    target_lang: str = "TR"

    # Altyazının okunacağı pencere (başlığında bu ifadelerden biri geçen ilk görünür pencere).
    # Bulunamazsa ana ekranın tamamı esas alınır.
    window_titles: list[str] = field(default_factory=lambda: ["PS Remote Play", "Chiaki"])
    region: Region = field(default_factory=Region)

    # OCR
    ocr_language: str = "en-US"
    capture_interval_ms: int = 120
    upscale: float = 2.0
    # "none": gri ton, "bright": sadece parlak (beyaz) yazıyı bırak — beyaz altyazılarda iyi çalışır.
    preprocess: str = "none"
    bright_threshold: int = 190
    min_text_chars: int = 3

    # Kararlılık: aynı metin art arda kaç kez okunursa çevrilsin.
    stable_reads: int = 2
    similarity: float = 0.88
    # Altyazı kaybolduktan kaç ms sonra Türkçesi de silinsin.
    clear_after_ms: int = 700

    # Overlay görünümü
    font_family: str = "Segoe UI"
    font_size: int = 26
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_px: int = 2
    # True: orijinal İngilizce altyazının üstünü kutuyla kapatır.
    cover_original: bool = True
    cover_color: str = "#101010"
    cover_padding_px: int = 6
    show_original: bool = False

    # Çeviri önbelleği (aynı satırı tekrar çevirip DeepL kotası harcamamak için).
    cache_file: str = "ceviri_onbellek.jsonl"

    def resolved_auth_key(self) -> str:
        return self.deepl_auth_key.strip() or os.environ.get("DEEPL_AUTH_KEY", "").strip()


def _from_dict(data: dict) -> Config:
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key not in known:
            continue
        if key == "region":
            region_known = {f.name for f in fields(Region)}
            cfg.region = Region(**{k: float(v) for k, v in value.items() if k in region_known})
        else:
            setattr(cfg, key, value)
    return cfg


def load(path: Path) -> Config:
    if not path.exists():
        cfg = Config()
        save(cfg, path)
        return cfg
    with path.open(encoding="utf-8") as fh:
        return _from_dict(json.load(fh))


def save(cfg: Config, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
