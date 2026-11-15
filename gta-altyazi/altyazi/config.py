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
class ExtraRegion:
    """Ana altyazı dışında okunan alan (görev hedefi, telefon mesajı...)."""

    name: str
    region: Region
    # Türkçe, İngilizcenin üstünü kapatan kutuda gösterilir ("cover") ya da hemen üstünde ("adaptive").
    display_mode: str = "cover"


# Kısayol tuşu (Ctrl+Alt+<tuş>) -> ek bölge adı.
EXTRA_REGION_KEYS = {"G": "gorev", "T": "telefon"}
EXTRA_REGION_LABELS = {"gorev": "Görev", "telefon": "Telefon"}

DEFAULT_INSTRUCTIONS = [
    "This is dialogue from a mature, uncensored crime video game. Translate profanity, insults and slang "
    "with equally strong, natural Turkish equivalents. Never soften, censor, euphemize or omit swear words.",
    "Keep the speaker's tone and register: casual street talk must stay casual Turkish street talk.",
]


@dataclass
class Config:
    # DeepL anahtarı. Boş bırakılırsa DEEPL_AUTH_KEY ortam değişkeni kullanılır.
    deepl_auth_key: str = ""
    source_lang: str = "EN"
    target_lang: str = "TR"
    # "hybrid": yerel model hemen taslak gösterir, DeepL gelince yerine geçer (en düşük gecikme + DeepL kalitesi)
    # "deepl":  yalnızca DeepL.   "local": yalnızca yerel model (internet/anahtar gerekmez).
    translation_mode: str = "hybrid"
    local_model_dir: str = "model"
    local_device: str = "auto"  # "auto" | "cuda" | "cpu"
    local_beam_size: int = 1
    # Taslağın yerine geçen son çeviri: "ollama" (ekran kartında yerel dil modeli: ücretsiz, sansürsüz)
    # veya "deepl" (ücretsiz plan; küfürleri kısmen yumuşatır).
    final_translator: str = "ollama"
    ollama_model: str = "aya-expanse:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_s: float = 10.0
    # "latency_optimized": en hızlı yanıt. "quality_optimized": daha doğal ama daha yavaş. "": DeepL varsayılanı.
    deepl_model_type: str = "latency_optimized"
    deepl_timeout_s: float = 4.0
    # Küfür ve argo yumuşatılmadan, aynı sertlikte çevrilsin (DeepL'e talimat olarak gönderilir).
    uncensored: bool = True
    deepl_instructions: list[str] = field(default_factory=lambda: list(DEFAULT_INSTRUCTIONS))
    # İsim/terim sözlüğü (İngilizce = Türkçe). Boş bırakılırsa kullanılmaz.
    glossary_file: str = "sozluk.txt"

    # Altyazının okunacağı pencere (başlığında bu ifadelerden biri geçen ilk görünür pencere).
    # Bulunamazsa ana ekranın tamamı esas alınır.
    window_titles: list[str] = field(default_factory=lambda: ["PS Remote Play", "Chiaki"])
    region: Region = field(default_factory=Region)
    # Görev / telefon gibi ek okuma bölgeleri (Ctrl+Alt+G / Ctrl+Alt+T ile seçilir).
    extra_regions: list[ExtraRegion] = field(default_factory=list)
    # Son satırlar paneli (Ctrl+Alt+H) kaç satır tutsun.
    history_size: int = 8

    # OCR
    ocr_language: str = "en-US"
    capture_interval_ms: int = 33
    # Yeni satır görülünce doğrulama okuması bu kadar ms sonra yapılır (tam kare aralığı beklenmez).
    confirm_delay_ms: int = 8
    upscale: float = 2.0
    # Altyazılar parlak (beyaz) yazılır: bölgede hiç parlak piksel yoksa OCR hiç çalıştırılmaz.
    skip_without_bright_text: bool = True
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
    # "adaptive": Türkçe, İngilizce altyazının hemen üstünde kutusuz yazılır; yazı rengi arkadaki
    #             sahneye göre anlık seçilir (açık sahnede siyah, koyu sahnede beyaz).
    # "cover":    Türkçe, İngilizcenin üstünü kapatan koyu bir kutuda sabit renkle yazılır.
    display_mode: str = "adaptive"
    font_family: str = "Segoe UI"
    font_size: int = 26
    outline_px: int = 2
    overlay_gap_px: int = 4
    # Yalnızca "cover" modunda kullanılır.
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    cover_color: str = "#101010"
    cover_padding_px: int = 6
    show_original: bool = False

    # Çeviri önbelleği (aynı satırı tekrar çevirip DeepL kotası harcamamak için).
    cache_file: str = "ceviri_onbellek.jsonl"

    def extra(self, name: str) -> ExtraRegion | None:
        return next((r for r in self.extra_regions if r.name == name), None)

    def set_extra(self, name: str, region: Region | None) -> None:
        """Ek bölgeyi ekler/günceller; None verilirse kaldırır."""
        old = self.extra(name)
        self.extra_regions = [r for r in self.extra_regions if r.name != name]
        if region is not None:
            self.extra_regions.append(ExtraRegion(name, region, old.display_mode if old else "cover"))

    def resolved_auth_key(self) -> str:
        return self.deepl_auth_key.strip() or os.environ.get("DEEPL_AUTH_KEY", "").strip()


def _region(value: dict) -> Region:
    known = {f.name for f in fields(Region)}
    return Region(**{k: float(v) for k, v in value.items() if k in known})


def _from_dict(data: dict) -> Config:
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key not in known:
            continue
        if key == "region":
            cfg.region = _region(value)
        elif key == "extra_regions":
            if not isinstance(value, list):
                continue
            cfg.extra_regions = [
                ExtraRegion(str(item["name"]), _region(item["region"]), str(item.get("display_mode", "cover")))
                for item in value
                if isinstance(item, dict) and "name" in item and "region" in item
            ]
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
