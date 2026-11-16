"""Ayarlar: config.json dosyasından okunur, eksik alanlar varsayılanla doldurulur."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .hotkeys import DEFAULT_HOTKEYS, merged as merged_hotkeys


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
    # Menüden / kısayolla kapatılabilir; bölge silinmeden okuma durur.
    enabled: bool = True


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
    # "auto": GPU varsa GPU; ancak son çevirmen yerel LLM (ollama) ise işlemci — ikisi GPU'yu
    # paylaşınca LLM satırları ~10 kat yavaşlıyor. "cuda" / "cpu": zorla.
    local_device: str = "auto"
    local_beam_size: int = 1
    # Taslağın yerine geçen son çeviri: "ollama" (ekran kartında yerel dil modeli: ücretsiz, sansürsüz)
    # veya "deepl" (ücretsiz plan; küfürleri kısmen yumuşatır).
    final_translator: str = "ollama"
    # 8 GB ekran kartına sığan ve küfürü çeviren model (RTX 4060 Ti 8 GB: %100 GPU, ~330 ms/satır).
    # gemma3:12b Türkçede daha iyi ama ~8 GB; 12 GB+ kartlarda tercih edilebilir (ollama_use_context: true ile).
    ollama_model: str = "aya-expanse:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_s: float = 10.0
    ollama_num_ctx: int = 2048
    # true: yerel LLM yalnızca küfür/argo içeren satırlarda kullanılır; diğer satırlarda yerel taslak
    # çeviri (daha doğru ve hızlı) nihai sonuçtur. false: her satır LLM'e gider.
    ollama_only_profanity: bool = True
    # Önceki satırı bağlam olarak gönder. Bağlamı da çeviren modellerde (ör. aya-expanse) false yapın.
    ollama_use_context: bool = False
    # Satır öncekinin devamıysa (önceki cümle bitmemiş / bu satır küçük harfle başlıyor) önceki satırı
    # yine de bağlam olarak gönder: yarım cümleler bağlamsız yanlış çevriliyor.
    ollama_context_continuation: bool = True
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
    # Ana altyazı okuması açık mı (menüden kapatılabilir).
    region_enabled: bool = True
    # Görev / telefon gibi ek okuma bölgeleri (Ctrl+Alt+G / Ctrl+Alt+T ile seçilir).
    extra_regions: list[ExtraRegion] = field(default_factory=list)
    # Son satırlar paneli (Ctrl+Alt+H) kaç satır tutsun.
    history_size: int = 8

    # OCR
    # "auto": RapidOCR (kurulu değilse Windows OCR). "windows": yalnızca Windows OCR.
    ocr_engine: str = "auto"
    # RapidOCR nerede çalışsın: "auto" (ekran kartı/DirectML varsa) | "cpu". Yerel dil modeli
    # yavaşlarsa "cpu" deneyin.
    ocr_device: str = "auto"
    ocr_language: str = "en-US"
    capture_interval_ms: int = 33
    # Yeni satır görülünce doğrulama okuması bu kadar ms sonra yapılır (tam kare aralığı beklenmez).
    confirm_delay_ms: int = 8
    upscale: float = 2.0
    # Altyazılar parlak (beyaz) yazılır: bölgede hiç parlak piksel yoksa OCR hiç çalıştırılmaz.
    skip_without_bright_text: bool = True
    # "auto": gri ton ve kontur maskesi denenir, gerçek metne en çok benzeyen okuma seçilir.
    # "none": yalnızca gri ton. "outline": yalnızca kontur maskesi. "bright": yalnızca parlak pikseller.
    preprocess: str = "auto"
    bright_threshold: int = 190
    # "outline": bu değerden koyu pikseller kontur sayılır; yazı pikselleri kontura en fazla
    # outline_radius piksel uzaklıkta olmalı.
    # Gerçek videoda (sıkıştırma: yazı saf beyaz değil, kontur tam siyah değil) sıkı eşikler
    # (232/90/3) harfleri parçaladı; gevşek değerler.
    dark_threshold: int = 120
    outline_radius: int = 4
    # "outline": yazı bu değerden parlak olmalı (altyazı ~240-255; açık gökyüzü genelde daha düşük).
    # Yazı eksik okunursa düşürün, açık sahnede harfler kalınlaşıyorsa yükseltin.
    outline_white_threshold: int = 205
    # "tophat": yerel arka plandan bu yarıçapta (piksel) daha parlak olan ince vuruşlar yazı sayılır.
    # Koyu sahnede 3, çok açık sahnede 2 biraz daha iyi.
    tophat_radius: int = 3
    min_text_chars: int = 3

    # Kararlılık: aynı metin art arda kaç kez okunursa çevrilsin.
    stable_reads: int = 2
    similarity: float = 0.88
    # Yeni satır, ilk okumasından en az bu kadar ms sonra da aynı okunmalı (videonun farklı bir
    # karesi). Geçiş/solma karelerinin bozuk okumalarının çevrilmesini önler.
    confirm_hold_ms: int = 40
    # RapidOCR okumadan bu kadar eminse (0..1) doğrulama okuması beklenmeden gösterilir.
    # 1.0 = her zaman doğrula (eski davranış, ~100-350 ms daha geç).
    fast_accept_confidence: float = 0.93
    # Altyazı ekrandayken yalnızca yazının alanı değişince yeniden oku (ekran kartı/işlemci dil
    # modeline ve çeviriye kalır). Yine de en geç bu kadar ms'de bir yeniden okunur.
    change_detection: bool = True
    reread_max_ms: int = 1000
    # Altyazı kaybolduktan kaç ms sonra Türkçesi de silinsin.
    clear_after_ms: int = 700

    # Overlay görünümü
    # "adaptive": Türkçe, İngilizce altyazının hemen üstünde kutusuz yazılır; yazı rengi arkadaki
    #             sahneye göre anlık seçilir (açık sahnede siyah, koyu sahnede beyaz).
    # "cover":    Türkçe, İngilizcenin üstünü kapatan koyu bir kutuda sabit renkle yazılır.
    display_mode: str = "adaptive"
    # Türkçe altyazı yayında (OBS) ve ekran görüntülerinde de görünsün. Yalnızca "adaptive" modda:
    # yazı İngilizce altyazının üstünde, okunan bölgenin dışında durur; OCR kendi yazımızı okumaz.
    # "cover" kutusu okunan yazının üstünde olduğu için her zaman yakalamadan gizlenir.
    capture_visible: bool = True
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

    # Kısayollar: işlem -> "ctrl+alt+r" (menüden değiştirilir; boş = atanmamış).
    hotkeys: dict = field(default_factory=lambda: dict(DEFAULT_HOTKEYS))
    # Başlatınca ayar menüsü açılsın.
    show_menu_on_start: bool = True

    # Oturum kaydı: görülen her satır (İngilizce + Türkçe) kayitlar/ klasörüne yazılır.
    session_log: bool = True
    session_log_dir: str = "kayitlar"

    # Dublaj: Türkçe satırı konuşan karakterin sesine uygun sesle okur (menüden / Ctrl+Alt+D anında).
    tts_enabled: bool = False
    # "edge": Microsoft Edge'in çevrimiçi doğal sesleri (Emel kadın, Ahmet erkek; ücretsiz, internet gerekir;
    # ulaşılamazsa Windows sesine düşer). "windows": yerleşik çevrimdışı ses.
    dub_engine: str = "edge"
    # Konuşanı oyun sesinden tanı: kadın/erkek ve karakter hafızası (aynı karaktere hep aynı ses).
    dub_detect_speaker: bool = True
    # Konuşanı tanımak için altyazı belirdikten sonra en az bu kadar ms ses dinlenir.
    dub_analysis_ms: int = 600
    characters_file: str = "karakter_sesleri.json"
    tts_voice: str = ""          # Windows sesi (boş: ilk Türkçe ses)
    tts_rate: float = 1.1        # 0.6 - 2.0
    tts_volume: float = 1.0      # 0.0 - 1.0

    # Çeviri önbelleği (aynı satırı tekrar çevirip DeepL kotası harcamamak için).
    cache_file: str = "ceviri_onbellek.jsonl"

    def extra(self, name: str) -> ExtraRegion | None:
        return next((r for r in self.extra_regions if r.name == name), None)

    def set_extra(self, name: str, region: Region | None) -> None:
        """Ek bölgeyi ekler/günceller; None verilirse kaldırır."""
        old = self.extra(name)
        self.extra_regions = [r for r in self.extra_regions if r.name != name]
        if region is not None:
            self.extra_regions.append(ExtraRegion(name, region, old.display_mode if old else "cover",
                                                  old.enabled if old else True))

    def resolved_auth_key(self) -> str:
        return self.deepl_auth_key.strip() or os.environ.get("DEEPL_AUTH_KEY", "").strip()


def _region(value: dict) -> Region:
    known = {f.name for f in fields(Region)}
    return Region(**{k: float(v) for k, v in value.items() if k in known})


# config.json yalnızca kullanıcının değiştirdiği ayarları saklar; varsayılanlar koddan gelir, böylece
# yeni sürümlerdeki iyileştirilmiş varsayılanlar mevcut kurulumlara da uygulanır.
CONFIG_VERSION = 2
# Sürüm 1 dosyaları TÜM ayarları yazıyordu. Bu anahtarlar eski bir varsayılana eşitse kullanıcı
# seçimi değil, donmuş eski varsayılandır: atılır ki yeni varsayılan geçerli olsun.
_OLD_DEFAULTS = {
    "preprocess": ["none", "outline"],
    "capture_interval_ms": [120, 50],
    "ollama_model": ["gemma3:12b"],
    "ollama_use_context": [True],
    "final_translator": ["claude", "deepl"],
}


def _from_dict(data: dict) -> Config:
    if data.get("config_version", 1) < 2:
        data = {k: v for k, v in data.items() if v not in _OLD_DEFAULTS.get(k, [])}
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
                ExtraRegion(str(item["name"]), _region(item["region"]), str(item.get("display_mode", "cover")),
                            bool(item.get("enabled", True)))
                for item in value
                if isinstance(item, dict) and "name" in item and "region" in item
            ]
        elif key == "hotkeys":
            cfg.hotkeys = merged_hotkeys(value if isinstance(value, dict) else {})
        else:
            setattr(cfg, key, value)
    return cfg


def load(path: Path) -> Config:
    if not path.exists():
        cfg = Config()
        save(cfg, path)
        return cfg
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = _from_dict(data)
    if data.get("config_version", 1) < CONFIG_VERSION:
        save(cfg, path)  # eski biçimi bir kez yeni (yalnızca değişiklikler) biçimine çevir
    return cfg


def save(cfg: Config, path: Path) -> None:
    defaults = asdict(Config())
    changed = {k: v for k, v in asdict(cfg).items() if v != defaults[k]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump({"config_version": CONFIG_VERSION, **changed}, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
