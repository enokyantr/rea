"""Altyazı bölgesinden metin okuma: ön işleme seçenekleri + en anlamlı OCR sonucunu seçme.

Tek bir ön işleme her sahnede işe yaramıyor (gerçek video: gri ton koyu sahnede kusursuz, açık
sahnede t→c/h→ij hatalı; sıkı kontur maskesi sıkıştırılmış videoda harfleri parçalıyor). Bu yüzden
"auto" modunda birkaç seçenek (kontur, parlak, gri) OCR'lanır ve gerçek metne en çok benzeyen sonuç seçilir. OCR satır
başına ~5-20 ms olduğu için bu ucuzdur; ilk seçenek yeterince iyiyse diğerleri hiç denenmez."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import imageproc, textproc

# İlk seçenek bu kaliteye ulaşırsa diğerleri denenmez (gecikme ve işlemci tasarrufu).
GOOD_ENOUGH = 0.97


@dataclass
class ReadResult:
    text: str
    variant: str = ""
    # (seçenek, okunan metin, puan) — --debug/--dump için
    candidates: list[tuple[str, str, float]] = field(default_factory=list)
    crop: np.ndarray | None = None
    # Motor bildiriyorsa (RapidOCR): okumanın güveni (0..1) ve yazının yatay ortası (girdi
    # görüntüsünün sol kenarına göre piksel).
    confidence: float | None = None
    center_x: float | None = None
    # Bulunan yazının kutusu (x0, y0, x1, y1), girdi görüntüsü koordinatlarında. Değişim tespiti
    # için: yalnızca bu alan değişince yeniden okunur.
    text_box: tuple[int, int, int, int] | None = None
    # Yoksayılan ifade çıkarıldıysa okunan asıl metin (`text` kalan kısımdır, boş olabilir).
    ignored: str = ""


def variants(gray: np.ndarray, cfg, mode: str | None = None) -> list[tuple[str, np.ndarray]]:
    """Denenecek OCR girdileri, sırayla. `cfg.preprocess`: auto | outline | bright | none | tophat.
    "all" (yalnızca --ocr-compare): hepsi."""
    mode = mode or cfg.preprocess
    out: list[tuple[str, np.ndarray]] = []
    # Sıra önemli: ilk okuma yeterince iyiyse diğerleri denenmez. Gerçek kayıtlarda (--ocr-compare,
    # 300 kare) en iyi birleşim kontur + parlak + gri; top-hat katkı vermedi ve en yavaşıydı.
    if mode in ("auto", "outline", "all"):
        mask = imageproc.text_mask(gray, "outline", cfg.bright_threshold, cfg.dark_threshold,
                                   cfg.outline_radius, cfg.outline_white_threshold)
        out.append(("kontur", imageproc.for_ocr(gray, mask, "outline")))
    if mode in ("auto", "bright", "all"):
        out.append(("parlak", imageproc.for_ocr(gray, gray >= cfg.bright_threshold, "bright")))
    if mode in ("auto", "none", "all"):
        out.append(("gri", gray))
    if mode in ("tophat", "all"):
        out.append(("tophat", imageproc.tophat(gray, cfg.tophat_radius)))
    return out


# Windows OCR küçük ve bulanık yazıda harfleri bölüyor ("some" -> "sorrje", "dick" -> "click").
# Yazı satırı bu yükseklikten (piksel) kısaysa OCR'dan önce daha çok büyütülür.
TARGET_TEXT_HEIGHT = 40
MAX_UPSCALE = 4.0


def upscale_factor(crop_height: int, cfg) -> float:
    """`crop_height`: yazı kutusunun yüksekliği (bright_bbox kenar payı dahil, 2 x 8 px)."""
    text_h = crop_height - 16
    if text_h <= 0:
        return cfg.upscale
    return max(cfg.upscale, min(MAX_UPSCALE, TARGET_TEXT_HEIGHT / text_h))


def read_text(gray: np.ndarray, recognize: Callable[[np.ndarray], list[str]], cfg,
              raw: bool = False, engine=None, retry_empty: bool = True) -> ReadResult:
    """`gray`: bölgenin gri tonu. `recognize`: görüntü -> satırlar. `raw`: motor (RapidOCR) doğal
    görüntüyü olduğu gibi ister; ön işleme seçenekleri ve büyütme uygulanmaz.
    `retry_empty=False`: ham okuma hiç yazı bulamadıysa süzgeçli ek okumalar yapılmaz (çağıran,
    yazısız ama parlak sahnede her karede üç OCR çalışmasın diye kısa süre kapatır)."""
    bbox = imageproc.bright_bbox(gray >= cfg.bright_threshold)
    left = top = 0
    if bbox is None:
        if cfg.skip_without_bright_text:
            return ReadResult("")
    else:
        top, bottom, left, right = bbox
        gray = gray[top:bottom, left:right]

    result = ReadResult("", crop=gray)
    factor = 1.0 if raw else upscale_factor(gray.shape[0] if bbox is not None else 0, cfg)
    best_score = None
    best_engine_state = None
    options = [("ham", lambda: gray)] if raw else [(n, (lambda i=i: i)) for n, i in variants(gray, cfg)]
    i = 0
    while i < len(options):
        name, make = options[i]
        i += 1
        text = textproc.normalize(recognize(imageproc.upscale(make(), factor)))
        score = textproc.reading_score(text)
        result.candidates.append((name, text, score))
        state = _engine_state(engine) if raw else None
        # Uzun ve temiz okuma kazanır; kısmi okuma ("Go to") tam okumayı yenemez.
        if best_score is None or score > best_score:
            best_score, result.text, result.variant, best_engine_state = score, text, name, state
        if (raw and name == "ham" and state is not None and state[0] is not None
                and state[0] < getattr(cfg, "ocr_retry_confidence", LOW_CONFIDENCE) and (text or retry_empty)):
            # Düşük kaliteli akış (chiaki-ng / zayıf bağlantı): motor emin değil. Bulanık yazıda
            # keskinleştirme, bloklu/kumlu yazıda medyan süzgeci okumayı belirgin düzeltiyor.
            options += [("keskin", lambda: imageproc.sharpen(gray)), ("medyan", lambda: imageproc.median3(gray))]
            continue
        if (textproc.text_quality(text) >= GOOD_ENOUGH and text.isascii()
                and len(text) >= cfg.min_text_chars):
            break
    if not textproc.is_plausible(result.text, cfg.min_text_chars):
        result.text = ""
    # Süzgeçli görüntüden çıkan anlamsız okuma ("Selitiyphamaykaot."): motor emin değil ve
    # kelimeler sözlükte yok -> yazı sayılmaz (çevrilip ekrana basılmasın).
    if (raw and result.text and best_engine_state is not None and best_engine_state[0] is not None
            and best_engine_state[0] < GARBAGE_CONFIDENCE and textproc.known_fraction(result.text) < 0.3):
        result.text = ""
    if raw and best_engine_state is not None:
        result.confidence, span, box = best_engine_state
        if span is not None:
            result.center_x = left + (span[0] + span[1]) / 2
        # Yoksayılan satırın da kutusu tutulur: değişim tespiti o yazı ekranda kaldıkça yeniden okumaz.
        if box is not None and result.text:
            result.text_box = (left + box[0], top + box[1], left + box[2], top + box[3])
    # Yoksayılan ifadeler: tüm bölgeler için tek yer; çeviriye, ekrana ve dublaja hiç ulaşmaz.
    kept = textproc.ignore_phrases(result.text, cfg.ignore_phrases, cfg.min_text_chars)
    if kept != result.text:
        result.ignored = result.text
        result.text = kept
    return result


# RapidOCR'ın en düşük kutu güveni bunun altındaysa görüntünün süzülmüş hâlleri de okunur
# (varsayılan; config.ocr_retry_confidence ile değişir).
LOW_CONFIDENCE = 0.90
# Bu güvenin altındaki okuma, kelimeleri sözlükte de yoksa atılır.
GARBAGE_CONFIDENCE = 0.70


def _engine_state(engine):
    if engine is None:
        return None
    return (getattr(engine, "last_confidence", None), getattr(engine, "last_span", None),
            getattr(engine, "last_box", None))
