"""Satırın duygusu ve dublaj tonlaması (platformdan bağımsız, hızlı: < 1 ms).

İki kaynak birleşir:
- Ses (asıl kaynak): konuşanın bu satırdaki ses düzeyi ve perdesi, KENDİ normaline göre. Bağıran
  karakter hem yükselir hem tizleşir (öfke/heyecan); üzgün/korkmuş konuşma alçalır.
- Metin: ünlem, BÜYÜK HARF, küfür, korku/üzüntü/heyecan kelimeleri, üç nokta.
Sonuç: duygu (neutral, excited, angry, sad, scared) + yoğunluk (0..1) -> TTS hız/perde/ses ayarı.
Karakterin "duygu gücü" (expressive) bu ayarı ölçekler: bazı karakterler daha abartılı tepki verir."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NEUTRAL, EXCITED, ANGRY, SAD, SCARED = "neutral", "excited", "angry", "sad", "scared"
LABELS_TR = {NEUTRAL: "sakin", EXCITED: "heyecanlı", ANGRY: "sinirli", SAD: "üzgün", SCARED: "korkmuş"}

# Duygu başına tonlama (tam yoğunlukta): hız %, perde %, ses düzeyi %. Yönler konuşma araştırmasından
# (Murray & Arnott 1993; Juslin & Laukka 2003): öfke ve heyecan daha yüksek perde, yoğunluk ve hız;
# üzüntü alçak, yavaş, kısık ve düz; korku tiz ve hızlı. Oranlar kulakla ayarlanacak başlangıç değerleri.
PROSODY = {
    NEUTRAL: (0, 0, 0),
    EXCITED: (10, 12, 15),
    ANGRY: (8, 7, 35),
    SAD: (-15, -10, -30),
    SCARED: (12, 12, -20),
}

_ANGRY = re.compile(r"\b(shut up|fuck\w*|motherfuck\w*|bitch\w*|asshole|bastard|damn\w*|kill|hate|get out|"
                    r"piece of shit|son of a|i swear|you stupid|idiot|screw you|back off)\b", re.I)
_EXCITED = re.compile(r"\b(yeah|yes+|woo+|wow|hell yeah|let'?s go|holy|amazing|awesome|we did it|"
                      r"oh my god|nice|ha+h?a+|lol|crazy|baby|money)\b", re.I)
_SAD = re.compile(r"\b(sorry|miss (you|him|her)|dead|died|dying|cry\w*|alone|lost|gone|funeral|"
                  r"i can'?t|never again|please forgive|hurts?)\b", re.I)
_SCARED = re.compile(r"\b(help|run|hide|oh no|oh god|please don'?t|don'?t shoot|they'?re coming|scared|"
                     r"what was that|get down|watch out|cops)\b", re.I)


@dataclass(frozen=True)
class Emotion:
    label: str
    intensity: float  # 0..1

    def prosody(self, expressive: float = 1.0) -> tuple[float, float, float]:
        rate, pitch, volume = PROSODY[self.label]
        k = max(0.0, min(1.0, self.intensity)) * expressive
        return rate * k, pitch * k, volume * k

    def describe(self) -> str:
        if self.label == NEUTRAL or self.intensity < 0.15:
            return LABELS_TR[NEUTRAL]
        return f"{LABELS_TR[self.label]} %{round(100 * self.intensity)}"


def text_cues(english: str) -> dict[str, float]:
    """Metindeki duygu işaretleri: duygu -> puan (0..1)."""
    scores = {EXCITED: 0.0, ANGRY: 0.0, SAD: 0.0, SCARED: 0.0}
    if not english:
        return scores
    bangs = english.count("!")
    caps = sum(1 for w in re.findall(r"[A-Za-z]{3,}", english) if w.isupper())
    shout = min(1.0, 0.35 * bangs + 0.3 * caps)
    if _ANGRY.search(english):
        scores[ANGRY] += 0.55 + 0.45 * shout
    if _EXCITED.search(english):
        scores[EXCITED] += 0.5 + 0.4 * shout
    if _SAD.search(english):
        scores[SAD] += 0.55
    if _SCARED.search(english):
        scores[SCARED] += 0.5 + 0.3 * shout
    if english.rstrip().endswith(("...", "…")) and not shout:
        scores[SAD] += 0.25  # tereddüt, yorgunluk
    if shout and max(scores.values()) == 0:
        scores[EXCITED] += 0.4 * shout  # nedeni belirsiz bağırma: heyecan
        scores[ANGRY] += 0.35 * shout
    return {k: min(1.0, v) for k, v in scores.items()}


def arousal(loud_db: float, f0: float, base_loud_db: float, base_f0: float) -> float:
    """Konuşanın bu satırdaki uyarılmışlığı, kendi normaline göre: -1 (alçak, sakin) .. +1 (bağırıyor).
    +6 dB ve +3 yarım ton ~ belirgin yükselme."""
    loud = (loud_db - base_loud_db) / 6.0
    pitch = 12 * math.log2(max(f0, 1) / max(base_f0, 1)) / 3.0
    return max(-1.0, min(1.0, 0.6 * loud + 0.4 * pitch))


def classify(english: str, arousal_value: float | None) -> Emotion:
    """Metin + ses uyarılmışlığından duygu. Ses yoksa yalnızca metin."""
    cues = text_cues(english)
    a = arousal_value if arousal_value is not None else 0.0
    if a > 0.25:
        # Yükselmiş ses: metin hangi yöne işaret ediyorsa (yoksa öfke/heyecan, metne göre)
        boost = (a - 0.25) / 0.75
        if cues[SCARED] > max(cues[ANGRY], cues[EXCITED]):
            cues[SCARED] += 0.6 * boost
        elif cues[ANGRY] >= cues[EXCITED] and (cues[ANGRY] > 0 or "!" in english):
            cues[ANGRY] += 0.7 * boost
        else:
            cues[EXCITED] += 0.6 * boost
    elif a < -0.3:
        cues[SAD] += 0.5 * (-a - 0.3) / 0.7  # alçak, kısık konuşma
        cues[EXCITED] *= 0.5
        cues[ANGRY] *= 0.6
    label, score = max(cues.items(), key=lambda kv: kv[1])
    if score < 0.2:
        return Emotion(NEUTRAL, 0.0)
    return Emotion(label, min(1.0, score))
