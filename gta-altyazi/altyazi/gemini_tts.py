"""Gemini TTS (Google AI Studio, ücretsiz katman): talimatla duygulu Türkçe ses.

Ses, "sinirli, bağırarak" gibi doğal dil talimatıyla tonlanır (Edge yalnızca hız/perde/ses gücü
değiştirebiliyor). Ücretsiz kota ve gecikme garanti değildir: dublaj bu motoru en fazla
`dub_gemini_wait_ms` bekler, gelmezse aynı anda hazırlanan Edge sesi çalınır (karma).

REST: POST {BASE}/models/{model}:generateContent?key=...  -> candidates[0].content.parts[*].inlineData
(base64 PCM; örnekleme hızı mimeType'tan okunur, ör. "audio/L16;codec=pcm;rate=24000")."""

from __future__ import annotations

import base64
import io
import re
import wave

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
# Model adı kaynaklar arasında değişiyor: sırayla denenir, çalışan hatırlanır.
MODELS = ["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]

# Hazır sesler (cinsiyete göre); karakterler sırayla farklı ses alır.
VOICES = {
    "kadin": ["Kore", "Aoede", "Leda", "Zephyr", "Despina", "Callirrhoe", "Autonoe", "Erinome", "Sulafat",
              "Vindemiatrix", "Achernar", "Gacrux", "Pulcherrima", "Laomedeia"],
    "erkek": ["Charon", "Puck", "Fenrir", "Orus", "Iapetus", "Algenib", "Enceladus", "Umbriel", "Algieba",
              "Rasalgethi", "Alnilam", "Schedar", "Achird", "Sadaltager", "Zubenelgenubi", "Sadachbia"],
}

# Duygu -> talimat (yoğunluğa göre "biraz"/"çok").
_STYLE = {
    "neutral": "naturally, in a casual conversational tone",
    "excited": "excitedly, with high energy",
    "angry": "angrily, raising the voice as if shouting",
    "sad": "sadly, quietly and slowly",
    "scared": "fearfully, tense and hurried, almost whispering",
}


class GeminiError(RuntimeError):
    pass


class GeminiQuota(GeminiError):
    """429: ücretsiz kota doldu / hız sınırı."""


class GeminiAuth(GeminiError):
    """401/403: anahtar geçersiz."""


class GeminiModelMissing(GeminiError):
    """404: bu model adı yok."""


def style_prompt(text: str, emotion: str, intensity: float) -> str:
    style = _STYLE.get(emotion, _STYLE["neutral"])
    if emotion != "neutral":
        style = ("very " if intensity >= 0.7 else "slightly " if intensity < 0.35 else "") + style
    return f"Say the following Turkish line {style}, like a voice actor in a crime video game:\n{text}"


def request_body(prompt: str, voice: str) -> dict:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }


def wav_from_response(data: dict) -> bytes:
    """Cevaptaki PCM'i WAV dosyasına çevirir."""
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if not inline or not inline.get("data"):
            continue
        pcm = base64.b64decode(inline["data"])
        mime = inline.get("mimeType") or inline.get("mime_type") or ""
        if mime.startswith("audio/wav") or pcm[:4] == b"RIFF":
            return pcm
        rate = int(m.group(1)) if (m := re.search(r"rate=(\d+)", mime)) else 24000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return buf.getvalue()
    raise GeminiError("cevapta ses yok")


def synthesize(session: requests.Session, api_key: str, model: str, prompt: str, voice: str,
               timeout: float) -> bytes:
    try:
        r = session.post(f"{BASE}/models/{model}:generateContent", params={"key": api_key},
                         json=request_body(prompt, voice), timeout=timeout)
    except requests.RequestException as exc:
        raise GeminiError(f"bağlantı: {type(exc).__name__}") from exc
    if r.status_code == 429:
        raise GeminiQuota("kota/hız sınırı (429)")
    if r.status_code in (401, 403):
        raise GeminiAuth(f"anahtar reddedildi ({r.status_code})")
    if r.status_code == 404:
        raise GeminiModelMissing(f"model yok: {model}")
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", {}).get("message", "")
        except ValueError:
            detail = r.text[:200]
        raise GeminiError(f"HTTP {r.status_code}: {detail[:200]}")
    return wav_from_response(r.json())
