"""Dublaj: Türkçe satırı konuşan karakterin sesine uygun (kadın/erkek, karaktere özgü) sesle okur.

Akış (her satır, arka plan iş parçacığında):
1. Altyazı belirdiği andan itibaren oyun sesinin kısa bir parçası dinlenir (audio.LoopbackRecorder).
2. voiceid: perde ve ses renginden kadın/erkek ve hafızadaki karakter bulunur (yoksa yeni karakter).
3. Ses: "edge" (Microsoft Edge'in çevrimiçi doğal sesleri: Emel kadın, Ahmet erkek; ücretsiz, internet
   gerekir) ya da "windows" (yerleşik çevrimdışı ses; kadın sesi perde yükseltilerek). Edge'e
   ulaşılamazsa o satır Windows sesiyle okunur.
4. Yeni satır gelince eski okuma hemen kesilir; geç kalan eski satır hiç okunmaz."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

from dataclasses import dataclass

from . import emotion as emo
from .voiceid import CharacterBook, features

# Cinsiyete göre Edge ses havuzu; karakterler sırayla farklı ses alır (Character.voice).
# "turkce": yalnızca Türkçe sesler (Emel, Ahmet; karakterler perde/hızla ayrılır).
# "genis": Microsoft'un çok dilli sesleri de (metnin dilini kendileri algılar; Türkçe telaffuzları
# yerli seslerden farklı/aksanlı olabilir). İlk karakter her zaman yerli Türkçe sesi alır.
EDGE_VOICES_TR = {"kadin": ["tr-TR-EmelNeural"], "erkek": ["tr-TR-AhmetNeural"]}
EDGE_VOICES_WIDE = {
    "kadin": ["tr-TR-EmelNeural", "en-US-AvaMultilingualNeural", "de-DE-SeraphinaMultilingualNeural",
              "en-US-EmmaMultilingualNeural", "fr-FR-VivienneMultilingualNeural"],
    "erkek": ["tr-TR-AhmetNeural", "en-US-AndrewMultilingualNeural", "de-DE-FlorianMultilingualNeural",
              "en-US-BrianMultilingualNeural", "fr-FR-RemyMultilingualNeural", "it-IT-GiuseppeMultilingualNeural",
              "en-AU-WilliamMultilingualNeural"],
}
# Hafızadaki bir karaktere bu kadar yakın ses, ilk EARLY_S saniyede kesin tanınır (beklemeden).
EARLY_S = 0.3
EARLY_MATCH = 0.6  # CharacterBook.MATCH'in oranı


@dataclass
class Delivery:
    """Bir satırın nasıl okunacağı: ses, karaktere özgü ayar + duygu."""
    gender: str = "erkek"
    voice: int = 0
    pitch_pct: float = 0.0
    rate_pct: float = 0.0
    volume_pct: float = 0.0
    note: str = ""
    emotion: str = "neutral"
    intensity: float = 0.0
# Windows sesi tek (genelde erkek, "Tolga"): kadın için perde yükseltilir (Hz).
_WINDOWS_FEMALE_PITCH_HZ = 70
_BASE_F0 = {"kadin": 200, "erkek": 120}


class TtsUnavailable(RuntimeError):
    pass


def turkish_voices() -> list[str]:
    """Kurulu Türkçe Windows seslerinin adları. Yoksa boş liste."""
    try:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
    except Exception:
        return []
    try:
        return [v.display_name for v in SpeechSynthesizer.all_voices if v.language.lower().startswith("tr")]
    except Exception:
        return []


def _pct(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{int(round(value))}%"


def _hz(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{int(round(value))}Hz"


class Speaker(threading.Thread):
    """`say(metin, başlangıç)`: satırı okur. `interrupt()`: o an okunanı keser (yeni satır geldi).
    Ayarlar (motor, hız, ses düzeyi, konuşan tanıma) her satırda `cfg`'den canlı okunur."""

    EDGE_RETRY_S = 120  # Edge art arda başarısızsa bu süre Windows sesiyle devam

    def __init__(self, cfg, book: CharacterBook | None, recorder=None, log: Callable[[str], None] = print):
        super().__init__(daemon=True, name="dublaj")
        self.cfg = cfg
        self.book = book
        self.recorder = recorder
        self._log = log
        self._jobs: queue.Queue = queue.Queue()
        self._generation = 0
        self._player = None
        self._windows_voice = None
        self._edge_fail = 0
        self._edge_off_until = 0.0
        self._gemini_off_until = 0.0
        self._gemini_fail = 0
        self._gemini_model = 0
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="dublaj-ses")
        import requests

        self._http = requests.Session()
        self.last_engine = ""
        self.fallbacks = 0
        self._tmp = Path(tempfile.gettempdir())
        self._file_index = 0
        self.voice_in_use = ""
        self.last_character = ""
        self.error: str | None = None
        self.ready = threading.Event()

    # --- arayüz (iş parçacığı güvenli) -----------------------------------------------------

    def say(self, text: str, onset: float | None = None, original: str = "") -> None:
        """`original`: İngilizce satır (duygu ipuçları için)."""
        try:  # yalnızca en yeni satır okunur
            while True:
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put((text, onset if onset is not None else time.monotonic(), self._generation, None, original))

    def interrupt(self) -> None:
        self._generation += 1
        player = self._player
        if player is not None:
            try:
                player.pause()
            except Exception:
                pass

    def stop(self) -> None:
        self.interrupt()
        self._jobs.put(None)

    def sample(self, gender: str, pitch_pct: int = 0, rate_pct: int = 0, voice: int = 0) -> None:
        """Menüden "Dinle": bu karakterin sesiyle örnek cümle."""
        self._jobs.put(("Merhaba, bu benim sesim.", None, self._generation,
                        Delivery(gender, voice, pitch_pct, rate_pct), ""))

    # --- iş parçacığı ----------------------------------------------------------------------

    def run(self) -> None:
        try:
            from winrt.windows.media.playback import MediaPlayer
            from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        except Exception as exc:
            self.error = f"Dublaj için Windows bileşenleri kurulu değil ({exc}); guncelle.bat çalıştırın."
            self.ready.set()
            return
        self._loop = asyncio.new_event_loop()
        self._synth = SpeechSynthesizer()
        voices = [v for v in SpeechSynthesizer.all_voices if v.language.lower().startswith("tr")]
        wanted = getattr(self.cfg, "tts_voice", "")
        self._windows_voice = next((v for v in voices if v.display_name == wanted), voices[0] if voices else None)
        if self._windows_voice is None and self.cfg.dub_engine not in ("edge", "gemini"):
            self.error = ("Türkçe Windows sesi yok: Ayarlar > Saat ve dil > Konuşma > Ses ekle > Türkçe, "
                          "ya da menüden ses motorunu \"edge\" yapın.")
            self.ready.set()
            return
        self._player = MediaPlayer()
        if self.cfg.dub_engine == "gemini" and self.cfg.gemini_api_key:
            self.voice_in_use = "Karma: Gemini (duygulu), sorun olursa anında Edge"
        elif self.cfg.dub_engine in ("edge", "gemini"):
            self.voice_in_use = "Edge (doğal Türkçe)" + (" — Gemini anahtarı yok" if self.cfg.dub_engine == "gemini" else "")
        else:
            self.voice_in_use = self._windows_voice.display_name if self._windows_voice else ""
        self.ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            try:
                self._speak_line(*job)
            except Exception as exc:  # bir satırın hatası dublajı durdurmasın
                self._log(f"Dublaj hatası: {exc}")

    def _speak_line(self, text: str, onset: float | None, generation: int, forced: Delivery | None,
                original: str) -> None:
        delivery = forced or self._delivery_for_line(onset, generation, original)
        if delivery is None or (forced is None and generation != self._generation):
            return  # beklerken yeni satır geldi
        rate = self.cfg.tts_rate * (1 + delivery.rate_pct / 100)
        now = time.monotonic()
        engine = self.cfg.dub_engine
        use_gemini = engine == "gemini" and bool(self.cfg.gemini_api_key) and now >= self._gemini_off_until
        use_edge = engine in ("edge", "gemini") and now >= self._edge_off_until
        # Karma: Gemini (duygulu) ile Edge AYNI ANDA istenir; Gemini dub_gemini_wait_ms içinde gelmezse
        # ya da hata verirse beklemeden hazır Edge sesi çalınır, o da yoksa Windows sesi.
        edge_job = self._pool.submit(self._edge_bytes, text, delivery, rate) if use_edge else None
        audio, ext, source = None, "", ""
        if use_gemini:
            gem_job = self._pool.submit(self._gemini_bytes, text, delivery)
            try:
                audio = gem_job.result(timeout=self.cfg.dub_gemini_wait_ms / 1000)
            except FutureTimeout:
                self.fallbacks += 1
                source = "Edge (Gemini geç kaldı)"
            except Exception:
                self.fallbacks += 1
                source = "Edge (Gemini hata)"
            if audio:
                ext, source = ".wav", "Gemini"
        if audio is None and edge_job is not None:
            try:
                audio = edge_job.result(timeout=5.0)
            except Exception:
                audio = None
            if audio:
                ext, source = ".mp3", source or "Edge"
        if forced is None and generation != self._generation:
            return
        if audio:
            self.last_engine = source
            self._play_file(self._write_temp(audio, ext))
        else:
            self.last_engine = "Windows"
            self._play_windows(text, delivery, rate)

    def _delivery_for_line(self, onset: float, generation: int, original: str) -> Delivery | None:
        """Konuşanı ve duygusunu bul. Tanınan karakter ilk EARLY_S saniyede hemen; değilse
        dub_analysis_ms kadar dinlenir. Ses yoksa yalnızca metinden duygu, varsayılan erkek ses."""
        detect = (self.cfg.dub_detect_speaker and self.recorder is not None and self.book is not None
                  and self.recorder.error is None)
        if not detect:
            mood = emo.classify(original, None)
            rate, pitch, volume = mood.prosody()
            return Delivery("erkek", 0, pitch, rate, volume, f"duygu: {mood.describe()}", mood.label, mood.intensity)
        early, full = onset + EARLY_S, onset + self.cfg.dub_analysis_ms / 1000
        feats = None
        while True:
            if generation != self._generation or not self._jobs.empty():
                return None
            now = time.monotonic()
            if now >= early:
                feats = features(self.recorder.window(onset - 0.15, now))
                if feats is not None:
                    best, dist = self.book.best(feats)
                    if (best is not None and dist <= self.book.MATCH * EARLY_MATCH) or now >= full:
                        break
            if now >= full:
                break
            time.sleep(0.03)
        if feats is None:
            mood = emo.classify(original, None)
            rate, pitch, volume = mood.prosody()
            self.last_character = f"tanınamadı (konuşma sesi yok/kısa) · {mood.describe()}"
            return Delivery("erkek", 0, pitch, rate, volume, "", mood.label, mood.intensity)
        known, dist = self.book.best(feats)
        # Duygu: bu satırdaki ses, karakterin KENDİ normaline göre (yeni karakterde yalnızca metin).
        level = (emo.arousal(feats.loud_db, feats.f0, known.loud_db, known.f0)
                 if known is not None and dist <= self.book.MATCH else None)
        ch = self.book.identify(feats)
        mood = emo.classify(original, level)
        rate, pitch, volume = mood.prosody(ch.expressive)
        took = int(1000 * (time.monotonic() - onset))
        self.last_character = (f"{ch.label} ({'kadın' if ch.gender == 'kadin' else 'erkek'}, {feats.f0:.0f} Hz) · "
                               f"{mood.describe()} · {took} ms'de tanındı")
        return Delivery(ch.gender, ch.voice, ch.pitch_pct + pitch, ch.rate_pct + rate, volume, "",
                        mood.label, min(1.0, mood.intensity * ch.expressive))

    # --- Edge (çevrimiçi) ----------------------------------------------------------------------

    def _edge_bytes(self, text: str, d: Delivery, rate: float) -> bytes | None:
        """Edge sesi (iş parçacığı havuzunda; kendi olay döngüsüyle)."""
        try:
            import edge_tts
        except ImportError:
            self._edge_off_until = time.monotonic() + 3600
            self._log("edge-tts kurulu değil; Windows sesi kullanılıyor (guncelle.bat).")
            return None
        pitch_hz = _BASE_F0[d.gender] * d.pitch_pct / 100
        pool = (EDGE_VOICES_WIDE if self.cfg.dub_voice_pool == "genis" else EDGE_VOICES_TR)[d.gender]
        voice = pool[d.voice % len(pool)]
        volume = max(-90.0, min(50.0, (self.cfg.tts_volume - 1) * 100 + d.volume_pct))

        async def fetch(name: str) -> bytes:
            comm = edge_tts.Communicate(text, name, rate=_pct((rate - 1) * 100),
                                        pitch=_hz(pitch_hz), volume=_pct(volume))
            data = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    data.extend(chunk["data"])
            return bytes(data)

        loop = asyncio.new_event_loop()
        try:
            try:
                data = loop.run_until_complete(asyncio.wait_for(fetch(voice), timeout=4.0))
            except Exception:
                if voice == pool[0]:
                    raise
                # Çok dilli ses okuyamadı: aynı satır yerli Türkçe sesle.
                data = loop.run_until_complete(asyncio.wait_for(fetch(pool[0]), timeout=4.0))
        except Exception as exc:
            self._edge_fail += 1
            if self._edge_fail >= 3:
                self._edge_off_until = time.monotonic() + self.EDGE_RETRY_S
                self._log(f"Edge seslerine ulaşılamıyor ({exc}); {self.EDGE_RETRY_S} sn Windows sesi kullanılacak.")
            return None
        finally:
            loop.close()
        if not data:
            return None
        self._edge_fail = 0
        return data

    def _write_temp(self, data: bytes, ext: str) -> Path:
        # İki dosya dönüşümlü: çalınmakta olan dosyanın üstüne yazılmaz.
        self._file_index ^= 1
        path = self._tmp / f"gta_altyazi_dublaj_{os.getpid()}_{self._file_index}{ext}"
        path.write_bytes(data)
        return path

    # --- Gemini (çevrimiçi, duygulu) ------------------------------------------------------------

    def _gemini_bytes(self, text: str, d: Delivery) -> bytes | None:
        from . import gemini_tts as gt

        pool = gt.VOICES[d.gender]
        voice = pool[d.voice % len(pool)]
        prompt = gt.style_prompt(text, d.emotion, d.intensity)
        while self._gemini_model < len(gt.MODELS):
            model = gt.MODELS[self._gemini_model]
            try:
                data = gt.synthesize(self._http, self.cfg.gemini_api_key, model, prompt, voice, timeout=6.0)
                self._gemini_fail = 0
                return data
            except gt.GeminiModelMissing:
                self._gemini_model += 1
            except gt.GeminiQuota:
                self._gemini_off_until = time.monotonic() + 600
                self._log("Gemini ücretsiz kotası/hız sınırı doldu: 10 dk Edge sesiyle devam.")
                raise
            except gt.GeminiAuth as exc:
                self._gemini_off_until = time.monotonic() + 86400
                self._log(f"Gemini anahtarı reddedildi ({exc}); Edge sesiyle devam. Menüden anahtarı kontrol edin.")
                raise
            except gt.GeminiError:
                self._gemini_fail += 1
                if self._gemini_fail >= 3:
                    self._gemini_off_until = time.monotonic() + 120
                    self._log("Gemini'ye ulaşılamıyor: 2 dk Edge sesiyle devam.")
                raise
        self._gemini_off_until = time.monotonic() + 86400
        self._log("Gemini ses modeli bulunamadı; Edge sesiyle devam.")
        return None

    def _play_file(self, path: Path) -> None:
        from winrt.windows.foundation import Uri
        from winrt.windows.media.core import MediaSource

        player = self._player
        player.pause()
        player.volume = 1.0  # ses düzeyi Edge isteğinde uygulandı
        player.source = MediaSource.create_from_uri(Uri(path.resolve().as_uri()))
        player.play()

    # --- Windows (çevrimdışı) ------------------------------------------------------------------

    def _play_windows(self, text: str, d: Delivery, rate: float) -> None:
        from winrt.windows.media.core import MediaSource

        if self._windows_voice is None:
            return
        pitch_hz = (_WINDOWS_FEMALE_PITCH_HZ if d.gender == "kadin" else 0) + _BASE_F0[d.gender] * d.pitch_pct / 100
        ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='tr-TR'>"
                f"<voice name='{escape(self._windows_voice.display_name)}'>"
                f"<prosody pitch='{_hz(pitch_hz)}' rate='{max(0.5, min(3.0, rate)):.2f}'>{escape(text)}</prosody>"
                f"</voice></speak>")
        self._synth.voice = self._windows_voice
        stream = self._loop.run_until_complete(self._ssml(ssml))
        player = self._player
        player.pause()
        player.volume = max(0.0, min(1.0, float(self.cfg.tts_volume) * (1 + d.volume_pct / 100)))
        player.source = MediaSource.create_from_stream(stream, stream.content_type)
        player.play()

    async def _ssml(self, ssml: str):
        return await self._synth.synthesize_ssml_to_stream_async(ssml)
