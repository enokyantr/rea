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
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

from .voiceid import CharacterBook, features

EDGE_VOICES = {"kadin": "tr-TR-EmelNeural", "erkek": "tr-TR-AhmetNeural"}
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
        self._tmp = Path(tempfile.gettempdir())
        self._file_index = 0
        self.voice_in_use = ""
        self.last_character = ""
        self.error: str | None = None
        self.ready = threading.Event()

    # --- arayüz (iş parçacığı güvenli) -----------------------------------------------------

    def say(self, text: str, onset: float | None = None) -> None:
        try:  # yalnızca en yeni satır okunur
            while True:
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put((text, onset if onset is not None else time.monotonic(), self._generation))

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

    def sample(self, gender: str, pitch_pct: int = 0, rate_pct: int = 0) -> None:
        """Menüden "Dinle": bu ayarlarla örnek cümle."""
        self._jobs.put(("Merhaba, bu benim sesim.", None, self._generation, (gender, pitch_pct, rate_pct)))

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
        if self._windows_voice is None and self.cfg.dub_engine != "edge":
            self.error = ("Türkçe Windows sesi yok: Ayarlar > Saat ve dil > Konuşma > Ses ekle > Türkçe, "
                          "ya da menüden ses motorunu \"edge\" yapın.")
            self.ready.set()
            return
        self._player = MediaPlayer()
        self.voice_in_use = "Edge (Emel / Ahmet)" if self.cfg.dub_engine == "edge" else \
            (self._windows_voice.display_name if self._windows_voice else "")
        self.ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            try:
                self._handle(*job)
            except Exception as exc:  # bir satırın hatası dublajı durdurmasın
                self._log(f"Dublaj hatası: {exc}")

    def _handle(self, text: str, onset: float | None, generation: int, forced=None) -> None:
        if forced is not None:
            gender, pitch_pct, rate_pct = forced
        else:
            gender, pitch_pct, rate_pct = self._voice_for_line(onset, generation)
            if gender is None:
                return  # beklerken yeni satır geldi
        if generation != self._generation and forced is None:
            return
        rate = self.cfg.tts_rate * (1 + rate_pct / 100)
        path = None
        if self.cfg.dub_engine == "edge" and time.monotonic() >= self._edge_off_until:
            path = self._edge(text, gender, pitch_pct, rate)
        if generation != self._generation and forced is None:
            return
        if path is not None:
            self._play_file(path)
        else:
            self._play_windows(text, gender, pitch_pct, rate)

    def _voice_for_line(self, onset: float, generation: int) -> tuple[str | None, int, int]:
        """Konuşanı bul. Ses yoksa/tanınamazsa erkek, varsayılan ayar."""
        if not (self.cfg.dub_detect_speaker and self.recorder is not None and self.book is not None
                and self.recorder.error is None):
            return "erkek", 0, 0
        # Konuşmanın başından en az dub_analysis_ms kadarını bekle (perde ve renk için).
        until = onset + self.cfg.dub_analysis_ms / 1000
        while time.monotonic() < until:
            if generation != self._generation or not self._jobs.empty():
                return None, 0, 0
            time.sleep(0.02)
        samples = self.recorder.window(onset - 0.15, time.monotonic())
        feats = features(samples)
        if feats is None:
            self.last_character = "tanınamadı (konuşma sesi yok/çok kısa)"
            return "erkek", 0, 0
        ch = self.book.identify(feats)
        self.last_character = f"{ch.label} ({'kadın' if ch.gender == 'kadin' else 'erkek'}, {feats.f0:.0f} Hz)"
        return ch.gender, ch.pitch_pct, ch.rate_pct

    # --- Edge (çevrimiçi) ----------------------------------------------------------------------

    def _edge(self, text: str, gender: str, pitch_pct: int, rate: float) -> Path | None:
        try:
            import edge_tts
        except ImportError:
            self._edge_off_until = time.monotonic() + 3600
            self._log("edge-tts kurulu değil; Windows sesi kullanılıyor (guncelle.bat).")
            return None
        pitch_hz = _BASE_F0[gender] * pitch_pct / 100

        async def fetch() -> bytes:
            comm = edge_tts.Communicate(text, EDGE_VOICES[gender], rate=_pct((rate - 1) * 100),
                                        pitch=_hz(pitch_hz), volume=_pct((self.cfg.tts_volume - 1) * 100))
            data = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    data.extend(chunk["data"])
            return bytes(data)

        try:
            data = self._loop.run_until_complete(asyncio.wait_for(fetch(), timeout=4.0))
        except Exception as exc:
            self._edge_fail += 1
            if self._edge_fail >= 3:
                self._edge_off_until = time.monotonic() + self.EDGE_RETRY_S
                self._log(f"Edge seslerine ulaşılamıyor ({exc}); {self.EDGE_RETRY_S} sn Windows sesi kullanılacak.")
            return None
        if not data:
            return None
        self._edge_fail = 0
        # İki dosya dönüşümlü: çalınmakta olan dosyanın üstüne yazılmaz.
        self._file_index ^= 1
        path = self._tmp / f"gta_altyazi_dublaj_{os.getpid()}_{self._file_index}.mp3"
        path.write_bytes(data)
        return path

    def _play_file(self, path: Path) -> None:
        from winrt.windows.foundation import Uri
        from winrt.windows.media.core import MediaSource

        player = self._player
        player.pause()
        player.volume = 1.0  # ses düzeyi Edge isteğinde uygulandı
        player.source = MediaSource.create_from_uri(Uri(path.resolve().as_uri()))
        player.play()

    # --- Windows (çevrimdışı) ------------------------------------------------------------------

    def _play_windows(self, text: str, gender: str, pitch_pct: int, rate: float) -> None:
        from winrt.windows.media.core import MediaSource

        if self._windows_voice is None:
            return
        pitch_hz = (_WINDOWS_FEMALE_PITCH_HZ if gender == "kadin" else 0) + _BASE_F0[gender] * pitch_pct / 100
        ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='tr-TR'>"
                f"<voice name='{escape(self._windows_voice.display_name)}'>"
                f"<prosody pitch='{_hz(pitch_hz)}' rate='{max(0.5, min(3.0, rate)):.2f}'>{escape(text)}</prosody>"
                f"</voice></speak>")
        self._synth.voice = self._windows_voice
        stream = self._loop.run_until_complete(self._ssml(ssml))
        player = self._player
        player.pause()
        player.volume = max(0.0, min(1.0, float(self.cfg.tts_volume)))
        player.source = MediaSource.create_from_stream(stream, stream.content_type)
        player.play()

    async def _ssml(self, ssml: str):
        return await self._synth.synthesize_ssml_to_stream_async(ssml)
