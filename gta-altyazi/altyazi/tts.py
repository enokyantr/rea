"""Sesli okuma (dublaj): Türkçe satırı Windows'un yerleşik Türkçe sesiyle okur (Windows.Media).

Ücretsiz ve yerel. Türkçe ses yoksa: Ayarlar > Saat ve dil > Konuşma > Ses ekle > Türkçe.
Yeni satır gelince önceki okuma kesilir (altyazıyla aynı anda kalsın)."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Callable


class TtsUnavailable(RuntimeError):
    pass


def turkish_voices() -> list[str]:
    """Kurulu Türkçe seslerin adları (Windows). Yoksa boş liste."""
    try:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
    except Exception:
        return []
    try:
        return [v.display_name for v in SpeechSynthesizer.all_voices if v.language.lower().startswith("tr")]
    except Exception:
        return []


class Speaker(threading.Thread):
    """Arka planda okur. `say()` iş parçacığı güvenli; en yeni satır öncekini keser."""

    def __init__(self, voice: str = "", rate: float = 1.15, volume: float = 1.0,
                 log: Callable[[str], None] = print):
        super().__init__(daemon=True, name="tts")
        self._voice_name = voice
        self.rate = rate
        self.volume = volume
        self._log = log
        self._queue: queue.Queue[str | None] = queue.Queue()
        self.voice_in_use = ""
        self.error: str | None = None
        self.ready = threading.Event()

    def say(self, text: str) -> None:
        # Kuyruktaki eski satırları at: yalnızca en yenisi okunur.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(text)

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        try:
            from winrt.windows.media.core import MediaSource
            from winrt.windows.media.playback import MediaPlayer
            from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        except Exception as exc:
            self.error = f"Sesli okuma için Windows bileşenleri kurulu değil ({exc}); guncelle.bat çalıştırın."
            self.ready.set()
            return
        loop = asyncio.new_event_loop()
        synth = SpeechSynthesizer()
        voices = [v for v in SpeechSynthesizer.all_voices if v.language.lower().startswith("tr")]
        chosen = next((v for v in voices if v.display_name == self._voice_name), voices[0] if voices else None)
        if chosen is None:
            self.error = ("Türkçe ses yok: Ayarlar > Saat ve dil > Konuşma > Ses ekle > Türkçe "
                          "(sonra programı yeniden başlatın).")
            self.ready.set()
            return
        synth.voice = chosen
        self.voice_in_use = chosen.display_name
        player = MediaPlayer()
        self.ready.set()
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                try:
                    synth.options.speaking_rate = max(0.5, min(3.0, float(self.rate)))
                except Exception:
                    pass  # eski Windows: hız ayarı yok
                stream = loop.run_until_complete(self._synthesize(synth, text))
                player.pause()
                player.volume = max(0.0, min(1.0, float(self.volume)))
                player.source = MediaSource.create_from_stream(stream, stream.content_type)
                player.play()
            except Exception as exc:  # bir satırın hatası okumayı durdurmasın
                self._log(f"Sesli okuma hatası: {exc}")

    @staticmethod
    async def _synthesize(synth, text: str):
        return await synth.synthesize_text_to_stream_async(text)
