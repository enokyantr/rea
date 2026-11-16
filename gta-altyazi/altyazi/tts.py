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
from collections import deque
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


def _duration(audio: bytes, ext: str) -> float:
    """Ses süresi (sn): WAV başlığından; MP3 için Edge'in 48 kbit/sn akışından tahmin."""
    if ext == ".wav" and audio[:4] == b"RIFF":
        import io
        import wave

        try:
            with wave.open(io.BytesIO(audio)) as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass
    return len(audio) * 8 / 48000


def _pct(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{int(round(value))}%"


def _hz(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{int(round(value))}Hz"


class _EdgeStream:
    """Edge sesinin inen MP3 parçaları (üretici: arka plan iş parçacığı, tüketici: çalıcı).
    Yineleyici olarak dolaşılır; `None` sonu belirtir. `close()` hem üreticiyi hem tüketiciyi bırakır."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self.first = threading.Event()   # ilk ses parçası geldi ya da akış bitti
        self.closed = False
        self.nbytes = 0
        self.error: Exception | None = None

    def put(self, data: bytes) -> None:
        if not self.closed and data:
            self.nbytes += len(data)
            self._q.put(data)
            self.first.set()

    def end(self, error: Exception | None = None) -> None:
        self.error = error
        self._q.put(None)
        self.first.set()

    def close(self) -> None:
        self.closed = True
        self._q.put(None)

    def __iter__(self):
        while True:
            data = self._q.get()
            if data is None:
                return
            yield data

    def read_all(self) -> bytes:
        return b"".join(self)


def _discard(job) -> None:
    """Kullanılmayacak Edge akışını (geldiğinde) kapat: indirme boşuna sürmesin."""
    def close(done):
        try:
            result = done.result()
        except Exception:
            return
        if isinstance(result, tuple) and result:
            result = result[0]
        if isinstance(result, _EdgeStream):
            result.close()
    job.add_done_callback(close)


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
        self._stream = None          # StreamPlayer (miniaudio): inerken çalar; yoksa MediaPlayer
        self._handle = None          # çalan akış konuşması (PlayHandle)
        self._feeding: _EdgeStream | None = None
        self._fast = None            # EdgeFast: hazır bağlantılar (ilk ses ~0,1 sn); yoksa normal edge-tts
        self._windows_voice = None
        self._edge_fail = 0
        self._edge_off_until = 0.0
        self._gemini_off_until = 0.0
        self._gemini_fail = 0
        self._gemini_model = 0
        self._pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="dublaj-ses")
        # Satır sesleri altyazı gelir gelmez (sıra beklemeden) bununla hazırlanır.
        self._prep = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dublaj-hazirlik")
        self._free_at = 0.0       # sıradaki satırların tahmini bittiği an (hız planı için)
        self._cps = 15.0          # Türkçe ses hızı (harf/sn, hız x1'de); çalınan satırlarla ayarlanır
        self._last_played = None  # (PlayHandle, harf sayısı, hız): süre ölçümü için
        import requests

        self._http = requests.Session()
        self.last_engine = ""
        self.fallbacks = 0
        self._prepared: dict = {}                 # satır anahtarı -> konuşan tanıma (Future)
        self._gemini_times: deque = deque(maxlen=8)  # son Gemini cevap süreleri (sn)
        self._gemini_skips = 0
        self.debug = False
        self.lag_s = 0.0                          # son satırın altyazıdan ne kadar geç başladığı
        # MediaPlayer yolunda okunan sesin tahmini bitişi (time.monotonic).
        self._file_until = 0.0
        self._tmp = Path(tempfile.gettempdir())
        self._file_index = 0
        self.voice_in_use = ""
        self.last_character = ""
        self.error: str | None = None
        self.ready = threading.Event()

    # --- arayüz (iş parçacığı güvenli) -----------------------------------------------------

    def prepare(self, key, onset: float) -> None:
        """Altyazı belirdi: konuşan tanımayı HEMEN başlat (çeviriyle paralel). `say` aynı `key` ile
        gelince sonuç büyük olasılıkla hazırdır."""
        if self.book is None or key in self._prepared:
            return
        self._prepared[key] = self._pool.submit(self._detect, onset)
        while len(self._prepared) > 8:
            self._prepared.pop(next(iter(self._prepared)))

    def say(self, text: str, onset: float | None = None, original: str = "", key=None) -> None:
        """Satırı sıraya ekle ve sesini HEMEN hazırlamaya başla (önceki satırlar çalarken).
        Satırlar yarıda kesilmez; geride kalındıkça okuma hızlanır (en çok +%45), çok geride kalan
        (dub_max_lag_ms) satır, arkasında yenisi varsa atlanır."""
        now = time.monotonic()
        onset = onset if onset is not None else now
        factor = self._plan(text, onset, self._jobs.qsize(), now)
        prepared = self._prep.submit(self._prepare, text, onset, original, key, None, factor)
        self._jobs.put((text, onset, self._generation, None, original, key, now, prepared))

    def _plan(self, text: str, onset: float, pending: int, now: float) -> float:
        """Hız katsayısı: satırın ne kadar geç başlayacağı tahminine göre (önceki satırlar + ses
        hazırlığı ~1 sn). Hedef: ses altyazıdan en fazla ~1,5 sn sonra başlasın."""
        start = max(now + 1.0, self._free_at)
        expected_lag = start - onset
        factor = 1 + min(0.45, 0.12 * min(pending, 2) + max(0.0, expected_lag - 1.5) * 0.25)
        rate = self.cfg.tts_rate * factor
        self._free_at = start + len(text) / (self._cps * max(0.5, rate)) + 0.1
        return factor

    def interrupt(self) -> None:
        """Her şeyi sustur ve sırayı boşalt (duraklatma, kapatma)."""
        self._generation += 1
        self.busy_until = time.monotonic()
        self._free_at = 0.0
        try:
            while True:
                job = self._jobs.get_nowait()
                if job is not None and len(job) > 7 and job[7] is not None:
                    _discard(job[7])
        except queue.Empty:
            pass
        feeding, self._feeding = self._feeding, None
        if feeding is not None:
            feeding.close()
        if self._stream is not None:
            self._stream.stop()
        player = self._player
        if player is not None:
            try:
                player.pause()
            except Exception:
                pass

    @property
    def busy_until(self) -> float:
        """Kendi dublajımızın bittiği/biteceği an: sokak dinleyicisi kendi sesimizi duymasın."""
        handle = self._handle
        if handle is None:
            return self._file_until
        if not handle.done:
            return max(self._file_until, time.monotonic() + 0.5)
        return max(self._file_until, (handle.finished_at or 0.0) + 0.15)

    @busy_until.setter
    def busy_until(self, value: float) -> None:
        self._file_until = value

    def playing(self) -> bool:
        """Akış çalıcıda hâlâ ses çalıyor mu (sokak dinleyicisi kendi dublajımızı duymasın)."""
        handle = self._handle
        return handle is not None and not handle.done

    def stop(self) -> None:
        self.interrupt()
        self._jobs.put(None)

    def sample(self, gender: str, pitch_pct: int = 0, rate_pct: int = 0, voice: int = 0) -> None:
        """Menüden "Dinle": bu karakterin sesiyle örnek cümle."""
        self._jobs.put(("Merhaba, bu benim sesim.", time.monotonic(), self._generation,
                        Delivery(gender, voice, pitch_pct, rate_pct), "", None, time.monotonic(), None))

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
        try:
            from .stream_player import StreamPlayer, StreamPlayerError

            try:
                self._stream = StreamPlayer(log=self._log)
            except StreamPlayerError as exc:
                self._log(f"Hızlı ses çalıcı açılamadı ({exc}); dublaj eski yolla (biraz daha gecikmeli) çalınacak.")
        except Exception as exc:
            self._log(f"Hızlı ses çalıcı yüklenemedi ({exc}); guncelle.bat çalıştırın.")
        if self.cfg.dub_engine in ("edge", "gemini"):
            try:
                from .edge_fast import EdgeFast

                self._fast = EdgeFast(log=self._log)
            except Exception as exc:
                self._log(f"Edge hızlı bağlantı kullanılamıyor ({exc}); normal bağlantıyla devam.")
        if self.cfg.dub_engine == "gemini" and self.cfg.gemini_api_key:
            self.voice_in_use = "Karma: Gemini (duygulu), sorun olursa anında Edge"
        elif self.cfg.dub_engine in ("edge", "gemini"):
            self.voice_in_use = "Edge (doğal Türkçe)" + (" — Gemini anahtarı yok" if self.cfg.dub_engine == "gemini" else "")
        else:
            self.voice_in_use = self._windows_voice.display_name if self._windows_voice else ""
        if self._stream is None and self.cfg.dub_engine in ("edge", "gemini"):
            self.voice_in_use += " · eski çalıcı (gecikmeli; guncelle.bat çalıştırın)"
        self.ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            try:
                self._speak_line(*job)
            except Exception as exc:  # bir satırın hatası dublajı durdurmasın
                self._log(f"Dublaj hatası: {exc}")
        if self._stream is not None:
            self._stream.close()
        if self._fast is not None:
            self._fast.close()

    def _prepare(self, text: str, onset: float, original: str, key, forced: Delivery | None,
                 factor: float):
        """Konuşan tanıma + ses hazırlığı (sıradan bağımsız, havuzda). Edge sesi akış olarak döner."""
        t0 = time.monotonic()
        delivery = forced or self._delivery(key, onset, original)
        t_detect = time.monotonic()
        rate = min(2.0, self.cfg.tts_rate * (1 + delivery.rate_pct / 100) * factor)
        audio, ext, source = self._synthesize(text, delivery, rate)
        return audio, ext, source, delivery, rate, t_detect - t0, time.monotonic() - t_detect

    def _speak_line(self, text: str, onset: float, generation: int, forced: Delivery | None, original: str,
                    key, queued_at: float, prepared=None) -> None:
        if generation != self._generation:
            if prepared is not None:
                _discard(prepared)
            return  # duraklatıldı / kapatıldı
        pending = self._jobs.qsize()
        if prepared is None:  # sıraya say() dışında girdi (örnek ses, testler): şimdi hazırla
            factor = 1 + min(0.45, 0.12 * min(pending, 2) + max(0.0, time.monotonic() - onset - 1.5) * 0.25)
            prepared_now = self._prepare(text, onset, original, key, forced, factor if forced is None else 1.0)
        else:
            prepared_now = None
        # Önceki satır bitene kadar bekle (bu satırın sesi o çalarken hazırlandı): satırlar kesilmez.
        t_wait = time.monotonic()
        while generation == self._generation and (time.monotonic() < self._file_until or self.playing()):
            time.sleep(0.01)
        waited = time.monotonic() - t_wait
        self._learn_speed()
        pending = self._jobs.qsize()
        max_lag = self.cfg.dub_max_lag_ms / 1000
        late = time.monotonic() - onset
        if generation != self._generation or (forced is None and pending and late > max_lag):
            if prepared is not None:
                _discard(prepared)
            elif prepared_now and isinstance(prepared_now[0], _EdgeStream):
                prepared_now[0].close()
            self._prepared.pop(key, None)
            if self.debug and generation == self._generation:
                self._log(f"dublaj: geride kalan satır atlandı ({1000 * late:.0f} ms): {text}")
            return
        if prepared_now is None:
            try:
                prepared_now = prepared.result(timeout=8.0)
            except Exception as exc:
                self._log(f"Dublaj sesi hazırlanamadı: {exc}")
                return
        audio, ext, source, delivery, rate, detect_s, synth_s = prepared_now
        if generation != self._generation:
            if isinstance(audio, _EdgeStream):
                audio.close()
            return
        if audio:
            self.last_engine = source
            self._play_audio(audio, ext)
            if self._handle is not None and self._stream is not None:
                self._last_played = (self._handle, len(text), rate)
        else:
            self.last_engine = "Windows"
            self.busy_until = time.monotonic() + 0.075 * len(text) / max(0.5, rate) + 0.15
            self._play_windows(text, delivery, rate)
        if self.debug and self._handle is not None and audio:
            self._handle.wait_started(1.0)  # gecikme ilk sesin aygıta verildiği andan ölçülsün
        self.lag_s = time.monotonic() - onset
        # Gerçek başlama tahminden geçse sonraki satırların planı da kaysın.
        self._free_at = max(self._free_at, time.monotonic() + len(text) / (self._cps * max(0.5, rate)))
        if self.debug and forced is None:
            self._log(f"dublaj: altyazıdan {1000 * self.lag_s:.0f} ms sonra başladı · tanıma "
                      f"{1000 * detect_s:.0f} ms · ses {1000 * synth_s:.0f} ms ({self.last_engine})"
                      f" · önceki satırı bekleme {1000 * waited:.0f} ms · sıradaki {pending}"
                      f" · hız x{rate / max(0.1, self.cfg.tts_rate):.2f}")

    def _learn_speed(self) -> None:
        """Biten satırın gerçek süresiyle harf/sn tahminini güncelle (hız planı doğru olsun)."""
        last, self._last_played = self._last_played, None
        if last is None:
            return
        handle, chars, rate = last
        if self.debug and handle.started_at and handle.finished_at:
            self._log(f"dublaj: önceki satır {1000 * (handle.finished_at - handle.started_at):.0f} ms çaldı "
                      f"(ses {1000 * handle.duration_s:.0f} ms, kırpılan sessizlik {1000 * handle.trimmed_s:.0f} ms, "
                      f"{chars / max(0.1, handle.duration_s * max(0.5, rate)):.1f} harf/sn)")
        if handle.done and handle.error is None and handle.duration_s > 0.5 and chars >= 8:
            cps = chars / (handle.duration_s * max(0.5, rate))
            if 6 < cps < 30:
                self._cps = 0.7 * self._cps + 0.3 * cps

    def _synthesize(self, text: str, delivery: Delivery, rate: float) -> tuple[bytes | None, str, str]:
        now = time.monotonic()
        engine = self.cfg.dub_engine
        use_gemini = engine == "gemini" and bool(self.cfg.gemini_api_key) and now >= self._gemini_off_until
        use_edge = engine in ("edge", "gemini") and now >= self._edge_off_until
        wait = self.cfg.dub_gemini_wait_ms / 1000
        if use_gemini and len(self._gemini_times) >= 4:
            typical = sorted(self._gemini_times)[len(self._gemini_times) // 2]
            if typical > wait * 1.1:
                # Gemini genelde geç kalıyor: beklemeden Edge; arada bir yine dene (süresi ölçülsün).
                self._gemini_skips += 1
                if self._gemini_skips % 5:
                    use_gemini = False
        # Karma: Gemini (duygulu) ile Edge AYNI ANDA istenir; Gemini zamanında gelmezse ya da hata
        # verirse beklemeden hazır Edge sesi çalınır, o da yoksa Windows sesi.
        edge_job = self._pool.submit(self._edge_open, text, delivery, rate) if use_edge else None
        audio, ext, source = None, "", ""
        if use_gemini:
            gem_job = self._pool.submit(self._timed_gemini, text, delivery)
            try:
                audio = gem_job.result(timeout=wait)
            except FutureTimeout:
                self.fallbacks += 1
                source = "Edge (Gemini geç kaldı)"
            except Exception:
                self.fallbacks += 1
                source = "Edge (Gemini hata)"
            if audio:
                ext, source = ".wav", "Gemini"
                if edge_job is not None:
                    _discard(edge_job)
                return audio, ext, source
        if edge_job is not None:
            try:
                audio = edge_job.result(timeout=5.0)
            except Exception:
                _discard(edge_job)
                audio = None
            if audio:
                ext, source = ".mp3", source or "Edge"
        return audio, ext, source

    def _timed_gemini(self, text: str, d: Delivery) -> bytes | None:
        t0 = time.monotonic()
        try:
            return self._gemini_bytes(text, d)
        finally:
            self._gemini_times.append(time.monotonic() - t0)

    def _delivery(self, key, onset: float, original: str) -> Delivery:
        """Karakter (önceden başlatılmış tanımadan) + bu satırın son metnine göre duygu."""
        job = self._prepared.pop(key, None)
        try:
            ch, level, f0 = job.result(timeout=1.5) if job is not None else self._detect(onset)
        except Exception:
            ch, level, f0 = None, None, 0.0
        if ch is None:
            mood = emo.classify(original, None)
            rate, pitch, volume = mood.prosody()
            self.last_character = f"tanınamadı (konuşma sesi yok/kısa) · {mood.describe()}"
            return Delivery("erkek", 0, pitch, rate, volume, "", mood.label, mood.intensity)
        mood = emo.classify(original, level)
        rate, pitch, volume = mood.prosody(ch.expressive)
        self.last_character = (f"{ch.label} ({'kadın' if ch.gender == 'kadin' else 'erkek'}, {f0:.0f} Hz) · "
                               f"{mood.describe()}")
        return Delivery(ch.gender, ch.voice, ch.pitch_pct + pitch, ch.rate_pct + rate, volume, "",
                        mood.label, min(1.0, mood.intensity * ch.expressive))

    def _detect(self, onset: float):
        """Konuşanı oyun sesinden bul: (karakter, uyarılmışlık, perde) ya da (None, None, 0).
        Hafızadaki karakter genelde ilk EARLY_S saniyede; emin olunamazsa dub_analysis_ms dinlenir."""
        if not (self.cfg.dub_detect_speaker and self.recorder is not None and self.book is not None
                and self.recorder.error is None):
            return None, None, 0.0
        early, full = onset + EARLY_S, onset + self.cfg.dub_analysis_ms / 1000
        feats = None
        while True:
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
            return None, None, 0.0
        known, dist = self.book.best(feats)
        level = (emo.arousal(feats.loud_db, feats.f0, known.loud_db, known.f0)
                 if known is not None and dist <= self.book.MATCH else None)
        return self.book.identify(feats), level, feats.f0

    # --- Edge (çevrimiçi) ----------------------------------------------------------------------

    def _edge_open(self, text: str, d: Delivery, rate: float) -> _EdgeStream | None:
        """Edge sesini akış olarak başlatır; ilk ses parçası gelince döner (gerisi inerken çalınır).
        İlk parça gelmezse çok dilli sesten yerli Türkçe sese geçer; o da olmazsa None."""
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
        options = {"rate": _pct((rate - 1) * 100), "pitch": _hz(pitch_hz), "volume": _pct(volume)}
        error: Exception | None = None
        fast = self._fast
        if fast is not None:
            stream = _EdgeStream()
            fast.request(text, voice, options["rate"], options["pitch"], options["volume"],
                         stream.put, stream.end, timeout=6.0 + 0.15 * len(text))
            if stream.first.wait(2.5) and stream.nbytes:
                self._edge_fail = 0
                return stream
            stream.close()  # hazır bağlantı olmadı: aynı satır normal yoldan
        for name in dict.fromkeys([voice, pool[0]]):
            stream = _EdgeStream()
            threading.Thread(target=self._edge_produce, args=(edge_tts, stream, text, name, options),
                             name="dublaj-edge", daemon=True).start()
            if not stream.first.wait(4.0):
                stream.close()
                error = TimeoutError("Edge 4 sn içinde ses göndermedi")
                continue
            if stream.nbytes == 0:
                error = stream.error or RuntimeError("Edge boş ses döndürdü")
                continue
            self._edge_fail = 0
            return stream
        self._edge_fail += 1
        if self._edge_fail >= 3:
            self._edge_off_until = time.monotonic() + self.EDGE_RETRY_S
            self._log(f"Edge seslerine ulaşılamıyor ({error}); {self.EDGE_RETRY_S} sn Windows sesi kullanılacak.")
        return None

    @staticmethod
    def _edge_produce(edge_tts, stream: _EdgeStream, text: str, voice: str, options: dict) -> None:
        """Arka planda: Edge parçalarını indikçe akışa yazar (kendi olay döngüsüyle)."""
        async def fetch() -> None:
            chunks = edge_tts.Communicate(text, voice, **options).stream()
            try:
                async for chunk in chunks:
                    if stream.closed:
                        return
                    if chunk.get("type") == "audio":
                        stream.put(chunk["data"])
            finally:
                await chunks.aclose()  # bağlantı düzgün kapansın (yarıda bırakılınca uyarı vermesin)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(fetch(), timeout=6.0 + 0.15 * len(text)))
            stream.end()
        except Exception as exc:
            stream.end(exc)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

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

    def _play_audio(self, audio, ext: str) -> None:
        """Hazır sesi çal: akış çalıcı varsa Edge inerken çalınır; yoksa dosya + MediaPlayer."""
        if self._stream is not None:
            from .stream_player import StreamPlayerError

            try:
                if isinstance(audio, _EdgeStream):
                    self._feeding = audio
                    self._handle = self._stream.play_chunks(audio)
                else:
                    self._handle = self._stream.play_wav(audio) if ext == ".wav" else \
                        self._stream.play_chunks([audio])
                return
            except StreamPlayerError as exc:
                self._log(f"Hızlı ses çalıcı kullanılamadı ({exc}); eski yola dönülüyor.")
                self._stream = None
        if isinstance(audio, _EdgeStream):
            audio = audio.read_all()
        self.busy_until = time.monotonic() + _duration(audio, ext) + 0.15
        self._play_file(self._write_temp(audio, ext))

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
