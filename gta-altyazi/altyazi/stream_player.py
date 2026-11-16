"""Düşük gecikmeli dublaj çalıcı: Edge'in MP3 parçalarını inerken çözüp hemen çalar.

Eski yol tüm MP3'ü bekler, geçici dosyaya yazar ve Windows MediaPlayer ile çalar (~0,3-0,8 sn
gecikme). Burada:

1. Çözme (decode_mp3_chunks): MP3 parçaları bir yineleyiciden (başka iş parçacığından beslenen
   engelleyici bir kuyruk olabilir) okunur, miniaudio'nun akış çözücüsüyle 24 kHz mono 16 bit
   PCM'e çevrilir. İlk ses, girdinin tamamı gelmeden üretilir (gerçek akış). Aygıt gerektirmez;
   testler bunu doğrudan sınar.
2. Çalma (StreamPlayer): miniaudio.PlaybackDevice (PortAudio/sounddevice gerekmez) sürekli açık
   tutulur; boştayken sessizlik çalar. Çözülen ses yaklaşık PREBUFFER_S kadar birikince aygıta
   verilmeye başlanır. Aynı anda tek konuşma çalar; yeni play_* öncekini hemen keser.

miniaudio kurulu değilse ya da ses aygıtı açılamazsa StreamPlayer() StreamPlayerError fırlatır;
çağıran eski MediaPlayer yoluna dönebilir."""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Iterator

import numpy as np

try:  # Kurulu değilse modül yine içe aktarılabilsin; StreamPlayer() açık bir hata verir.
    import miniaudio
except Exception as _exc:  # pragma: no cover - kurulum eksikliği
    miniaudio = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None

# Edge'in biçimi ("audio-24khz-48kbitrate-mono-mp3"); WAV girdileri de buna dönüştürülür.
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bayt (16 bit)

# Çalmaya başlamadan önce biriktirilecek çözülmüş ses (kısa ağ takılmalarına karşı tampon).
PREBUFFER_S = 0.12
# Çözücüye bir okumada verilecek en az bayt (girdi bitmedikçe). minimp3 bir çerçeveyi ve bir
# sonrakinin başlığını görmeden eşlenemiyor; çok kısa okumalar (<~1 çerçeve) akışı erken
# bitiriyor ya da bozuyor. 1024 bayt, 24 kHz MPEG-2 çerçevesinin (en çok ~480 bayt) iki katından
# fazla; 48 kbit/s'te ~170 ms ses eder ama Edge sesi gerçek zamandan hızlı indiği için bekleme kısa.
_MIN_READ = 1024
# Konuşmanın başındaki/sonundaki sessizlik kırpılır (Edge sesleri boşlukla başlar ve biter); bu
# eşiğin altı sessiz sayılır (~-40 dBFS). Kırpılan uçlarda bu kadar ses bırakılır (doğal giriş/çıkış).
SILENCE_LEVEL = 300
KEEP_EDGE_S = 0.04
# Çözücü her seferde bu kadar kare üretir (20 ms): ilk ses ve durdurma tepkisi için küçük.
_FRAMES_PER_BLOCK = 480


class StreamPlayerError(RuntimeError):
    """Akış çalıcı kullanılamıyor (miniaudio yok, aygıt açılamadı, bozuk veri...)."""


# --- Çözme (aygıtsız, test edilebilir) ---------------------------------------------------------

if miniaudio is not None:

    class _ChunkSource(miniaudio.StreamableSource):
        """miniaudio'ya parça yineleyicisinden bayt veren kaynak.

        read() veri gelene kadar bekler (çözücü iş parçacığında çalışır). Gelen her şey saklanır:
        çözücü başlarken ID3 için 10 bayt okuyup başa geri sarıyor; sarma desteklenmezse ilk
        çerçeveler kayboluyor ve bit deposu bozulduğu için tüm ses farklı çözülüyordu. Bir
        konuşmanın MP3'ü birkaç on KB olduğundan saklamak sorun değil."""

        def __init__(self, chunks: Iterable[bytes]):
            self._it = iter(chunks)
            self._data = bytearray()
            self._pos = 0
            self._eof = False

        def read(self, num_bytes: int) -> bytes:
            want = min(num_bytes, _MIN_READ)
            while len(self._data) - self._pos < want and not self._eof:
                try:
                    chunk = next(self._it)
                except StopIteration:
                    self._eof = True
                    break
                if chunk:
                    self._data += chunk
            out = bytes(self._data[self._pos:self._pos + num_bytes])
            self._pos += len(out)
            return out

        def seek(self, offset: int, origin) -> bool:
            # Yalnızca elde olan veri içinde sarılabilir (sona sarma: uzunluk henüz bilinmiyor).
            if origin == miniaudio.SeekOrigin.START:
                base = 0
            elif origin == miniaudio.SeekOrigin.CURRENT:
                base = self._pos
            else:
                return False
            target = base + offset
            if not 0 <= target <= len(self._data):
                return False
            self._pos = target
            return True


def _require_miniaudio() -> None:
    if miniaudio is None:
        raise StreamPlayerError(f"miniaudio kurulu değil: {_IMPORT_ERROR}")


def decode_mp3_chunks(chunks: Iterable[bytes], frames_per_block: int = _FRAMES_PER_BLOCK) -> Iterator[bytes]:
    """MP3 bayt parçalarını, geldikçe 24 kHz mono 16 bit PCM bloklarına çevirir.

    `chunks` engelleyici olabilir (ör. kuyruktan okuyan üreteç); bu üreteç yalnızca gerekli kadar
    parça çeker, yani ilk PCM bloğu girdinin tamamı gelmeden döner. Bozuk/boş veride
    StreamPlayerError fırlatır (zaten üretilmiş bloklar geçerlidir). Yineleyicinin kendi hatası
    olduğu gibi yükseltilir."""
    _require_miniaudio()
    source = _ChunkSource(chunks)
    try:
        gen = miniaudio.stream_any(
            source, source_format=miniaudio.FileFormat.MP3, output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=CHANNELS, sample_rate=SAMPLE_RATE, frames_to_read=frames_per_block)
    except miniaudio.MiniaudioError as exc:
        if source.error_in_readcallback is not None:
            raise source.error_in_readcallback from exc
        raise StreamPlayerError(f"MP3 çözücü başlatılamadı (bozuk ya da boş veri): {exc}") from exc
    produced = False
    try:
        while True:
            try:
                block = next(gen)
            except StopIteration:
                if not produced:
                    raise StreamPlayerError("MP3 verisinde çözülebilir ses yok (bozuk ya da boş)") from None
                return
            except miniaudio.MiniaudioError as exc:
                if source.error_in_readcallback is not None:
                    raise source.error_in_readcallback from exc
                raise StreamPlayerError(f"MP3 çözülemedi: {exc}") from exc
            if len(block):
                produced = True
                yield block.tobytes()
    finally:
        gen.close()  # çözücüyü serbest bırakır


def decode_wav(wav_bytes: bytes) -> bytes:
    """Tüm bir WAV'ı (ör. Gemini'nin 16 bit PCM'i) çalıcının biçimine (24 kHz mono 16 bit) çevirir."""
    _require_miniaudio()
    try:
        decoded = miniaudio.decode(bytes(wav_bytes), output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=CHANNELS, sample_rate=SAMPLE_RATE)
    except miniaudio.MiniaudioError as exc:
        raise StreamPlayerError(f"WAV çözülemedi: {exc}") from exc
    return decoded.samples.tobytes()


def _scale(pcm: bytes, volume: float) -> bytes:
    """16 bit PCM'in sesini ölçekler (1.0: değişmez); taşmalar kırpılır."""
    if abs(volume - 1.0) < 1e-6:
        return pcm
    import numpy as np
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * max(0.0, float(volume))
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


# --- Çalma ------------------------------------------------------------------------------------

class PlayHandle:
    """Tek bir konuşmanın durumu. Çözücü iş parçacığı besler, aygıt geri çağrısı tüketir."""

    def __init__(self, prebuffer_bytes: int, tail_bytes: int):
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._pos = 0
        self._prebuffer = prebuffer_bytes
        self._tail_left = tail_bytes  # bittikten sonra aygıt tamponu boşalsın diye sessizlik
        self._ready = False
        self._decoded_bytes = 0
        self._decode_done = threading.Event()
        self._started = threading.Event()
        self._done = threading.Event()
        self._stopped = False
        self._heard_voice = False  # baştaki sessizlik geçildi mi
        self._lead = b""
        self.trimmed_s = 0.0       # kırpılan sessizlik (sn)
        self._started_at: float | None = None
        self.finished_at: float | None = None  # bittiği/durdurulduğu time.monotonic() anı
        self.error: Exception | None = None  # çözme hatası (varsa); o ana kadarki ses yine çalar

    # Çağıran tarafı
    def wait(self, timeout: float | None = None) -> bool:
        """Konuşma bitene ya da durdurulana kadar bekler; zaman aşımında False."""
        return self._done.wait(timeout)

    def wait_started(self, timeout: float | None = None) -> bool:
        """İlk ses aygıta verilene kadar bekler (bitmiş/durmuşsa da döner); başladıysa True."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._started.is_set() and not self._done.is_set():
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                break
            self._started.wait(0.02 if left is None else min(0.02, left))
        return self._started.is_set()

    @property
    def started_at(self) -> float | None:
        """İlk sesin aygıta verildiği time.monotonic() anı (duyulması ~1 aygıt periyodu sonra)."""
        return self._started_at

    @property
    def duration_s(self) -> float:
        """Şimdiye kadar çözülen sesin süresi; çözme bitince toplam süre."""
        return self._decoded_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    # Çözücü tarafı
    def _feed(self, pcm: bytes) -> None:
        if not self._heard_voice:
            pcm = self._skip_leading_silence(pcm)
            if not pcm:
                return
        with self._lock:
            if self._stopped:
                return
            self._buf += pcm
            self._decoded_bytes += len(pcm)
            if len(self._buf) - self._pos >= self._prebuffer:
                self._ready = True

    def _skip_leading_silence(self, pcm: bytes) -> bytes:
        # Atılan sessizliğin son KEEP_EDGE_S'i saklanır: ses başlayınca önüne eklenir.
        pcm = self._lead + pcm
        keep = int(KEEP_EDGE_S * SAMPLE_RATE) * SAMPLE_WIDTH
        samples = np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16)
        loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > SILENCE_LEVEL)
        if not len(loud):
            self._lead = pcm[-keep:]
            self.trimmed_s += (len(pcm) - len(self._lead)) / (SAMPLE_RATE * SAMPLE_WIDTH)
            return b""
        self._heard_voice = True
        start = max(0, int(loud[0]) * SAMPLE_WIDTH - keep)
        self.trimmed_s += start / (SAMPLE_RATE * SAMPLE_WIDTH)
        self._lead = b""
        return pcm[start:]

    def _trim_trailing_silence(self) -> None:
        """Çözme bitti: henüz çalınmamış sondaki sessizliği at (kilit altında çağrılır)."""
        tail = np.frombuffer(bytes(self._buf[self._pos:])[:(len(self._buf) - self._pos) // 2 * 2], np.int16)
        if not len(tail):
            return
        loud = np.flatnonzero(np.abs(tail.astype(np.int32)) > SILENCE_LEVEL)
        keep = (int(loud[-1]) + 1 + int(KEEP_EDGE_S * SAMPLE_RATE)) if len(loud) else 0
        if keep < len(tail):
            cut = len(tail) - keep
            del self._buf[len(self._buf) - cut * SAMPLE_WIDTH:]
            self._decoded_bytes -= cut * SAMPLE_WIDTH
            self.trimmed_s += cut / SAMPLE_RATE

    def _finish_decoding(self, error: Exception | None = None) -> None:
        with self._lock:
            if not self._stopped:
                self._trim_trailing_silence()
            if error is not None and self.error is None:
                self.error = error
            self._ready = True
            if self._decoded_bytes == 0:  # hiç ses yok: beklemeden bitir
                self._mark_done()
        self._decode_done.set()

    def _stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._buf = bytearray()
            self._pos = 0
            self._mark_done()

    def _mark_done(self) -> None:
        if not self._done.is_set():
            self.finished_at = time.monotonic()
            self._done.set()

    # Aygıt tarafı (ses geri çağrısı; kısa ve hatasız olmalı)
    def _take(self, nbytes: int) -> tuple[bytes, bool]:
        """En çok nbytes PCM verir; ikinci değer True ise konuşma bitti."""
        with self._lock:
            if self._stopped or self._done.is_set():
                return b"", True
            if not self._ready:
                return b"", False  # ön tampon dolmadı (ya da ilk veri gelmedi)
            out = bytes(self._buf[self._pos:self._pos + nbytes])
            self._pos += len(out)
            if self._pos >= 65536:  # tüketilen başı at (bellek sınırlı kalsın)
                del self._buf[:self._pos]
                self._pos = 0
            if out and self._started_at is None:
                self._started_at = time.monotonic()
                self._started.set()
            if self._decode_done.is_set() and self._pos >= len(self._buf):
                self._tail_left -= nbytes - len(out)
                if self._tail_left <= 0:
                    self._mark_done()
                    return out, True
            return out, False


class StreamPlayer:
    """Tek konuşmalık düşük gecikmeli çalıcı (iş parçacığı güvenli).

    play_chunks / play_wav hemen döner (engellemez); çözme ayrı bir iş parçacığında yapılır,
    çalma ~PREBUFFER_S ses birikince aygıtın bir sonraki geri çağrısında başlar. Başlamayı
    beklemek için handle.wait_started(), bitişi beklemek için handle.wait() kullanılır.

    device_factory yalnızca testler içindir: periyot (ms) alıp start(gen)/close() sunan bir nesne
    döndürür; gen'e send(kare_sayısı) gönderilir, o kadar karelik PCM baytı döner."""

    def __init__(self, log: Callable[[str], None] = print, *, period_ms: int = 30,
                 prebuffer_s: float = PREBUFFER_S, device_factory: Callable | None = None):
        self._log = log
        self._lock = threading.Lock()
        self._current: PlayHandle | None = None
        self._closed = False
        frame_bytes = CHANNELS * SAMPLE_WIDTH
        self._prebuffer_bytes = int(prebuffer_s * SAMPLE_RATE) * frame_bytes
        # miniaudio varsayılan 3 periyot tampon kullanır; bitişte o kadar sessizlik daha verilir
        self._tail_bytes = 3 * int(period_ms * SAMPLE_RATE / 1000) * frame_bytes
        if device_factory is None:
            _require_miniaudio()
            device_factory = _open_device
        gen = self._render_gen()
        next(gen)
        try:
            self._device = device_factory(period_ms)
            self._device.start(gen)
        except StreamPlayerError:
            raise
        except Exception as exc:
            raise StreamPlayerError(f"Ses çıkış aygıtı açılamadı: {exc}") from exc

    def play_chunks(self, chunks: Iterable[bytes], volume: float = 1.0, speed: float = 1.0) -> PlayHandle:
        """MP3 parçalarını geldikçe çalar; hemen döner. Önceki konuşmayı keser.

        `chunks` başka iş parçacığından beslenen engelleyici bir yineleyici olabilir; durdurulan
        bir konuşmanın çözücüsü, yineleyici bir sonraki parçayı verene ya da bitene kadar bekler
        (sonra sessizce çıkar), bu yüzden çağıran durdurduğu konuşmanın kuyruğunu kapatmalıdır.
        `speed` yok sayılır (hız Edge'e rate= ile verilmeli; burada değiştirmek perdeyi bozar)."""
        del speed
        return self._start(lambda: decode_mp3_chunks(chunks), volume)

    def play_wav(self, wav_bytes: bytes, volume: float = 1.0) -> PlayHandle:
        """Tüm bir WAV'ı çalar (gerekirse 24 kHz monoya çevrilir); hemen döner."""
        def blocks() -> Iterator[bytes]:
            pcm = decode_wav(wav_bytes)
            step = _FRAMES_PER_BLOCK * CHANNELS * SAMPLE_WIDTH * 8
            for i in range(0, len(pcm), step):
                yield pcm[i:i + step]
        return self._start(blocks, volume)

    def stop(self) -> None:
        """Çalan konuşmayı hemen keser (aygıt bir sonraki periyotta sessizliğe geçer)."""
        with self._lock:
            old, self._current = self._current, None
        if old is not None:
            old._stop()

    def busy(self) -> bool:
        """Çalan ya da çalmayı bekleyen (ön tamponda) bir konuşma var mı?"""
        with self._lock:
            return self._current is not None and not self._current.done

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._device.close()
        except Exception as exc:  # kapanışta hata önemsiz
            self._log(f"Ses aygıtı kapatılamadı: {exc}")

    # --- iç ---
    def _start(self, make_blocks: Callable[[], Iterator[bytes]], volume: float) -> PlayHandle:
        handle = PlayHandle(self._prebuffer_bytes, self._tail_bytes)
        with self._lock:
            if self._closed:
                raise StreamPlayerError("Akış çalıcı kapatıldı")
            old, self._current = self._current, handle
        if old is not None:
            old._stop()
        threading.Thread(target=self._decode_worker, args=(handle, make_blocks, volume),
                         name="dublaj-cozucu", daemon=True).start()
        return handle

    def _decode_worker(self, handle: PlayHandle, make_blocks: Callable[[], Iterator[bytes]],
                       volume: float) -> None:
        error: Exception | None = None
        blocks = None
        try:
            blocks = make_blocks()
            for pcm in blocks:
                if handle.done:  # durduruldu: çözmeyi bırak
                    break
                handle._feed(_scale(pcm, volume))
        except Exception as exc:
            error = exc
            self._log(f"Dublaj sesi çözülemedi: {exc}")
        finally:
            if blocks is not None and hasattr(blocks, "close"):
                try:
                    blocks.close()
                except Exception:
                    pass
            handle._finish_decoding(error)

    def _render(self, nframes: int) -> bytes:
        nbytes = nframes * CHANNELS * SAMPLE_WIDTH
        with self._lock:
            handle = self._current
        out = b""
        if handle is not None:
            out, finished = handle._take(nbytes)
            if finished:
                with self._lock:
                    if self._current is handle:
                        self._current = None
        if len(out) < nbytes:
            out += bytes(nbytes - len(out))
        return out

    def _render_gen(self):
        """Aygıt geri çağrısının üreteci: istenen kare sayısını alır, PCM verir. Asla hata vermez
        (miniaudio hata alınca aygıtı durdurur)."""
        nframes = yield b""
        while True:
            try:
                out = self._render(nframes)
            except Exception:
                out = bytes(nframes * CHANNELS * SAMPLE_WIDTH)
            nframes = yield out


def _open_device(period_ms: int):
    return miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16, nchannels=CHANNELS,
                                    sample_rate=SAMPLE_RATE, buffersize_msec=period_ms, app_name="gta-altyazi")
