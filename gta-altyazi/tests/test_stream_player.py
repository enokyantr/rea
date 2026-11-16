"""stream_player: akışlı MP3 çözme ve çalıcı mantığı (ses aygıtı olmadan; sahte aygıtla)."""

import io
import queue
import threading
import time
import wave

import numpy as np
import pytest

miniaudio = pytest.importorskip("miniaudio")
lameenc = pytest.importorskip("lameenc")  # yalnızca test MP3'ü üretmek için

from altyazi import stream_player as sp  # noqa: E402

SR = sp.SAMPLE_RATE


def make_mp3(seconds=1.5, kbps=48, seed=0) -> bytes:
    """Edge biçimine benzer (24 kHz mono, 48 kbit/s) bir test MP3'ü üretir."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    sig = np.sin(2 * np.pi * (220 + 200 * t) * t) * 7000 + rng.normal(0, 800, len(t))
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(SR)
    enc.set_channels(1)
    enc.set_quality(5)
    return bytes(enc.encode(sig.astype(np.int16).tobytes()) + enc.flush())


def whole_decode(mp3: bytes) -> bytes:
    return miniaudio.decode(mp3, nchannels=1, sample_rate=SR).samples.tobytes()


def split(data: bytes, sizes):
    out, i, k = [], 0, 0
    while i < len(data):
        n = sizes[k % len(sizes)]
        out.append(data[i:i + n])
        i, k = i + n, k + 1
    return out


@pytest.mark.parametrize("sizes", [[1], [7, 300], [144], [1000], [4096], [100000]])
def test_chunked_decode_equals_whole_file(sizes):
    mp3 = make_mp3()
    pcm = b"".join(sp.decode_mp3_chunks(split(mp3, sizes)))
    assert pcm == whole_decode(mp3)


def test_decode_yields_first_audio_before_input_ends():
    mp3 = make_mp3(seconds=3)
    chunks = split(mp3, [512])
    pulled = []

    def source():
        for c in chunks:
            pulled.append(len(c))
            yield c

    gen = sp.decode_mp3_chunks(source())
    first = next(gen)
    assert first
    assert len(pulled) < len(chunks) // 4  # yalnızca başı okundu
    rest = b"".join(gen)
    assert first + rest == whole_decode(mp3)


def test_decode_streams_from_blocking_queue():
    """Üretici, ilk PCM gelmeden geri kalanı göndermez: akış değilse burada kilitlenir."""
    mp3 = make_mp3(seconds=2)
    chunks = split(mp3, [600])
    q: queue.Queue = queue.Queue()
    first_audio = threading.Event()

    def producer():
        for c in chunks[:3]:
            q.put(c)
        first_audio.wait(5)
        for c in chunks[3:]:
            q.put(c)
        q.put(None)

    def from_queue():
        while (c := q.get(timeout=10)) is not None:
            yield c

    threading.Thread(target=producer, daemon=True).start()
    out = []
    for block in sp.decode_mp3_chunks(from_queue()):
        out.append(block)
        first_audio.set()
    assert first_audio.is_set()
    assert b"".join(out) == whole_decode(mp3)


def test_corrupt_or_empty_input_raises_clear_error():
    with pytest.raises(sp.StreamPlayerError):
        list(sp.decode_mp3_chunks([bytes(range(256)) * 20]))
    with pytest.raises(sp.StreamPlayerError):
        list(sp.decode_mp3_chunks([]))


def test_iterator_error_propagates():
    def broken():
        yield make_mp3()[:2000]
        raise ConnectionError("ağ koptu")

    with pytest.raises(ConnectionError):
        list(sp.decode_mp3_chunks(broken()))


def test_scale_volume_and_clipping():
    pcm = np.array([1000, -1000, 30000], dtype=np.int16).tobytes()
    assert np.frombuffer(sp._scale(pcm, 0.5), np.int16).tolist() == [500, -500, 15000]
    assert np.frombuffer(sp._scale(pcm, 2.0), np.int16).tolist() == [2000, -2000, 32767]
    assert sp._scale(pcm, 1.0) is pcm


# --- Sahte aygıtla çalıcı ---

class FakeDevice:
    def __init__(self, period_ms):
        self.period_frames = period_ms * SR // 1000
        self.gen = None
        self.closed = False

    def start(self, gen):
        self.gen = gen

    def close(self):
        self.closed = True

    def pull(self, periods=1) -> bytes:
        return b"".join(self.gen.send(self.period_frames) for _ in range(periods))


def make_player(**kw):
    dev = {}

    def factory(period_ms):
        dev["d"] = FakeDevice(period_ms)
        return dev["d"]

    player = sp.StreamPlayer(log=lambda m: None, device_factory=factory, **kw)
    return player, dev["d"]


def pump_until_done(dev, handle, limit=2000) -> bytes:
    out = []
    for _ in range(limit):
        out.append(dev.pull())
        if handle.done:
            break
    return b"".join(out)


def test_player_plays_decoded_audio_then_finishes():
    player, dev = make_player()
    mp3 = make_mp3()
    h = player.play_chunks(split(mp3, [700]))
    assert h._decode_done.wait(10)
    assert player.busy()
    out = pump_until_done(dev, h)
    expected = whole_decode(mp3)
    assert h.done and h.wait(0) and h.error is None
    assert h.started_at is not None
    assert out[:len(expected)] == expected  # tamamı, baştan, eksiksiz
    assert not any(out[len(expected):])  # sonrası sessizlik
    assert abs(h.duration_s - len(expected) / 2 / SR) < 1e-9
    assert not player.busy()
    assert not any(dev.pull())  # boşta sessizlik
    player.close()
    assert dev.closed


def test_player_waits_for_prebuffer_before_starting():
    player, dev = make_player(prebuffer_s=0.12)
    mp3 = make_mp3(seconds=2)
    q: queue.Queue = queue.Queue()

    def from_queue():
        while (c := q.get(timeout=10)) is not None:
            yield c

    h = player.play_chunks(from_queue())
    assert not any(dev.pull(3)) and h.started_at is None  # veri yok: sessizlik, başlamadı
    for c in split(mp3, [600]):
        q.put(c)
    q.put(None)
    assert h._decode_done.wait(10)
    assert any(dev.pull()) and h.started_at is not None
    player.stop()


def test_stop_and_new_play_cut_previous():
    player, dev = make_player()
    h1 = player.play_chunks(split(make_mp3(seconds=3), [800]))
    assert h1._decode_done.wait(10)
    dev.pull(2)
    h2 = player.play_chunks(split(make_mp3(seconds=1, seed=1), [800]))
    assert h1.wait(0)  # öncekisi hemen bitti
    assert h2._decode_done.wait(10)
    assert any(dev.pull())
    player.stop()
    assert h2.wait(0) and not player.busy()
    assert not any(dev.pull())


def test_corrupt_stream_finishes_with_error():
    player, dev = make_player()
    h = player.play_chunks([b"\x00" * 3000])
    assert h.wait(5)
    assert isinstance(h.error, sp.StreamPlayerError)
    assert h.started_at is None and not player.busy()


def test_play_wav_resamples_and_scales():
    player, dev = make_player()
    sr = 16000
    pcm = (np.sin(2 * np.pi * 300 * np.arange(sr) / sr) * 10000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    h = player.play_wav(buf.getvalue(), volume=0.5)
    assert h._decode_done.wait(10)
    assert abs(h.duration_s - 1.0) < 0.01
    out = np.frombuffer(pump_until_done(dev, h), np.int16)
    assert h.done and h.error is None
    assert 4000 < np.abs(out).max() <= 5100  # yarı ses


def test_device_failure_raises_stream_player_error():
    def factory(period_ms):
        raise RuntimeError("aygıt yok")

    with pytest.raises(sp.StreamPlayerError):
        sp.StreamPlayer(log=lambda m: None, device_factory=factory)


def test_closed_player_rejects_play():
    player, dev = make_player()
    player.close()
    with pytest.raises(sp.StreamPlayerError):
        player.play_wav(b"")


# --- Dublaj seslendiricisiyle birlikte (Edge akışı -> çalıcı) ---

def _speaker_with_player():
    from altyazi.config import Config
    from altyazi.tts import Delivery, Speaker

    player, dev = make_player()
    stop = threading.Event()
    heard = []

    def pump():  # aygıt: 30 ms'lik periyodu ~6 kat hızlı ister
        while not stop.is_set():
            heard.append((time.monotonic(), dev.pull()))
            time.sleep(0.005)
    threading.Thread(target=pump, daemon=True).start()
    speaker = Speaker(Config(dub_engine="edge"), None, None, log=lambda m: None)
    speaker._stream = player
    speaker._delivery = lambda *a: Delivery()
    return speaker, heard, stop


def _slow_edge(mp3, chunk=600, delay=0.03):
    """Edge gibi: parçalar ağdan yavaş yavaş gelir."""
    from altyazi.tts import _EdgeStream

    def open_(text, d, rate):
        stream = _EdgeStream()

        def produce():
            for i in range(0, len(mp3), chunk):
                if stream.closed:
                    return
                stream.put(mp3[i:i + chunk])
                time.sleep(delay)
            stream.end()
        threading.Thread(target=produce, daemon=True).start()
        stream.first.wait(2)
        return stream
    return open_


def _audible(heard):
    return [t for t, pcm in heard if np.abs(np.frombuffer(pcm, np.int16)).max(initial=0) > 2000]


def test_dub_starts_before_edge_download_finishes():
    speaker, heard, stop = _speaker_with_player()
    mp3 = make_mp3(2.0)
    speaker._edge_open = _slow_edge(mp3)  # tamamı ~1 sn'de iner
    t0 = time.monotonic()
    speaker._speak_line("Selam", t0, speaker._generation, None, "", None, t0)
    assert speaker.playing()
    speaker._handle.wait(10)
    stop.set()
    first = _audible(heard)[0]
    assert first - t0 < 0.5  # indirme bitmeden (~1 sn) ses çıktı
    assert speaker.busy_until >= speaker._handle.finished_at


def test_next_dub_line_waits_for_previous_instead_of_cutting_it():
    speaker, heard, stop = _speaker_with_player()
    speaker._edge_open = _slow_edge(make_mp3(0.8), delay=0.0)
    t0 = time.monotonic()
    speaker._speak_line("Bir", t0, speaker._generation, None, "", None, t0)
    first = speaker._handle
    speaker._speak_line("İki", time.monotonic(), speaker._generation, None, "", None, time.monotonic())
    assert first.done and first.error is None and first.duration_s > 0.7  # sonuna kadar çaldı
    second = speaker._handle
    assert second.wait_started(5)
    assert second is not first and second.started_at >= first.finished_at - 0.05
    second.wait(10)
    stop.set()


def test_interrupt_stops_playback_and_releases_download():
    speaker, heard, stop = _speaker_with_player()
    speaker._edge_open = _slow_edge(make_mp3(3.0), delay=0.05)
    t0 = time.monotonic()
    speaker._speak_line("Uzun", t0, speaker._generation, None, "", None, t0)
    feeding = speaker._feeding
    speaker.interrupt()
    assert feeding.closed and speaker._handle.done and not speaker.playing()
    stop.set()
