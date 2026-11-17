"""Karma dublaj: Gemini (duygulu) önce, sorun olursa anında Edge, o da yoksa Windows."""

import base64
import io
import time
import wave

import pytest

from altyazi import gemini_tts as gt
from altyazi.config import Config
from altyazi.tts import Delivery, Speaker


class Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload, self.text = status, payload or {}, ""

    def json(self):
        return self._payload


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append((url, json))
        return self.responses.pop(0)


def _audio_payload(pcm=b"\x00\x01" * 2400, mime="audio/L16;codec=pcm;rate=24000"):
    return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": mime,
                                                                   "data": base64.b64encode(pcm).decode()}}]}}]}


def test_gemini_request_and_wav_response():
    body = gt.request_body(gt.style_prompt("Kapa çeneni!", "angry", 0.9), "Charon")
    assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Charon"
    assert "very angrily" in body["contents"][0]["parts"][0]["text"]
    wav = gt.synthesize(Session(Resp(200, _audio_payload())), "k", "m", "p", "Kore", 1)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)


@pytest.mark.parametrize("status,error", [(429, gt.GeminiQuota), (403, gt.GeminiAuth), (404, gt.GeminiModelMissing),
                                          (500, gt.GeminiError)])
def test_gemini_errors_are_typed(status, error):
    with pytest.raises(error):
        gt.synthesize(Session(Resp(status, {"error": {"message": "x"}})), "k", "m", "p", "Kore", 1)


def _speaker(**cfg_overrides):
    settings = {"dub_engine": "gemini", "gemini_api_key": "anahtar", "dub_gemini_wait_ms": 300, **cfg_overrides}
    cfg = Config(**settings)
    sp = Speaker(cfg, None, None, log=lambda m: None)
    played = []
    sp._play_file = lambda path: played.append(path.suffix)
    sp._play_windows = lambda *a: played.append("windows")
    sp._delivery = lambda *a: Delivery(emotion="angry", intensity=0.8)
    return sp, played


def test_gemini_in_time_is_played():
    sp, played = _speaker()
    sp._gemini_bytes = lambda text, d: b"RIFFgemini"
    sp._edge_open = lambda *a: b"edge"
    sp._speak_line("Kapa çeneni!", time.monotonic(), sp._generation, None, "Shut up!", None, time.monotonic())
    assert played == [".wav"] and sp.last_engine == "Gemini"


def test_slow_gemini_falls_back_to_ready_edge_without_waiting_longer():
    sp, played = _speaker()

    def slow(text, d):
        time.sleep(2)
        return b"RIFFlate"
    sp._gemini_bytes = slow
    sp._edge_open = lambda *a: b"edge"
    t0 = time.monotonic()
    sp._speak_line("Kapa çeneni!", time.monotonic(), sp._generation, None, "Shut up!", None, time.monotonic())
    assert played == [".mp3"] and "geç kaldı" in sp.last_engine
    assert time.monotonic() - t0 < 1.0  # en fazla dub_gemini_wait_ms (+ küçük pay)


def test_gemini_error_and_edge_down_uses_windows():
    sp, played = _speaker()

    def broken(text, d):
        raise gt.GeminiQuota("429")
    sp._gemini_bytes = broken
    sp._edge_open = lambda *a: None
    sp._speak_line("Kapa çeneni!", time.monotonic(), sp._generation, None, "Shut up!", None, time.monotonic())
    assert played == ["windows"]


def test_quota_switches_gemini_off_for_a_while(monkeypatch):
    sp, _ = _speaker()
    sp._http = Session(Resp(429))
    with pytest.raises(gt.GeminiQuota):
        sp._gemini_bytes("Selam", Delivery())
    assert sp._gemini_off_until > time.monotonic() + 500  # sonraki satırlar beklemeden Edge


def test_missing_model_name_tries_the_next_one():
    sp, _ = _speaker()
    sp._http = Session(Resp(404), Resp(200, _audio_payload()))
    assert sp._gemini_bytes("Selam", Delivery()).startswith(b"RIFF")
    assert sp._gemini_model == 1



def _queue_speaker():
    sp, played = _speaker(dub_engine="edge")
    sp._edge_open = lambda text, d, rate: text.encode() * 8000  # ~1,3 sn "mp3" (48 kbit/sn tahmini)
    sp._play_file = lambda path: played.append(path.read_bytes()[:1].decode())
    return sp, played


def test_lines_are_not_cut_and_play_in_order():
    sp, played = _queue_speaker()
    now = time.monotonic()
    sp.busy_until = now + 0.3  # önceki satır hâlâ çalıyor
    sp._speak_line("A", now, sp._generation, None, "", None, now)
    assert played == ["A"] and time.monotonic() >= now + 0.3  # öncekinin bitmesi beklendi, kesilmedi
    first_end = sp.busy_until
    sp._speak_line("B", time.monotonic(), sp._generation, None, "", None, time.monotonic())
    assert played == ["A", "B"] and time.monotonic() >= first_end - 0.05


def test_far_behind_line_is_skipped_when_newer_lines_wait():
    sp, played = _queue_speaker()
    old = time.monotonic() - 6  # 6 sn önce başlamış satır
    sp._jobs.put(("yeni", time.monotonic(), sp._generation, None, "", None, time.monotonic()))
    sp._speak_line("eski", old, sp._generation, None, "", None, old)
    assert played == []  # sırada yenisi varken çok geride kalan atlanır
    sp._jobs.get_nowait()
    sp._speak_line("eski ama tek", old, sp._generation, None, "", None, old)
    assert played == ["e"]  # sırada başka satır yoksa yine okunur


def test_backlog_speeds_reading_up():
    sp, _ = _queue_speaker()
    rates = []
    sp._synthesize = lambda text, d, rate: (rates.append(rate), (None, "", ""))[1]
    sp._play_windows = lambda *a: None
    sp._speak_line("tek", time.monotonic(), sp._generation, None, "", None, time.monotonic())
    for _ in range(2):
        sp._jobs.put(("x", time.monotonic(), sp._generation, None, "", None, time.monotonic()))
    sp._speak_line("sıra var", time.monotonic(), sp._generation, None, "", None, time.monotonic())
    assert rates[1] > rates[0] * 1.2


def test_consistently_late_gemini_is_not_waited_for():
    sp, played = _speaker(dub_engine="gemini")
    sp._gemini_times.extend([2.0, 2.2, 1.9, 2.1])  # hep bekleme süresinden geç
    called = []
    sp._gemini_bytes = lambda text, d: called.append(1) or b"RIFF"
    sp._edge_open = lambda *a: b"edge"
    t0 = time.monotonic()
    sp._speak_line("Selam", time.monotonic(), sp._generation, None, "", None, time.monotonic())
    assert played == [".mp3"] and time.monotonic() - t0 < 0.3 and called == []


def test_said_lines_are_prepared_immediately_not_after_previous_starts():
    sp, played = _queue_speaker()
    started = []

    def synth(text, d, rate):
        started.append((text, time.monotonic()))
        time.sleep(0.3)
        return text.encode() * 8000, ".mp3", "Edge"
    sp._synthesize = synth
    t0 = time.monotonic()
    sp.say("A", t0)
    sp.say("B", t0)
    time.sleep(0.1)
    assert [t for t, _ in started] == ["A", "B"]  # ikisi de hemen hazırlanmaya başladı
    sp._speak_line(*sp._jobs.get_nowait())
    sp.busy_until = 0
    sp._speak_line(*sp._jobs.get_nowait())
    assert played == ["A", "B"]


def test_rate_rises_when_dub_falls_behind_subtitles():
    sp, _ = _queue_speaker()
    now = time.monotonic()
    first = sp._plan("Evet, orada çok zaman geçirdiğini görebiliyorum.", now, 0, now)
    later = sp._plan("Evet, mümkün olduğunca çok.", now + 0.5, 0, now + 0.5)  # önceki daha bitmemiş
    assert first == 1.0 and 1.15 < later <= 1.45
    assert sp._plan("x", now - 30, 0, now) == pytest.approx(1.45)  # hız sınırı


def test_interrupt_releases_prepared_streams():
    from altyazi.tts import _EdgeStream
    sp, _ = _queue_speaker()
    stream = _EdgeStream()
    sp._synthesize = lambda text, d, rate: (stream, ".mp3", "Edge")
    sp.say("A", time.monotonic())
    time.sleep(0.1)
    sp.interrupt()
    time.sleep(0.05)
    assert stream.closed


class FakeFast:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def request(self, text, voice, rate, pitch, volume, on_audio, on_end, timeout):
        self.calls.append(voice)
        for c in self.chunks:
            on_audio(c)
        on_end(None if self.chunks else ConnectionError("yok"))


def test_edge_uses_ready_connection_first(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "edge_tts", types.ModuleType("edge_tts"))
    sp, _ = _speaker(dub_engine="edge")
    sp._fast = FakeFast([b"a", b"b"])
    stream = sp._edge_open("Selam", Delivery(gender="erkek"), 1.1)
    assert stream.read_all() == b"ab" and sp._fast.calls == ["tr-TR-AhmetNeural"]


def test_failed_ready_connection_falls_back_to_normal_edge(monkeypatch):
    import sys
    import types
    from altyazi.tts import Speaker as Sp
    monkeypatch.setitem(sys.modules, "edge_tts", types.ModuleType("edge_tts"))
    used = []

    def produce(edge_tts, stream, text, voice, options):
        used.append(voice)
        stream.put(b"normal")
        stream.end()
    monkeypatch.setattr(Sp, "_edge_produce", staticmethod(produce))
    sp, _ = _speaker(dub_engine="edge")
    sp._fast = FakeFast([])
    stream = sp._edge_open("Selam", Delivery(gender="erkek"), 1.1)
    assert stream.read_all() == b"normal" and used == ["tr-TR-AhmetNeural"]


# --- Tahmini ses: konuşan tanıma bitmeden Edge istenir ---------------------------------------

def _char(cid, gender, pitch=0, voice=0):
    from altyazi.voiceid import Character
    return Character(id=cid, gender=gender, f0=120 if gender == "erkek" else 210, timbre=[1.0] * 12,
                     pitch_pct=pitch, voice=voice)


def _spec_speaker(first_chunk_s=0.0, slow_gender=None):
    """Tanıması elle bitirilen, Edge isteklerini kaydeden konuşmacı (akışlar _EdgeStream)."""
    from concurrent.futures import Future
    from altyazi.tts import _Detection, _EdgeStream
    sp, played = _speaker(dub_engine="edge")
    del sp._delivery  # gerçek tanıma yolu
    calls = []

    def synth(text, d, rate):
        stream = _EdgeStream()
        calls.append((time.monotonic(), d, stream))
        time.sleep(first_chunk_s + (0.5 if d.gender == slow_gender else 0.0))
        stream.put(b"ses-" + d.gender.encode())
        stream.end()
        return stream, ".mp3", "Edge"
    sp._synthesize = synth
    det = _Detection()
    det.job = Future()
    sp._prepared["satir"] = det
    return sp, det, calls


def test_speculative_voice_starts_before_detection_and_matching_detection_keeps_it():
    from altyazi.tts import _Turn
    sp, det, calls = _spec_speaker()
    adam = _char(1, "erkek")
    sp._recent.append((time.monotonic(), adam))  # önceki satırı o söyledi
    t0 = time.monotonic()
    job = sp._prep.submit(sp._prepare, "Selam", t0, "Hi", "satir", None, 1.0, _Turn())
    time.sleep(0.15)
    assert len(calls) == 1 and calls[0][1].gender == "erkek" and not det.job.done()  # tanıma beklenmedi
    det.job.set_result((adam, None, 120.0))
    audio, ext, source, d, rate, detect_s, synth_s, guess = job.result(2)
    assert guess == "tuttu" and audio is calls[0][2] and not audio.closed
    assert len(calls) == 1 and d.speaker == 1  # ikinci istek yok
    assert "Karakter 1" in sp.last_character


def test_wrong_speculative_voice_is_replaced_and_closed():
    from altyazi.tts import _Turn
    sp, det, calls = _spec_speaker(slow_gender="kadin")  # doğru (kadın) ses 0,5 sn geç gelir
    sp._recent.append((time.monotonic(), _char(1, "erkek")))
    turn = _Turn()
    job = sp._prep.submit(sp._prepare, "Selam", time.monotonic(), "Hi", "satir", None, 1.0, turn)
    time.sleep(0.1)
    turn.mark()  # çalma sırası çoktan geldi: yine de yanlış cinsiyetteki ses çalınmaz
    det.job.set_result((_char(2, "kadin"), None, 210.0))
    audio, ext, source, d, rate, detect_s, synth_s, guess = job.result(3)
    assert [c[1].gender for c in calls] == ["erkek", "kadin"]
    assert d.gender == "kadin" and audio is calls[1][2] and "tutmadı" in guess
    assert calls[0][2].closed and not audio.closed  # tutmayan tahminin akışı bırakıldı
    assert audio.read_all() == b"ses-kadin"


def test_two_recent_speakers_are_both_tried_in_a_dialogue():
    from altyazi.tts import _Turn
    sp, det, calls = _spec_speaker()
    a, b = _char(1, "erkek"), _char(2, "kadin")
    now = time.monotonic()
    sp._recent.extend([(now - 3, a), (now - 1, b)])  # A, B konuştu; sıra büyük olasılıkla A'da
    job = sp._prep.submit(sp._prepare, "Selam", now, "Hi", "satir", None, 1.0, _Turn())
    time.sleep(0.1)
    assert [c[1].speaker for c in calls] == [2, 1]
    det.job.set_result((a, None, 120.0))
    audio, *_, guess = job.result(2)
    assert guess == "tuttu" and audio is calls[1][2] and calls[0][2].closed and len(calls) == 2


def test_unfinished_detection_plays_the_guess_shortly_after_the_turn_comes():
    from altyazi.tts import LATE_S, _Turn
    sp, det, calls = _spec_speaker()
    sp._recent.append((time.monotonic(), _char(1, "erkek")))
    turn = _Turn()
    job = sp._prep.submit(sp._prepare, "Selam", time.monotonic(), "Hi", "satir", None, 1.0, turn)
    time.sleep(0.1)
    turn.mark()
    t_due = time.monotonic()
    audio, *_, guess = job.result(2)
    waited = time.monotonic() - t_due
    assert LATE_S - 0.05 <= waited < LATE_S + 0.2 and "tahmin" in guess and not audio.closed
    det.job.set_result((None, None, 0.0))


def test_same_character_with_other_emotion_waits_for_right_tone_only_briefly():
    from altyazi.tts import LATE_S, _Turn
    sp, det, calls = _spec_speaker()
    adam = _char(1, "erkek")
    sp._recent.append((time.monotonic(), adam))

    def synth(text, d, rate):  # doğru tonlama (bağıran) çok geç gelir
        from altyazi.tts import _EdgeStream
        stream = _EdgeStream()
        calls.append((time.monotonic(), d, stream))
        if len(calls) > 1:
            time.sleep(1.0)
        stream.put(b"x")
        stream.end()
        return stream, ".mp3", "Edge"
    sp._synthesize = synth
    turn = _Turn()
    job = sp._prep.submit(sp._prepare, "Kapa çeneni!", time.monotonic(), "Shut up!", "satir", None, 1.0, turn)
    time.sleep(0.1)
    det.job.set_result((adam, 1.0, 160.0))  # ses yükselmiş: sinirli, tonlama tahminden farklı
    time.sleep(0.05)
    turn.mark()
    t_due = time.monotonic()
    audio, ext, source, d, *_, guess = job.result(3)
    assert len(calls) == 2 and audio is calls[0][2] and "tonlama" in guess
    assert time.monotonic() - t_due < LATE_S + 0.2


def test_edge_fast_stops_downloading_a_discarded_stream():
    import asyncio
    aiohttp = pytest.importorskip("aiohttp")  # edge-tts ile kurulur; yoksa test atlanır
    from altyazi.edge_fast import EdgeFast
    from altyazi.tts import _EdgeStream

    class Msg:
        def __init__(self, data):
            self.type, self.data = aiohttp.WSMsgType.BINARY, data

    class Ws:
        def __init__(self):
            self.sent, self.read = [], 0

        async def send_str(self, s):
            self.sent.append(s)

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.read += 1
            if self.read > 50:
                raise StopAsyncIteration
            return Msg((2).to_bytes(2, "big") + b"\r\n" + b"ses")

    class Comm:
        mkssml = staticmethod(lambda config, text: text)
        remove_incompatible_characters = staticmethod(lambda t: t)
        ssml_headers_plus_data = staticmethod(lambda *a: "istek")
        connect_id = staticmethod(lambda: "id")
        date_to_string = staticmethod(lambda: "zaman")
        get_headers_and_data = staticmethod(lambda data, n: ({}, data[2 + n:]))

    fast = object.__new__(EdgeFast)
    fast._comm, fast._aiohttp, fast._config_cls = Comm, aiohttp, lambda *a: None
    stream, ws = _EdgeStream(), Ws()
    got = []

    def on_audio(data):
        got.append(data)
        stream.close()  # ilk parçadan sonra artık istenmiyor
        return stream.put(data)
    asyncio.run(fast._speak(ws, "Selam", "v", "+0%", "+0Hz", "+0%", on_audio))
    assert got == [b"ses"] and ws.read == 1
