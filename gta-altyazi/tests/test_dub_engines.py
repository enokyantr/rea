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
