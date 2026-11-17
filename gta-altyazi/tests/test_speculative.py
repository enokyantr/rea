"""Önceden (spekülatif) son çeviri: doğrulanan satır aynıysa yeniden kullanılır, farklıysa asla."""

import threading
import time

from altyazi.hybrid import DRAFT, FINAL, HybridTranslator
from altyazi.textproc import Dialogue
from altyazi.translate import CachedTranslator


class FakeLLM:
    """İstekleri kaydeden, isteğe göre bekleten ya da gecikmeli yanıt veren sahte son çevirmen."""

    cancellable = True

    def __init__(self, delay=0.0, gated=False, fail_first=False):
        self.delay = delay
        self.cancelled = []
        self.release = threading.Event()
        if not gated:
            self.release.set()
        self.fail_first = fail_first
        self.requests = []  # (metin, bağlam değeri, geçmiş)
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def translate(self, text, context, cancel=None):
        with self._lock:
            self.requests.append((text, None if context is None else str(context),
                                  tuple(getattr(context, "history", ()))))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            failing = self.fail_first and len(self.requests) == 1
        try:
            end = time.monotonic() + 2
            while not self.release.wait(0.002):
                assert time.monotonic() < end
                if cancel is not None and cancel.is_set():
                    self.cancelled.append(text)
                    raise RuntimeError("iptal")
            time.sleep(self.delay)
            if failing:
                raise ConnectionError("geçici")
            return f"llm:{text}"
        finally:
            with self._lock:
                self.active -= 1


class FakeDraft:
    def __init__(self):
        self.calls = []

    def translate(self, text, context):
        self.calls.append(text)
        return f"yerel:{text}"


class Collector:
    def __init__(self):
        self.results, self.errors = [], []
        self.done = threading.Event()
        self.final_at = None

    def on_result(self, stage, text, took):
        self.results.append((stage, text))
        if stage == FINAL:
            self.final_at = time.perf_counter()
            self.done.set()

    def on_error(self, exc, stage, draft_shown):
        self.errors.append((stage, draft_shown))
        if stage == FINAL:
            self.done.set()


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.002)
    return False


def make(llm, draft=None):
    final = CachedTranslator(llm, None)
    return HybridTranslator(final, draft), final


def submit(h, text, context=None):
    c = Collector()
    h.submit(text, context, lambda: True, c.on_result, c.on_error)
    return c


def test_matching_confirmed_line_reuses_in_flight_request_and_saves_llm_latency():
    llm = FakeLLM(delay=0.3)
    h, _ = make(llm)
    t0 = time.perf_counter()
    assert h.prefetch("Get in the car.", lambda: None)
    time.sleep(0.1)  # doğrulama okuması
    c = submit(h, "get in  the car.")  # aynı önbellek anahtarı (boşluk/büyük harf)
    assert c.done.wait(2)
    assert c.results == [(FINAL, "llm:Get in the car.")]
    assert len(llm.requests) == 1  # ikinci istek yok
    # İlk okumadan son çeviriye ~0,3 sn (onaydan sonra ~0,2); spekülasyonsuz 0,1 + 0,3 olurdu.
    assert c.final_at - t0 < 0.38
    h.shutdown()


def test_finished_speculation_is_delivered_instantly_and_committed_only_when_used():
    llm = FakeLLM()
    h, final = make(llm)
    h.prefetch("Shut the fuck up.", lambda: None)
    assert wait_until(lambda: h._spec is not None and h._spec.future.done())
    assert final.cached("Shut the fuck up.") is None  # kullanılmadan önbelleğe yazılmaz
    c = submit(h, "Shut the fuck up.")
    assert c.results == [(FINAL, "llm:Shut the fuck up.")]  # eşzamanlı, taslak bile gerekmedi
    assert final.cached("shut the fuck up.") == "llm:Shut the fuck up."
    assert len(llm.requests) == 1


def test_mismatched_speculation_is_never_shown_nor_cached():
    llm = FakeLLM(gated=True)
    draft = FakeDraft()
    h, final = make(llm, draft)
    h.prefetch("Get in the cab.", lambda: None)  # yanlış okuma
    assert wait_until(lambda: len(llm.requests) == 1)  # istek sürüyor
    c = submit(h, "Get in the car.")
    llm.release.set()
    assert c.done.wait(2)
    time.sleep(0.05)
    assert c.results == [(DRAFT, "yerel:Get in the car."), (FINAL, "llm:Get in the car.")]
    assert all("cab" not in text for _, text in c.results)
    assert [r[0] for r in llm.requests] == ["Get in the cab.", "Get in the car."]
    assert final.cached("Get in the cab.") is None  # yanlış okumanın çevirisi önbelleği kirletmez
    h.shutdown()


def test_queued_mismatched_speculation_is_never_sent():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    first = submit(h, "Previous line.")  # son çevirmen meşgul
    assert wait_until(lambda: len(llm.requests) == 1)
    h.prefetch("Get in the cab.", lambda: None)
    c = submit(h, "Get in the car.")
    llm.release.set()
    assert first.done.wait(2) and c.done.wait(2)
    assert [r[0] for r in llm.requests] == ["Previous line.", "Get in the car."]
    assert c.results == [(FINAL, "llm:Get in the car.")]
    h.shutdown()


def test_only_one_speculative_request_per_region_newer_read_supersedes():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    assert h.prefetch("Get in the ca", lambda: None)
    assert wait_until(lambda: len(llm.requests) == 1)  # başladı (iptal edilemez)
    assert h.prefetch("Get in the cab", lambda: None)
    assert h.prefetch("Get in the car.", lambda: None)  # öncekinin yerine geçer
    assert not h.prefetch("Get in the car.", lambda: None)  # aynı okuma: yeni istek yok
    assert wait_until(lambda: llm.cancelled == ["Get in the ca"])  # sürmekte olan yarıda kesildi
    c = submit(h, "Get in the car.")
    llm.release.set()
    assert c.done.wait(2)
    time.sleep(0.05)
    # Arada kalan "cab" hiç gönderilmedi, "ca" yarıda kesildi; aynı anda en fazla bir istek.
    assert [r[0] for r in llm.requests] == ["Get in the ca", "Get in the car."]
    assert llm.max_active == 1
    assert c.results == [(FINAL, "llm:Get in the car.")]
    h.shutdown()


def test_regions_speculate_independently_but_share_the_cache():
    llm = FakeLLM(gated=True)
    final = CachedTranslator(llm, None)
    main, phone = HybridTranslator(final, None), HybridTranslator(final, None)
    main.prefetch("Line one.", lambda: None)
    phone.prefetch("Text message.", lambda: None)
    llm.release.set()
    a, b = submit(main, "Line one."), submit(phone, "Text message.")
    assert a.done.wait(2) and b.done.wait(2)
    assert sorted(r[0] for r in llm.requests) == ["Line one.", "Text message."]


def test_context_is_evaluated_when_request_starts_so_history_is_current():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    history = []
    ctx = lambda: Dialogue("Line one.", history) if history else None  # noqa: E731
    first = submit(h, "Line one.")
    assert wait_until(lambda: len(llm.requests) == 1)
    h.prefetch("Line two.", ctx)  # önceki satırın çevirisi henüz gelmedi
    history.append(("Line one.", "llm:Line one."))  # ...şimdi geldi (on_result → geçmiş)
    llm.release.set()
    assert first.done.wait(2)
    assert wait_until(lambda: len(llm.requests) == 2)
    real = Dialogue("Line one.", history)
    c = submit(h, "Line two.", real)
    assert c.done.wait(2)
    assert llm.requests[1] == ("Line two.", "Line one.", (("Line one.", "llm:Line one."),))
    assert len(llm.requests) == 2  # aynı istem: yeniden kullanıldı


def test_unstarted_speculation_takes_the_confirmed_context():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    first = submit(h, "Line one.")
    assert wait_until(lambda: len(llm.requests) == 1)
    h.prefetch("Line two.", lambda: None)
    real = Dialogue("Line one.", [("Line one.", "bir")])
    c = submit(h, "Line two.", real)  # istek başlamadan doğrulandı
    llm.release.set()
    assert first.done.wait(2) and c.done.wait(2)
    assert llm.requests[1] == ("Line two.", "Line one.", (("Line one.", "bir"),))
    assert len(llm.requests) == 2


def test_started_speculation_with_different_history_is_not_reused():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    h.prefetch("Line two.", lambda: Dialogue("Line one."))
    assert wait_until(lambda: len(llm.requests) == 1)
    real = Dialogue("Line one.", [("Line one.", "bir")])  # geçmiş bu arada değişti: istem farklı
    c = submit(h, "Line two.", real)
    llm.release.set()
    assert c.done.wait(2)
    assert len(llm.requests) == 2 and llm.requests[1][2] == (("Line one.", "bir"),)
    h.shutdown()


def test_failed_speculation_is_not_adopted_real_request_is_sent():
    llm = FakeLLM(fail_first=True)
    h, _ = make(llm)
    h.prefetch("Hi there.", lambda: None)
    assert wait_until(lambda: h._spec is not None and h._spec.future.done())
    c = submit(h, "Hi there.")
    assert c.done.wait(2)
    assert c.results == [(FINAL, "llm:Hi there.")] and c.errors == []
    assert len(llm.requests) == 2


def test_adopted_speculation_that_fails_is_retried_once_normally():
    llm = FakeLLM(gated=True, fail_first=True)
    h, _ = make(llm, FakeDraft())
    h.prefetch("Hi there.", lambda: None)
    assert wait_until(lambda: len(llm.requests) == 1)
    c = submit(h, "Hi there.")
    assert wait_until(lambda: c.results == [(DRAFT, "yerel:Hi there.")])
    llm.release.set()
    assert c.done.wait(2)
    assert c.results == [(DRAFT, "yerel:Hi there."), (FINAL, "llm:Hi there.")] and c.errors == []
    assert len(llm.requests) == 2
    h.shutdown()


def test_real_errors_are_still_reported_after_adopted_speculation_fails():
    class Offline:
        cancellable = True
        calls = 0

        def translate(self, text, context, cancel=None):
            Offline.calls += 1
            time.sleep(0.02)
            raise ConnectionError("offline")

    h = HybridTranslator(CachedTranslator(Offline(), None), None)
    h.prefetch("Hi there.", lambda: None)
    c = submit(h, "Hi there.")
    assert c.done.wait(2)
    assert c.errors == [(FINAL, False)] and c.results == []
    assert Offline.calls == 2  # önceden çeviri + bir normal deneme, döngü yok
    h.shutdown()


def test_wasted_speculation_is_aborted_so_the_confirmed_line_does_not_wait():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    h.prefetch("Get in the car,", lambda: None)  # solma sırasında yarım okuma
    assert wait_until(lambda: len(llm.requests) == 1)
    c = submit(h, "Get in the car, we're leaving.")
    assert wait_until(lambda: llm.cancelled == ["Get in the car,"])  # ekran kartı hemen boşaldı
    assert wait_until(lambda: len(llm.requests) == 2)
    llm.release.set()
    assert c.done.wait(2)
    assert c.results == [(FINAL, "llm:Get in the car, we're leaving.")]
    h.shutdown()


def test_superseded_running_speculation_is_aborted():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    h.prefetch("Get in the", lambda: None)
    assert wait_until(lambda: len(llm.requests) == 1)
    h.prefetch("Get in the car.", lambda: None)
    assert wait_until(lambda: llm.cancelled == ["Get in the"])
    assert wait_until(lambda: [r[0] for r in llm.requests] == ["Get in the", "Get in the car."])
    llm.release.set()
    c = submit(h, "Get in the car.")
    assert c.done.wait(2) and c.results == [(FINAL, "llm:Get in the car.")]
    assert len(llm.requests) == 2
    h.shutdown()


def test_cancel_prefetch_drops_unsent_request():
    llm = FakeLLM(gated=True)
    h, _ = make(llm)
    first = submit(h, "Previous line.")
    assert wait_until(lambda: len(llm.requests) == 1)
    h.prefetch("Gone line.", lambda: None)
    h.cancel_prefetch()  # satır kayboldu
    llm.release.set()
    assert first.done.wait(2)
    time.sleep(0.05)
    assert [r[0] for r in llm.requests] == ["Previous line."]


def test_draft_prefetch_fills_draft_cache():
    draft = FakeDraft()
    h = HybridTranslator(None, draft)
    assert not h.prefetch("Get in the car.", lambda: None, draft=True)  # son çevirmen yok
    assert wait_until(lambda: h._draft_get("Get in the car.") is not None)
    c = submit(h, "Get in the car.")
    assert c.results == [(FINAL, "yerel:Get in the car.")]  # eşzamanlı
    assert draft.calls == ["Get in the car."]


def test_draft_is_not_prefetched_unless_asked():
    draft = FakeDraft()
    h = HybridTranslator(None, draft)
    h.prefetch("Get in the car.", lambda: None)
    time.sleep(0.05)
    assert draft.calls == []


def test_routed_lines_speculate_only_when_they_go_to_the_llm():
    llm = FakeLLM()
    h = HybridTranslator(CachedTranslator(llm, None), FakeDraft(), final_for=lambda t: "fuck" in t)
    assert not h.prefetch("Get in the car.", lambda: None)
    assert h.prefetch("Shut the fuck up.", lambda: None)
    assert wait_until(lambda: len(llm.requests) == 1)
    assert llm.requests[0][0] == "Shut the fuck up."


def test_cached_line_is_not_prefetched():
    llm = FakeLLM()
    h, final = make(llm)
    final.translate("Hi.")
    assert not h.prefetch("hi.", lambda: None)
    assert len(llm.requests) == 1


# --- Ollama: iptal edilebilir (akışlı) istek ------------------------------------------------------


class StreamingOllama:
    """Yanıtı belirteç belirteç akıtan sahte Ollama; istemci bağlantıyı kapatırsa fark eder."""

    def __init__(self, tokens, gap=0.02):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import json

        self.bodies, self.disconnected, self.sent = [], threading.Event(), 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.bodies.append(body)
                if not body["stream"]:
                    data = json.dumps({"message": {"content": "".join(tokens)}, "done": True}).encode()
                    self.send_response(200)
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("content-type", "application/x-ndjson")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                try:
                    for i, tok in enumerate(tokens + [None]):
                        event = {"message": {"content": tok or ""}, "done": tok is None}
                        line = (json.dumps(event) + "\n").encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                        self.wfile.flush()
                        fake.sent = i + 1
                        time.sleep(gap)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    fake.disconnected.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_ollama_streamed_request_matches_normal_request_and_output():
    from altyazi.llm import OllamaBackend

    fake = StreamingOllama(["Kapa ", "lan ", "çeneni!"], gap=0.001)
    try:
        backend = OllamaBackend("aya-expanse:8b", fake.url)
        context = Dialogue("Where's my money?", [("Where's my money?", "Param nerede?")])
        normal = backend.translate("Shut the fuck up!", context)
        streamed = backend.translate("Shut the fuck up!", context, cancel=threading.Event())
        assert normal == streamed == "Kapa lan çeneni!"
        a, b = fake.bodies
        assert a.pop("stream") is False and b.pop("stream") is True
        assert a == b  # istem ve seçenekler birebir aynı (önbellek/önek aynı)
    finally:
        fake.server.shutdown()


def test_ollama_speculative_request_is_aborted_on_cancel():
    from altyazi.llm import Cancelled, OllamaBackend

    fake = StreamingOllama([f"t{i} " for i in range(200)], gap=0.01)
    try:
        backend = OllamaBackend("aya-expanse:8b", fake.url)
        cancel = threading.Event()
        errors = []

        def run():
            try:
                backend.translate("Get in the car,", None, cancel=cancel)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=run)
        t.start()
        assert wait_until(lambda: fake.sent >= 3)
        cancel.set()
        t.join(2)
        assert not t.is_alive() and isinstance(errors[0], Cancelled)
        assert fake.disconnected.wait(2)  # sunucu bağlantının kapandığını gördü: üretim durur
        assert fake.sent < 100
    finally:
        fake.server.shutdown()
