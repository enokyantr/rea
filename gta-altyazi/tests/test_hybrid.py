import threading
import time

import pytest

from altyazi.hybrid import DRAFT, FINAL, HybridTranslator
from altyazi.translate import CachedTranslator


class Gate:
    """Çeviriyi testin istediği ana kadar bekleten sahte çevirmen."""

    def __init__(self, prefix, fail=False):
        self.prefix, self.fail = prefix, fail
        self.release = threading.Event()
        self.calls = 0

    def translate(self, text, context):
        self.calls += 1
        assert self.release.wait(2)
        if self.fail:
            raise ConnectionError("offline")
        return f"{self.prefix}:{text}"


class Collector:
    def __init__(self):
        self.results, self.errors = [], []
        self.done = threading.Event()

    def on_result(self, stage, text, took):
        self.results.append((stage, text))
        if stage == FINAL:
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
        time.sleep(0.005)
    return False


def make(draft=None, final_backend=None):
    final = CachedTranslator(final_backend, None) if final_backend else None
    return HybridTranslator(final, draft), final


def test_draft_first_then_deepl_replaces_it():
    local, deepl = Gate("yerel"), Gate("deepl")
    h, _ = make(local, deepl)
    c = Collector()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error)
    local.release.set()
    assert wait_until(lambda: c.results == [(DRAFT, "yerel:Hi")])
    deepl.release.set()
    assert c.done.wait(2)
    assert c.results == [(DRAFT, "yerel:Hi"), (FINAL, "deepl:Hi")]


def test_late_draft_is_dropped_when_deepl_wins():
    local, deepl = Gate("yerel"), Gate("deepl")
    h, _ = make(local, deepl)
    c = Collector()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error)
    deepl.release.set()
    assert c.done.wait(2)
    local.release.set()
    h.shutdown()
    time.sleep(0.05)
    assert c.results == [(FINAL, "deepl:Hi")]


def test_cache_hit_is_instant_and_skips_local_model():
    local, deepl = Gate("yerel"), Gate("deepl")
    deepl.release.set()
    h, final = make(local, deepl)
    final.translate("Hi")  # önbelleğe al
    c = Collector()
    h.submit("hi", None, lambda: True, c.on_result, c.on_error)
    assert c.results == [(FINAL, "deepl:Hi")]  # eşzamanlı, bekleme yok
    assert local.calls == 0


def test_deepl_failure_keeps_draft_on_screen():
    local, deepl = Gate("yerel"), Gate("deepl", fail=True)
    local.release.set()
    h, _ = make(local, deepl)
    c = Collector()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error)
    assert wait_until(lambda: c.results == [(DRAFT, "yerel:Hi")])
    deepl.release.set()
    assert c.done.wait(2)
    assert c.errors == [(FINAL, True)]


def test_stale_line_delivers_nothing():
    local, deepl = Gate("yerel"), Gate("deepl")
    current = {"v": True}
    h, _ = make(local, deepl)
    c = Collector()
    h.submit("Hi", None, lambda: current["v"], c.on_result, c.on_error)
    current["v"] = False  # bu arada yeni satır geldi
    local.release.set()
    deepl.release.set()
    h.shutdown()
    time.sleep(0.1)
    assert c.results == []


def test_local_only_mode_result_is_final():
    local = Gate("yerel")
    local.release.set()
    h, _ = make(local, None)
    assert h.mode == "local"
    c = Collector()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error)
    assert c.done.wait(2)
    assert c.results == [(FINAL, "yerel:Hi")]


def test_requires_a_translator():
    with pytest.raises(ValueError):
        HybridTranslator(None, None)


def test_final_only_for_selected_lines_draft_is_final_otherwise():
    local, deepl = Gate("yerel"), Gate("llm")
    local.release.set()
    deepl.release.set()
    h = HybridTranslator(CachedTranslator(deepl, None), local, final_for=lambda t: "fuck" in t)
    c = Collector()
    h.submit("Get in the car", None, lambda: True, c.on_result, c.on_error)
    assert c.done.wait(2)
    assert c.results == [(FINAL, "yerel:Get in the car")]  # küfürsüz: taslak nihai, LLM çağrılmadı
    assert deepl.calls == 0
    c2 = Collector()
    h.submit("Shut the fuck up", None, lambda: True, c2.on_result, c2.on_error)
    assert c2.done.wait(2)
    assert c2.results[-1] == (FINAL, "llm:Shut the fuck up") and deepl.calls == 1


def test_draft_running_is_set_only_while_the_local_model_translates():
    local = Gate("yerel")
    h, _ = make(local, None)
    c = Collector()
    assert not h.draft_running()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error)
    assert wait_until(h.draft_running)  # OCR bu sırada işlemciyi bırakır
    local.release.set()
    assert c.done.wait(2)
    assert wait_until(lambda: not h.draft_running())


def test_repeated_line_uses_draft_cache_instantly():
    local = Gate("yerel")
    local.release.set()
    h, _ = make(local, None)
    c = Collector()
    h.submit("Get in the car.", None, lambda: True, c.on_result, c.on_error)
    assert c.done.wait(2) and local.calls == 1
    c2 = Collector()
    h.submit("Get in the car.", None, lambda: True, c2.on_result, c2.on_error)
    assert c2.results == [(FINAL, "yerel:Get in the car.")]  # eşzamanlı, model çağrılmadı
    assert local.calls == 1


def test_backends_that_ignore_source_accept_it():
    deepl = Gate("deepl")
    deepl.release.set()
    h, final = make(None, deepl)
    c = Collector()
    h.submit("Hi", None, lambda: True, c.on_result, c.on_error, source="asr")
    assert c.done.wait(2) and c.results == [(FINAL, "deepl:Hi")]
    # Kaynak DeepL'in çevirisini değiştirmez: OCR satırı aynı önbellek kaydını kullanır
    assert final.cached("Hi") == "deepl:Hi" and deepl.calls == 1


def test_final_only_lines_skip_the_draft_translator():
    import threading
    from altyazi.hybrid import FINAL, HybridTranslator

    class Draft:
        calls = []

        def translate(self, text, context):
            self.calls.append(text)
            return "taslak"

    class Final:
        def translate(self, text, context, source="ocr"):
            return "son"

    from altyazi.translate import CachedTranslator

    draft = Draft()
    tr = HybridTranslator(CachedTranslator(Final(), None), draft, final_only=lambda t: "mija" in t)
    got, done = [], threading.Event()

    def on_result(stage, text, took):
        got.append((stage, text))
        if stage == FINAL:
            done.set()
    tr.submit("Que pasa, mija?", None, lambda: True, on_result, lambda *a: None)
    assert done.wait(5)
    assert got == [(FINAL, "son")] and draft.calls == []
    done.clear()
    tr.submit("Hello there.", None, lambda: True, on_result, lambda *a: None)
    assert done.wait(5)
    assert draft.calls == ["Hello there."]  # normal satırda taslak yine çalışır
    assert got[0] == (FINAL, "son")
    tr.shutdown()


def test_stale_line_cancels_its_running_final_request_and_stays_silent():
    import threading
    import time

    from altyazi.hybrid import HybridTranslator
    from altyazi.translate import CachedTranslator

    started, saw_cancel = threading.Event(), threading.Event()

    class Slow:
        cancellable = True

        def translate(self, text, context, cancel=None):
            started.set()
            for _ in range(200):
                if cancel is not None and cancel.is_set():
                    saw_cancel.set()
                    raise RuntimeError("iptal")
                time.sleep(0.01)
            return "uzun"

    current = {"on": True}
    got, errors = [], []
    tr = HybridTranslator(CachedTranslator(Slow(), None), None)
    tr.submit("A long line.", None, lambda: current["on"], lambda *a: got.append(a),
              lambda *a: errors.append(a))
    assert started.wait(5)
    current["on"] = False  # yeni satır geldi
    assert saw_cancel.wait(2)  # üretim yarıda kesildi (2 sn'lik tam çeviri beklenmedi)
    time.sleep(0.1)
    assert got == [] and errors == []
    tr.shutdown()


def test_low_priority_translator_waits_for_the_main_subtitle():
    import threading

    import altyazi.hybrid as hy
    from altyazi.hybrid import HybridTranslator
    from altyazi.translate import CachedTranslator

    class Quick:
        def translate(self, text, context):
            return "çeviri"

    main_busy = threading.Event()
    low = HybridTranslator(CachedTranslator(Quick(), None), None)
    low.low_priority = True
    done = threading.Event()
    with hy._PRIORITY_LOCK:
        hy._main_active = 1
    try:
        low.submit("Hello there.", None, lambda: True, lambda *a: done.set(), lambda *a: None)
        assert not done.wait(0.3)  # ana altyazı bitene kadar bekler
        with hy._PRIORITY_LOCK:
            hy._main_active = 0
        assert done.wait(3)
    finally:
        with hy._PRIORITY_LOCK:
            hy._main_active = 0
        main_busy.set()
        low.shutdown()
