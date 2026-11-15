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
