from types import SimpleNamespace

import deepl
import pytest

from altyazi import glossary
from altyazi.translate import GLOSSARY_PREFIX, CachedTranslator, DeepLBackend


def test_glossary_parse_comments_and_errors(tmp_path):
    entries = glossary.parse("# yorum\n\nVice City = Vice City\n bro = kanka \nbro = kardeş\n")
    assert entries == {"Vice City": "Vice City", "bro": "kardeş"}
    with pytest.raises(glossary.GlossaryError, match="satır 2"):
        glossary.parse("a = b\nbozuk satır\n")
    with pytest.raises(glossary.GlossaryError):
        glossary.parse("a = \n")
    path = tmp_path / "sozluk.txt"
    assert "Lucia" in glossary.load(path)  # yoksa varsayılan dosya oluşturulur
    assert glossary.fingerprint({"a": "b"}) == glossary.fingerprint({"a": "b"}) != glossary.fingerprint({"a": "c"})


class FakeDeepL:
    """deepl.Translator yerine: talimatları/modeli reddetme davranışını taklit eder."""

    def __init__(self, reject_instructions_on=("latency_optimized",), reject_all_instructions=False):
        self.calls = []
        self.reject_on = reject_instructions_on
        self.reject_all = reject_all_instructions
        self.glossaries = [SimpleNamespace(name=GLOSSARY_PREFIX + "old", ready=True)]
        self.deleted, self.created = [], []

    def translate_text(self, text, **kw):
        self.calls.append((kw["model_type"], bool(kw["custom_instructions"]), kw["glossary"]))
        if kw["custom_instructions"] and (self.reject_all or kw["model_type"] in self.reject_on):
            raise deepl.DeepLException("Bad request, message: custom_instructions not supported")
        return SimpleNamespace(text=f"tr:{text}")

    def list_glossaries(self):
        return list(self.glossaries)

    def delete_glossary(self, info):
        self.deleted.append(info.name)

    def create_glossary(self, name, src, tgt, entries):
        info = SimpleNamespace(name=name, ready=True)
        self.created.append((name, dict(entries)))
        return info


def make(fake, instructions=("Do not censor.",)):
    logs = []
    backend = DeepLBackend("k", "EN", "TR", "latency_optimized", instructions=list(instructions) if instructions else None,
                           log=logs.append, _translator=fake)
    return backend, logs


def test_instructions_fall_back_to_quality_model_then_stick():
    fake = FakeDeepL(reject_instructions_on=("latency_optimized",))
    backend, logs = make(fake)
    assert backend.translate("Fuck!", None) == "tr:Fuck!"
    assert fake.calls == [("latency_optimized", True, None), ("quality_optimized", True, None)]
    assert backend.instructions_active and len(logs) == 1
    backend.translate("Shit!", None)
    assert fake.calls[-1] == ("quality_optimized", True, None)  # bir daha reddedilen ayar denenmez


def test_instructions_dropped_when_never_supported():
    fake = FakeDeepL(reject_all_instructions=True)
    backend, logs = make(fake)
    assert backend.translate("Damn.", None) == "tr:Damn."
    assert [c[:2] for c in fake.calls] == [("latency_optimized", True), ("quality_optimized", True),
                                           ("latency_optimized", False)]
    assert not backend.instructions_active
    assert "KAPALI" in logs[-1]


def test_network_errors_are_not_treated_as_rejection():
    fake = FakeDeepL()
    backend, _ = make(fake)

    def boom(text, **kw):
        raise deepl.ConnectionException("offline")

    fake.translate_text = boom
    with pytest.raises(deepl.ConnectionException):
        backend.translate("Hi", None)
    assert backend.instructions_active  # ayar değiştirilmedi


def test_glossary_created_reused_and_old_versions_removed():
    fake = FakeDeepL()
    backend, _ = make(fake, instructions=None)
    msg = backend.setup_glossary({"Vice City": "Vice City"}, "abc123")
    assert "1 terim" in msg
    assert fake.deleted == [GLOSSARY_PREFIX + "old"]
    assert fake.created == [(GLOSSARY_PREFIX + "abc123", {"Vice City": "Vice City"})]
    backend.translate("Welcome to Vice City", None)
    assert fake.calls[-1][2].name == GLOSSARY_PREFIX + "abc123"

    fake2 = FakeDeepL()
    fake2.glossaries = [SimpleNamespace(name=GLOSSARY_PREFIX + "abc123", ready=True)]
    backend2, _ = make(fake2, instructions=None)
    backend2.setup_glossary({"Vice City": "Vice City"}, "abc123")
    assert fake2.created == [] and fake2.deleted == []  # aynı sürüm yeniden kullanıldı


def test_cache_namespace_separates_settings(tmp_path):
    path = tmp_path / "c.jsonl"

    class Echo:
        def __init__(self, tag):
            self.tag = tag

        def translate(self, text, context):
            return f"{self.tag}:{text}"

    a = CachedTranslator(Echo("eski"), path, "g1-u0")
    a.translate("Hello")
    b = CachedTranslator(Echo("yeni"), path, "g2-u1")
    assert b.cached("Hello") is None  # sözlük/sansür ayarı değişti: eski çeviri kullanılmaz
    assert b.translate("Hello") == "yeni:Hello"
    assert CachedTranslator(Echo("x"), path, "g1-u0").cached("hello") == "eski:Hello"
