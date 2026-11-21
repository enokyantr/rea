"""Sözlük kaynakları/önceliği, satırda terim eşleşmesi, satıra özel önbellek anahtarı ve DeepL sınırı."""

import time
from types import SimpleNamespace

import pytest

from altyazi import glossary
from altyazi.glossary import Matcher
from altyazi.hybrid import FINAL, HybridTranslator
from altyazi.llm import OllamaBackend
from altyazi.translate import CachedTranslator, DeepLBackend


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    write(d / glossary.GENERAL_FILE, "boss = patron\nRoxy = Genel Roxy\nsafehouse = sığınak\n")
    write(d / "gta6_isimler.txt", "Roxy = Roxy\nVice City = Vice City\n")
    write(d / "baska_isimler.txt", "Lamar = Lamar\n")
    return d


def test_priority_general_then_game_names_then_user(tmp_path, data_dir):
    user = write(tmp_path / "sozluk.txt", "Vice City = Vays Siti\nbro = kanka\n")
    entries = glossary.load(user, data_dir)
    assert entries["boss"] == "patron"  # genel sözlük
    assert entries["Roxy"] == "Roxy"  # oyun adı genel sözlüğü geçer
    assert entries["Lamar"] == "Lamar"  # tüm *_isimler.txt dosyaları
    assert entries["Vice City"] == "Vays Siti"  # kullanıcı her şeyi geçer
    assert entries["bro"] == "kanka"
    # core: genel sözlük hariç (DeepL sınırı ve taslak/son çevirmen yönlendirmesi için)
    assert "boss" not in entries.core and entries.core["Vice City"] == "Vays Siti" and "Roxy" in entries.core


def test_missing_bundled_files_are_optional(tmp_path):
    user = write(tmp_path / "sozluk.txt", "bro = kanka\n")
    assert glossary.load(user, tmp_path / "yok") == {"bro": "kanka"}
    empty = tmp_path / "bos"
    empty.mkdir()
    assert glossary.load(user, empty) == {"bro": "kanka"}


def test_bundled_file_error_names_the_file(tmp_path, data_dir):
    write(data_dir / glossary.GENERAL_FILE, "boss = patron\nbozuk\n")
    with pytest.raises(glossary.GlossaryError, match=r"oyun_sozlugu\.txt satır 2"):
        glossary.load(write(tmp_path / "sozluk.txt", ""), data_dir)


def test_bundled_data_is_found_next_to_the_module_and_parses():
    # Programla gelen tüm sözlükler hatasız okunur (çalışma klasöründen bağımsız, modüle göre).
    assert glossary.DATA_DIR.is_dir()
    general, names = glossary.load_bundled()
    assert names["Dre'Quan Priest"] == "Dre'Quan Priest" and names["Leonida Keys"] == "Leonida Keys"
    assert all(src and dst for src, dst in {**general, **names}.items())


@pytest.mark.parametrize("line, expected", [
    ("Welcome to vice city, man.", ("Vice City",)),  # büyük-küçük harf
    ("Welcome to Vice\n  City.", ("Vice City",)),  # OCR satır sonu/boşluk
    ("Mike likes vices.", ()),  # tam kelime: "Ike", "Vice" alt dizgede eşleşmez
    ("Ike, take the Leonida Keys exit.", ("Ike", "Leonida Keys")),  # uzun ifade önce, "Leonida" ayrıca yok
    ("Leonida is hot.", ("Leonida",)),
    ("Dre'Quan Priest is here.", ("Dre'Quan Priest",)),
    ("Dre’Quan's car.", ("Dre'Quan",)),  # tipografik kesme + iyelik eki
    ("Dre'Quan said so.", ("Dre'Quan",)),  # "Dre" ve "Quan" Dre'Quan içinde eşleşmez
    ("Lucia's gun. Jason'll drive.", ("Lucia", "Jason")),
    ("Bae-Luxe and Bae Luxe", ("Bae-Luxe",)),  # ayraç da eşleşmeli
    ("Lucia, Lucia!", ("Lucia",)),
])
def test_matcher_whole_words_longest_first_apostrophes_case(line, expected):
    m = Matcher({"Ike": "Ike", "Vice City": "Vice City", "Vice": "Mengene", "Leonida": "Leonida",
                 "Leonida Keys": "Leonida Keys", "Dre'Quan": "Dre'Quan", "Dre'Quan Priest": "Dre'Quan Priest",
                 "Dre": "Dre", "Quan": "Kuan", "Lucia": "Lucia", "Jason": "Jason", "Bae-Luxe": "Bae-Luxe"})
    assert m.terms(line) == expected


def test_needs_final_ignores_general_glossary(tmp_path, data_dir):
    entries = glossary.load(write(tmp_path / "sozluk.txt", "Grove Street = Grove Sokağı\n"), data_dir)
    route = glossary.needs_final(entries)
    assert route("Meet me on grove street.")
    assert not route("The boss is in the safehouse.")  # genel terimler her satırı LLM'e göndermesin
    assert not route("Roxy is in Vice City.")  # kendisiyle eşlenen adlar


def ollama(entries):
    return OllamaBackend("aya-expanse:8b", "http://127.0.0.1:9", glossary=entries)


def test_cache_key_depends_only_on_terms_matched_in_the_line(tmp_path):
    base = {"Vice City": "Vice City", "boss": "patron"}
    key = CachedTranslator(ollama(base), None).key
    line = "Welcome to Vice City, boss."
    # İlgisiz terim eklemek: satır anahtarı aynı (önbellek silinmez)
    assert CachedTranslator(ollama({**base, "safehouse": "sığınak"}), None).key(line) == key(line)
    # Satırda geçen terimin karşılığı değişince anahtar değişir
    assert CachedTranslator(ollama({**base, "boss": "şef"}), None).key(line) != key(line)
    # Satırda geçen yeni (daha uzun) terim eklenince de
    assert CachedTranslator(ollama({**base, "Welcome to Vice City": "Vice City'ye hoş geldin"}),
                            None).key(line) != key(line)
    # Terimsiz satır: yalnızca ad alanı + metin
    assert "\x1ft" not in key("Get in the car.")
    assert key("welcome to  VICE city, boss.") == key(line)


def test_cache_is_reused_after_unrelated_glossary_change(tmp_path):
    path = tmp_path / "c.jsonl"

    class Recorder(OllamaBackend):
        calls = []

        def translate(self, text, context, cancel=None):
            self.calls.append(text)
            return f"tr:{text}"

    def make(entries):
        return CachedTranslator(Recorder("m", "http://127.0.0.1:9", glossary=entries), path)

    make({"Vice City": "Vice City", "boss": "patron"}).translate("Welcome to Vice City.")
    make({"Vice City": "Vice City", "boss": "şef"}).translate("Welcome to Vice City.")  # ilgisiz değişiklik
    assert Recorder.calls == ["Welcome to Vice City."]
    make({"Vice City": "Vays Siti", "boss": "şef"}).translate("Welcome to Vice City.")
    assert Recorder.calls == ["Welcome to Vice City."] * 2


def test_speculative_and_confirmed_paths_use_the_same_term_aware_key():
    class LLM:
        cancellable = True

        def __init__(self):
            self.inner = ollama({"Vice City": "Vice City"})
            self.requests = []

        def cache_namespace(self):
            return self.inner.cache_namespace()

        def cache_terms(self, text):
            return self.inner.cache_terms(text)

        def translate(self, text, context, cancel=None):
            self.requests.append(text)
            return f"llm:{text}"

    llm = LLM()
    final = CachedTranslator(llm, None)
    h = HybridTranslator(final, None)
    assert h.prefetch("Welcome to Vice City.", lambda: None)
    end = time.monotonic() + 2
    while not (h._spec is not None and h._spec.future.done()):
        assert time.monotonic() < end
        time.sleep(0.002)
    results = []
    h.submit("welcome to vice  city.", None, lambda: True, lambda s, t, _: results.append((s, t)),
             lambda *a: results.append(a))
    assert results == [(FINAL, "llm:Welcome to Vice City.")] and len(llm.requests) == 1
    assert final.cached("Welcome to Vice City.") == "llm:Welcome to Vice City."
    h.shutdown()


class FakeDeepL:
    def __init__(self):
        self.created = []

    def list_glossaries(self):
        return []

    def create_glossary(self, name, src, tgt, entries):
        self.created.append(dict(entries))
        return SimpleNamespace(name=name, ready=True)


def test_deepl_never_gets_the_general_game_glossary():
    fake = FakeDeepL()
    backend = DeepLBackend("k", "EN", "TR", _translator=fake)
    core = {"Vice City": "Vice City", "bro": "kanka"}
    entries = glossary.Glossary({"boss fight": "boss savaşı", **core}, core=core)
    msg = backend.setup_glossary(entries, glossary.fingerprint(entries))
    assert fake.created == [core] and "2 terim" in msg and "genel oyun sözlüğü DeepL'e gönderilmez" in msg
    assert backend.cache_namespace() == f"g{glossary.fingerprint(core)}-u0"

    plain = {"Vice City": "Vice City"}  # düz sözlük (çekirdek bilgisi yok): olduğu gibi
    backend.setup_glossary(plain, glossary.fingerprint(plain))
    assert fake.created[-1] == plain
