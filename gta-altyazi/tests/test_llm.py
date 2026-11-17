"""OllamaBackend: yerel sahte Ollama sunucusuna karşı (gönderilen istek ve çıktı temizliği doğrulanır)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from altyazi.llm import OllamaBackend, OllamaUnavailable, clean_output
from altyazi.translate import CachedTranslator


class FakeOllama:
    def __init__(self, models=("aya-expanse:8b",)):
        self.models = list(models)
        self.chats: list[dict] = []
        self.reply = "Kapa lan şu çeneni, göt herif!"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, body, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._json({"models": [{"name": m} for m in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                if self.path == "/api/chat":
                    if body["model"] not in fake.models:
                        return self._json({"error": "model not found"}, 404)
                    fake.chats.append(body)
                    return self._json({"message": {"role": "assistant", "content": fake.reply}, "done": True})
                if self.path == "/api/pull":
                    lines = [{"status": "pulling manifest"},
                             {"status": "downloading", "total": 100, "completed": 50},
                             {"status": "downloading", "total": 100, "completed": 100},
                             {"status": "success"}]
                    data = "".join(json.dumps(x) + "\n" for x in lines).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/x-ndjson")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    fake.models.append(body["model"])

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture
def fake():
    server = FakeOllama()
    yield server
    server.server.shutdown()


def test_chat_request_has_rules_examples_context_and_keeps_model_loaded(fake):
    backend = OllamaBackend("aya-expanse:8b", fake.url, glossary={"Vice City": "Vice City"})
    assert backend.translate("Shut the fuck up, asshole!", "Where's my money?") == "Kapa lan şu çeneni, göt herif!"
    body = fake.chats[-1]
    msgs = body["messages"]
    assert msgs[0]["role"] == "system" and "Never soften" in msgs[0]["content"]
    assert "<glossary>" in msgs[0]["content"]  # kural sistemde, terimler yalnızca satırda geçerse
    assert "Vice City = Vice City" not in msgs[0]["content"]
    assert any(m["role"] == "assistant" and "siktiğimin" in m["content"] for m in msgs)  # üslup örneği
    assert "never ADD swearing" in msgs[0]["content"]  # küfürsüz satıra küfür eklenmesin
    # Bağlamın çevrilmediğini gösteren örnek: bağlamlı kullanıcı mesajına yalnızca satırın çevirisi
    assert {"role": "assistant", "content": "Sağdaki ilk çıkıştan sap."} in msgs
    assert msgs[-1] == {"role": "user",
                        "content": "<context>Where's my money?</context>\n<line>Shut the fuck up, asshole!</line>"}
    assert body["stream"] is False and body["keep_alive"] == "60m"


def test_censored_mode_has_no_profanity_rule_or_examples(fake):
    backend = OllamaBackend("aya-expanse:8b", fake.url, uncensored=False)
    backend.translate("Hi", None)
    msgs = fake.chats[-1]["messages"]
    assert "Never soften" not in msgs[0]["content"]
    assert not any("siktiğimin" in m["content"] for m in msgs)


def test_model_detection_and_pull(fake):
    backend = OllamaBackend("gemma3:12b", fake.url)
    assert not backend.has_model()
    progress = []
    backend.pull(progress.append)
    assert progress[0] == "pulling manifest" and "downloading 50%" in progress and progress[-1] == "success"
    assert backend.has_model()
    assert OllamaBackend("aya-expanse", fake.url).has_model() is False  # :latest ile eşleşmez
    fake.models.append("aya-expanse:latest")
    assert OllamaBackend("aya-expanse", fake.url).has_model()


def test_missing_model_and_stopped_ollama_give_clear_errors(fake):
    with pytest.raises(OllamaUnavailable, match="ollama pull"):
        OllamaBackend("yok:1b", fake.url).translate("Hi", None)
    stopped = OllamaBackend("aya-expanse:8b", "http://127.0.0.1:9")
    with pytest.raises(OllamaUnavailable, match="ollama.com/download"):
        stopped.has_model()


def test_cache_namespace_tracks_model_and_rules(fake, tmp_path):
    a = OllamaBackend("aya-expanse:8b", fake.url)
    assert a.cache_namespace() != OllamaBackend("gemma3:12b", fake.url).cache_namespace()
    assert a.cache_namespace() != OllamaBackend("aya-expanse:8b", fake.url, uncensored=False).cache_namespace()
    cache = CachedTranslator(a, tmp_path / "c.jsonl")
    cache.translate("Hello")
    cache.translate("hello")
    assert len(fake.chats) == 1


@pytest.mark.parametrize("raw,expected", [
    ("Kapa çeneni!", "Kapa çeneni!"),
    ('"Kapa çeneni!"', "Kapa çeneni!"),
    ("Türkçe: Kapa çeneni!", "Kapa çeneni!"),
    ("<line>Kapa çeneni!</line>", "Kapa çeneni!"),
    ("Kapa çeneni!\n\nNot: 'shut up' deyimi...", "Kapa çeneni!"),
    ("“Hassiktir kanka.”", "Hassiktir kanka."),
    ("Önce şunu yap.\nSonra git.", "Önce şunu yap. Sonra git."),
    ('Bu ne amına koyayım kanka? (Not: "nigga" kelimesi Türkçe\'ye doğrudan çevrilemez...)',
     "Bu ne amına koyayım kanka?"),
    ("Kapa çeneni lan.\nNote: kept the profanity.", "Kapa çeneni lan."),
    ("Bu (yani şu) iş.", "Bu (yani şu) iş."),
])
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


def test_num_ctx_and_context_switch(fake):
    backend = OllamaBackend("aya-expanse:8b", fake.url, num_ctx=1024, use_context=False)
    backend.translate("Hi", "Previous line.")
    body = fake.chats[-1]
    assert body["options"]["num_ctx"] == 1024
    assert body["messages"][-1]["content"] == "<line>Hi</line>"  # bağlam gönderilmedi
    backend.translate("or lawyer she fucking with.", "if she ever stop fucking with that brain surgeon")
    assert fake.chats[-1]["messages"][-1]["content"].startswith("<context>")  # yarım cümle: bağlamla
    assert backend.cache_namespace() != OllamaBackend("aya-expanse:8b", fake.url).cache_namespace()


@pytest.mark.parametrize("reply,meta", [
    # Gerçek loglardan: çeviri yerine açıklama/ret
    ('(Not translated as per guidelines. In Turkish street talk, "nigga" would be "it" or "şerefsiz".)', True),
    ("Bu cümle anlaşılmaz ve hatalı görünüyor. Orijinal metinde bulunmadığı için Türkçe'ye çeviremem.", True),
    ("Bu mesajın anlamını anlamıyorum. Görünüşe göre bir hata veya teknik sorun var. Lütfen daha açık bir "
     "talimat veya soru ile geri dönün.", True),
    ("Seni anlamıyorum, ne diyorsun lan?", False),
    ("Direksiyonu çevir, hadi!", False),
    ("Siktir git lan.", False),
    ("Lan...", False),
])
def test_meta_replies_are_detected(reply, meta):
    from altyazi.llm import is_meta_reply

    assert is_meta_reply(reply) is meta


def test_ensure_running_starts_ollama_when_it_is_down(monkeypatch):
    import altyazi.llm as llm

    backend = OllamaBackend("aya-expanse:8b", "http://127.0.0.1:9", use_context=False)
    alive = iter([False, False, True])
    monkeypatch.setattr(backend, "_alive", lambda: next(alive))
    monkeypatch.setattr(llm, "_find_ollama", lambda: "ollama.exe")
    started = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: started.append(args))
    logs = []
    assert backend.ensure_running(logs.append, wait_s=2)
    assert started == [["ollama.exe", "serve"]] and logs


def test_ensure_running_does_nothing_when_ollama_is_up(monkeypatch):
    backend = OllamaBackend("aya-expanse:8b", "http://127.0.0.1:9", use_context=False)
    monkeypatch.setattr(backend, "_alive", lambda: True)
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("başlatılmamalı")))
    assert backend.ensure_running(lambda m: None)


@pytest.mark.parametrize("previous,line,expected", [
    ("if she ever stop fucking with that brain surgeon", "or lawyer she fucking with.", True),
    ("Maybe if you got rid of that haircut you got", "you'd get some money.", True),
    ("I'll see you at work.", "Ah, man, don't hate me.", False),
    (None, "or lawyer she fucking with.", False),
])
def test_is_continuation(previous, line, expected):
    from altyazi.llm import is_continuation

    assert is_continuation(previous, line) is expected


def test_glossary_terms_with_own_translation_go_to_final():
    from altyazi.glossary import needs_final

    route = needs_final({"Vice City": "Vice City", "Grove Street": "Grove Sokağı"})
    assert route("Meet me on grove street tonight.")
    assert not route("Welcome to Vice City.")  # kendisiyle eşlenen isim: taslak model de korur


def test_dialogue_history_is_sent_as_conversation_turns(fake):
    from altyazi.textproc import Dialogue

    backend = OllamaBackend("aya-expanse:8b", fake.url, use_context=False)
    history = [("Who's that guy?", "O adam kim?"), ("That's my cousin, Lamar.", "Bu benim kuzenim, Lamar.")]
    backend.translate("He's cool, trust me.", Dialogue("That's my cousin, Lamar.", history))
    messages = fake.chats[-1]["messages"]
    assert messages[-1]["content"] == "<line>He's cool, trust me.</line>"  # yalnızca yeni satır çevrilir
    tail = [m["content"] for m in messages[-5:-1]]
    assert tail == ["<line>Who's that guy?</line>", "O adam kim?",
                    "<line>That's my cousin, Lamar.</line>", "Bu benim kuzenim, Lamar."]
    assert "<context>" not in messages[-1]["content"]


def test_only_matched_glossary_terms_go_into_the_user_message_and_system_stays_identical(fake):
    from altyazi.textproc import Dialogue

    entries = {"Vice City": "Vice City", "boss": "patron", "safehouse": "sığınak",
               **{f"term{i}": f"terim{i}" for i in range(400)}}
    backend = OllamaBackend("aya-expanse:8b", fake.url, glossary=entries, use_context=False)
    backend.translate("Hey boss, welcome to Vice City.", None)
    backend.translate("Get in the car.", None)
    history = [("Where's the safehouse?", "Sığınak nerede?")]
    backend.translate("The Safehouse is close, boss.", Dialogue("Where's the safehouse?", history))
    first, plain, third = (c["messages"] for c in fake.chats[-3:])
    # Sistem istemi ve örnekler (önek) her satırda birebir aynı: Ollama önbelleği yeniden kullanır
    assert first[:-1] == plain[:-1] == third[:-3]
    assert "term1" not in first[0]["content"] and "boss = patron" not in first[0]["content"]
    assert first[-1]["content"] == ("<glossary>\nboss = patron\nVice City = Vice City\n</glossary>\n"
                                    "<line>Hey boss, welcome to Vice City.</line>")
    assert plain[-1]["content"] == "<line>Get in the car.</line>"
    # Geçmiş turları değişmez (terim eklenmez); terimler yalnızca yeni satırda
    assert third[-3:-1] == [{"role": "user", "content": "<line>Where's the safehouse?</line>"},
                            {"role": "assistant", "content": "Sığınak nerede?"}]
    assert third[-1]["content"] == ("<glossary>\nsafehouse = sığınak\nboss = patron\n</glossary>\n"
                                    "<line>The Safehouse is close, boss.</line>")


def test_glossary_tag_is_cleaned_from_output():
    assert clean_output("<glossary></glossary>Vice City'ye hoş geldin.") == "Vice City'ye hoş geldin."


def test_num_gpu_is_sent_only_when_set(fake):
    OllamaBackend("aya-expanse:8b", fake.url).translate("Hi", None)
    assert "num_gpu" not in fake.chats[-1]["options"]
    OllamaBackend("aya-expanse:8b", fake.url, num_gpu=12).translate("Hi", None)
    assert fake.chats[-1]["options"]["num_gpu"] == 12
    OllamaBackend("aya-expanse:8b", fake.url, num_gpu=0).translate("Hi", None)  # 0 = tamamen işlemci
    assert fake.chats[-1]["options"]["num_gpu"] == 0
