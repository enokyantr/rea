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
    assert "Vice City = Vice City" in msgs[0]["content"]
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
])
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


def test_num_ctx_and_context_switch(fake):
    backend = OllamaBackend("aya-expanse:8b", fake.url, num_ctx=1024, use_context=False)
    backend.translate("Hi", "Previous line")
    body = fake.chats[-1]
    assert body["options"]["num_ctx"] == 1024
    assert body["messages"][-1]["content"] == "<line>Hi</line>"  # bağlam gönderilmedi
    assert backend.cache_namespace() != OllamaBackend("aya-expanse:8b", fake.url).cache_namespace()
