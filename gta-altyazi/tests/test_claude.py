"""ClaudeBackend: gerçek Anthropic SDK'sı, yerel sahte HTTP sunucusuna karşı (gönderilen istek doğrulanır)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import pytest

from altyazi.translate import CachedTranslator, ClaudeBackend, TranslationRefused


class FakeAnthropic:
    def __init__(self):
        self.requests: list[dict] = []
        self.stop_reason = "end_turn"
        self.reply = "Kapa lan çeneni, göt herif!"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: dict, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests.append({"path": self.path, "headers": dict(self.headers)})
                self._send({"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5",
                            "created_at": "2026-01-01T00:00:00Z"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                content = [] if fake.stop_reason == "refusal" else [{"type": "text", "text": fake.reply}]
                self._send({"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                            "content": content, "stop_reason": fake.stop_reason, "stop_sequence": None,
                            "usage": {"input_tokens": 10, "output_tokens": 5}})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture
def fake():
    server = FakeAnthropic()
    yield server
    server.server.shutdown()


def make(fake, uncensored=True, glossary=None):
    client = anthropic.Anthropic(api_key="test-key", base_url=fake.url, max_retries=0)
    return ClaudeBackend("test-key", "claude-opus-5-5", "low", 5.0, uncensored, glossary, _client=client)


def test_request_shape_uncensored_glossary_and_context(fake):
    backend = make(fake, glossary={"Vice City": "Vice City"})
    out = backend.translate("Shut the fuck up, asshole!", "Where's my money?")
    assert out == "Kapa lan çeneni, göt herif!"
    req = fake.requests[-1]
    body = req["body"]
    assert req["path"].startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert "thinking" not in body and "temperature" not in body
    assert "Never soften" in body["system"] and "Vice City = Vice City" in body["system"]
    user = body["messages"][0]["content"]
    assert user == "<context>Where's my money?</context>\n<line>Shut the fuck up, asshole!</line>"


def test_censored_mode_omits_profanity_rule(fake):
    backend = make(fake, uncensored=False)
    backend.translate("Hi", None)
    assert "Never soften" not in fake.requests[-1]["body"]["system"]
    assert fake.requests[-1]["body"]["messages"][0]["content"] == "<line>Hi</line>"


def test_refusal_raises_so_draft_stays(fake):
    fake.stop_reason = "refusal"
    with pytest.raises(TranslationRefused):
        make(fake).translate("Something", None)


def test_check_uses_free_models_endpoint(fake):
    assert "Claude Opus 5.5" in make(fake).check()
    assert fake.requests[-1]["path"].startswith("/v1/models/claude-opus-5-5")


def test_cache_namespace_tracks_prompt(fake, tmp_path):
    a, b = make(fake), make(fake, uncensored=False)
    assert a.cache_namespace() != b.cache_namespace()
    cache = CachedTranslator(a, tmp_path / "c.jsonl")
    cache.translate("Hello")
    cache.translate("hello")
    assert len([r for r in fake.requests if "body" in r]) == 1  # ikinci çağrı önbellekten
