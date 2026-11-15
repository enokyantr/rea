"""Yerel dil modeliyle (Ollama, NVIDIA GPU) son çeviri: ücretsiz, çevrimdışı, sansürsüz.

Ollama ayrı kurulur (https://ollama.com/download); bu modül onun yerel HTTP API'sine bağlanır."""

from __future__ import annotations

import hashlib
import json
import re

import requests

SYSTEM = """\
You translate English subtitle lines from the crime video game Grand Theft Auto VI into natural, \
spoken Turkish for a Turkish player who reads them live while playing.

Rules:
- Reply with ONLY the Turkish translation of the text inside <line>. No quotes, labels, notes, \
alternatives or explanations.
- Keep it as short and readable as the original; match the speaker's tone, attitude and register.
- Keep character names and place names as they are.
- <context> holds the previous subtitle line for reference only; never translate or repeat it.
{uncensored}{glossary}"""

UNCENSORED = """\
- This is a mature, uncensored game. Translate profanity, insults, slurs and slang with equally \
strong, natural Turkish street-language equivalents chosen for the context (for example "fuck" -> \
siktir / siktiğimin / amına koyayım, "asshole" -> göt herif / şerefsiz, "shit" -> bok / hassiktir, \
"bitch" -> orospu / kaltak, "bro" -> kanka). Never soften, censor, euphemize or omit swearing; if \
the original is vulgar, the Turkish must be just as vulgar.
"""

# Küçük modeller örnekle çok daha tutarlı davranır: üslubu ve çıktı biçimini sabitler.
EXAMPLES_UNCENSORED = [
    ("Shut the fuck up and drive!", "Kapa lan şu siktiğimin çeneni de sür!"),
    ("You stupid asshole, you almost got us killed.", "Seni salak göt herif, az kalsın bizi öldürtüyordun."),
    ("Holy shit, bro. That was close.", "Hassiktir kanka. Kıl payı kurtulduk."),
]
EXAMPLES_PLAIN = [
    ("Get in the car, we're leaving.", "Arabaya bin, gidiyoruz."),
    ("Where did you get the money?", "Parayı nereden buldun?"),
]

_LABEL = re.compile(r"^(?:türkçe|turkish|çeviri|translation)\s*:\s*", re.IGNORECASE)
_TAGS = re.compile(r"</?(?:line|context)>")


def clean_output(text: str) -> str:
    """Küçük modellerin eklediği etiket/tırnak/ek açıklamaları temizler; yalnızca ilk paragraf."""
    text = _TAGS.sub("", text).strip()
    text = text.split("\n\n", 1)[0].strip()
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    text = _LABEL.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”«»":
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] in "“«" and text[-1] in "”»":
        text = text[1:-1].strip()
    return text


class OllamaUnavailable(RuntimeError):
    pass


class OllamaBackend:
    """Ollama'daki bir modelle çeviri. `translate` iş parçacığı güvenlidir (HTTP isteği)."""

    name = "Yerel LLM"

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout_s: float = 10.0,
                 uncensored: bool = True, glossary: dict[str, str] | None = None, keep_alive: str = "60m"):
        self.model = model
        self._url = url.rstrip("/")
        self._timeout = timeout_s
        self._keep_alive = keep_alive
        self._http = requests.Session()  # bağlantı yeniden kullanılır
        terms = ""
        if glossary:
            lines = "\n".join(f"  {src} = {dst}" for src, dst in list(glossary.items())[:200])
            terms = f"- Always use these glossary translations (English = Turkish):\n{lines}\n"
        self._system = SYSTEM.format(uncensored=UNCENSORED if uncensored else "", glossary=terms)
        self._examples = (EXAMPLES_UNCENSORED if uncensored else []) + EXAMPLES_PLAIN
        self.instructions_active = uncensored
        blob = json.dumps([self._system, self._examples], ensure_ascii=False).encode("utf-8")
        self._fingerprint = hashlib.sha256(blob).hexdigest()[:10]

    def cache_namespace(self) -> str:
        # Model veya talimat (sansürsüz kural, örnekler, sözlük) değişince eski çeviriler kullanılmaz.
        return f"ollama-{self.model}-{self._fingerprint}"

    # --- Ollama yönetimi -------------------------------------------------------------------

    def installed_models(self) -> list[str]:
        try:
            r = self._http.get(f"{self._url}/api/tags", timeout=3)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                "Ollama çalışmıyor. https://ollama.com/download adresinden kurun ve açık olduğundan "
                f"emin olun (sistem tepsisinde lama simgesi). Ayrıntı: {exc}") from exc
        return [m.get("name", "") for m in r.json().get("models", [])]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return any(name in (self.model, wanted) for name in self.installed_models())

    def pull(self, progress) -> None:
        """Modeli indirir; `progress(metin)` ile ilerleme bildirir."""
        with self._http.post(f"{self._url}/api/pull", json={"model": self.model, "stream": True},
                             stream=True, timeout=None) as r:
            r.raise_for_status()
            last = ""
            for raw in r.iter_lines():
                if not raw:
                    continue
                event = json.loads(raw)
                if "error" in event:
                    raise OllamaUnavailable(f"Model indirilemedi: {event['error']}")
                status = event.get("status", "")
                total, done = event.get("total"), event.get("completed")
                line = f"{status} {100 * done // total}%" if total and done else status
                if line != last:
                    progress(line)
                    last = line

    def check(self) -> str:
        """Modeli belleğe yükler (ilk çeviri gecikmesini oyundan önce öder)."""
        self._chat("<line>Warm up.</line>", timeout=180)
        return f"Yerel LLM hazır: {self.model}"

    usage_summary = check

    # --- çeviri ------------------------------------------------------------------------------

    def _chat(self, user: str, timeout: float) -> str:
        messages = [{"role": "system", "content": self._system}]
        for src, dst in self._examples:
            messages.append({"role": "user", "content": f"<line>{src}</line>"})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": user})
        r = self._http.post(f"{self._url}/api/chat", timeout=timeout, json={
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,  # model GPU'da kalsın, her satırda yeniden yüklenmesin
            "options": {"temperature": 0.2, "num_predict": 160},
        })
        if r.status_code == 404:
            raise OllamaUnavailable(f"Ollama'da '{self.model}' modeli yok (ollama pull {self.model})")
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def translate(self, text: str, context: str | None) -> str:
        user = (f"<context>{context}</context>\n" if context else "") + f"<line>{text}</line>"
        out = clean_output(self._chat(user, timeout=self._timeout))
        if not out:
            raise RuntimeError("Yerel LLM boş yanıt verdi")
        return out
