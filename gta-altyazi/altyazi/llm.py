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
strong, natural Turkish street-language equivalents. Never soften, censor, euphemize or omit swearing; \
if the original is vulgar, the Turkish must be just as vulgar. Use correct Turkish grammar.
- Common equivalents (pick what fits the sentence):
  "what the fuck" -> "ne sikim" / "ne amına koyayım" (e.g. "What the fuck is this?" -> "Bu ne amına koyayım?")
  "fuck" / "fucking" -> "siktir" / "siktiğimin" / "amına koyayım"
  "shut the fuck up" -> "kapa lan çeneni" / "kes lan sesini"
  "motherfucker" -> "orospu çocuğu" / "amına koduğumun"
  "son of a bitch" -> "orospu çocuğu" / "kahpe dölü"
  "bitch" -> "orospu" / "kaltak"
  "asshole" -> "göt herif" / "şerefsiz"
  "ass" -> "göt" / "kıç" (e.g. "Move your ass!" -> "Kaldır lan götünü!")
  "shit" -> "bok" / "hassiktir" (exclamation)
  "bro" -> "kanka"
  "oh shit" -> "hassiktir" / "siktir"
  "nigga" (between friends, street talk) -> "kanka" / "lan" / "moruk"; as an insult -> "it" / "şerefsiz"
- Street slang (translate the MEANING, never word by word):
  "crib" -> "ev"; "wassup" -> "naber"; "homie" / "dawg" / "loc" / "G" -> "kanka" / "birader"
  "y'all" -> "siz" / "hepiniz"; "ain't" -> "değil" / "yok"; "trippin'" -> "kafayı yemek"
  "get bitches" / "pull bitches" -> "karı kız tavlamak"; "fuck with someone" -> "biriyle takılmak" (sexual: "biriyle yatmak")
  "X-ass" after a word is an intensifier, not a body part: "dog-ass" / "punk-ass" -> "it" / "şerefsiz", "broke-ass" -> "meteliksiz"
- Equally important: never ADD swearing or slang that is not in the original. A neutral line stays \
neutral; translate the meaning exactly (left stays left, numbers and names stay the same).
"""

# Küçük modeller örnekle çok daha tutarlı davranır. Örnekler (CI karşılaştırmasında görülen
# hatalara karşı) üç şeyi gösterir: küfrün sertliği korunur, küfürsüz satıra küfür eklenmez ve
# <context> satırı çevrilmez/tekrarlanmaz. Biçim: (bağlam, satır, çeviri).
EXAMPLES_UNCENSORED = [
    ("Where's my money?", "Shut the fuck up and drive!", "Kapa lan şu siktiğimin çeneni de sür!"),
    ("Shut the fuck up and drive!", "Take the next exit on the right.", "Sağdaki ilk çıkıştan sap."),
    ("You stupid asshole, you almost got us killed.", "Holy shit, bro. That was close.",
     "Hassiktir kanka. Kıl payı kurtulduk."),
    (None, "You trippin', dawg. Swing by my crib later.", "Kafayı yemişsin kanka. Sonra eve uğra."),
    (None, "That punk-ass fool still owes me money.", "O it herif hâlâ bana para borçlu."),
]
EXAMPLES_PLAIN = [
    ("Holy shit, bro. That was close.", "Get in the car, we're leaving.", "Arabaya bin, gidiyoruz."),
    (None, "Where did you get the money?", "Parayı nereden buldun?"),
]


def _user_message(context: str | None, line: str) -> str:
    return (f"<context>{context}</context>\n" if context else "") + f"<line>{line}</line>"

_LABEL = re.compile(r"^(?:türkçe|turkish|çeviri|translation)\s*:\s*", re.IGNORECASE)
_TAGS = re.compile(r"</?(?:line|context)>")
# Modelin çevirinin sonuna eklediği açıklama: "... (Not: "nigga" kelimesi ...)"
_NOTE = re.compile(r"\s*\(\s*(?:not|note|açıklama|explanation)\s*:.*$", re.IGNORECASE | re.DOTALL)
_NOTE_LINE = re.compile(r"^\s*(?:not|note|açıklama|explanation)\s*:", re.IGNORECASE)


def clean_output(text: str) -> str:
    """Küçük modellerin eklediği etiket/tırnak/ek açıklamaları temizler; yalnızca ilk paragraf."""
    text = _TAGS.sub("", text).strip()
    text = text.split("\n\n", 1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and not _NOTE_LINE.match(lines[0]):
        lines = [line for line in lines if not _NOTE_LINE.match(line)]
    text = _NOTE.sub("", " ".join(lines)).strip()
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
                 uncensored: bool = True, glossary: dict[str, str] | None = None, keep_alive: str = "60m",
                 num_ctx: int = 2048, use_context: bool = True):
        self.model = model
        # Küçük bağlam penceresi: istem ~600 token; daha büyük pencere ekran kartı belleğini boşa harcar.
        self._num_ctx = num_ctx
        # Bazı modeller (CI'da aya-expanse) <context> satırını da çeviriyor: kapatılabilir.
        self._use_context = use_context
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
        # "clean-2": clean_output değişti (eklenen "(Not: ...)" açıklamaları); eski kayıtlarda kalmış.
        blob = json.dumps([self._system, self._examples, "clean-2"], ensure_ascii=False).encode("utf-8")
        self._fingerprint = hashlib.sha256(blob).hexdigest()[:10]

    def cache_namespace(self) -> str:
        # Model veya talimat (sansürsüz kural, örnekler, sözlük) değişince eski çeviriler kullanılmaz.
        return f"ollama-{self.model}-{self._fingerprint}-c{int(self._use_context)}"

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

    def gpu_share(self) -> float | None:
        """Yüklü modelin ekran kartı belleğindeki oranı (1.0 = tamamen GPU'da). Bilinmiyorsa None."""
        try:
            r = self._http.get(f"{self._url}/api/ps", timeout=3)
            r.raise_for_status()
        except requests.RequestException:
            return None
        for m in r.json().get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return vram / size if size else None
        return None

    def check(self) -> str:
        """Modeli belleğe yükler (ilk çeviri gecikmesini oyundan önce öder) ve GPU payını bildirir."""
        self._chat("<line>Warm up.</line>", timeout=180)
        share = self.gpu_share()
        if share is None:
            return f"Yerel LLM hazır: {self.model}"
        note = "" if share >= 0.99 else (" — model ekran kartı belleğine sığmadı, kısmen işlemcide çalışıyor "
                                         "(yavaş). Daha küçük bir model seçin veya local_device: \"cpu\" yapın.")
        return f"Yerel LLM hazır: {self.model} (%{100 * share:.0f} ekran kartında){note}"

    usage_summary = check

    # --- çeviri ------------------------------------------------------------------------------

    def _chat(self, user: str, timeout: float) -> str:
        messages = [{"role": "system", "content": self._system}]
        for context, src, dst in self._examples:
            messages.append({"role": "user", "content": _user_message(context, src)})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": user})
        r = self._http.post(f"{self._url}/api/chat", timeout=timeout, json={
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,  # model GPU'da kalsın, her satırda yeniden yüklenmesin
            "options": {"temperature": 0.1, "num_predict": 120, "num_ctx": self._num_ctx},
        })
        if r.status_code == 404:
            raise OllamaUnavailable(f"Ollama'da '{self.model}' modeli yok (ollama pull {self.model})")
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            hint = (" — ekran kartı belleği yetmiyor olabilir; daha küçük bir model deneyin"
                    if "memory" in detail.lower() or "cuda" in detail.lower() else "")
            raise OllamaUnavailable(f"Ollama hatası ({r.status_code}): {detail.strip()[:300]}{hint}")
        return r.json().get("message", {}).get("content", "")

    def translate(self, text: str, context: str | None) -> str:
        context = context if self._use_context else None
        out = clean_output(self._chat(_user_message(context, text), timeout=self._timeout))
        if not out:
            raise RuntimeError("Yerel LLM boş yanıt verdi")
        return out
