"""Yerel dil modeliyle (Ollama, NVIDIA GPU) son çeviri: ücretsiz, çevrimdışı, sansürsüz.

Ollama ayrı kurulur (https://ollama.com/download); bu modül onun yerel HTTP API'sine bağlanır."""

from __future__ import annotations

import hashlib
import json
import re
import threading

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
- Earlier turns are the previous lines of the same conversation with your translations. Keep names, \
nicknames, forms of address (sen/siz), slang and the meaning consistent with them; resolve pronouns and \
references from them. Translate only the new line.
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
    (None, "Nigga...", "Lan..."),
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


# Küçük model bazen çeviri yerine açıklama/ret yazıyor (gerçek loglar):
# "(Not translated as per guidelines. In Turkish street talk, "nigga" would be ...)",
# "Bu cümle anlaşılmaz ve hatalı görünüyor... Türkçe'ye çeviremem."
_META = re.compile(
    r"\b(?:not translated|translat(?:e|ed|ion|ing)\b|guidelines?|original (?:text|line)|as an ai|"
    r"i can(?:no|')t|çeviremem|çevrilemez|çevirmek mümkün değil|orijinal metin|anlaşılmaz|"
    r"bu mesaj(?:ın|ı)?\b|daha (?:açık|net) bir (?:talimat|soru|ifade)|nasıl yardımcı olabilirim)",
    re.IGNORECASE,
)


def is_continuation(previous: str | None, text: str) -> bool:
    """Satır önceki altyazının devamı mı? Önceki cümle bitmemiş ya da bu satır küçük harfle başlıyor."""
    if not previous or not text:
        return False
    return previous.rstrip()[-1:] not in ".!?…\"'”" or text.lstrip()[:1].islower()


def is_meta_reply(text: str) -> bool:
    """Çeviri değil, modelin kendi açıklaması mı? (Tamamı parantez içinde ya da açıklama ifadesi.)"""
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return True
    return bool(_META.search(stripped))


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


class Cancelled(RuntimeError):
    """Önceden çeviri isteği artık gerekmiyor (yanlış okuma/yeni satır): üretim yarıda kesildi."""


def _find_ollama() -> str | None:
    import os
    import shutil

    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


class OllamaBackend:
    """Ollama'daki bir modelle çeviri. `translate` iş parçacığı güvenlidir (HTTP isteği)."""

    name = "Yerel LLM"
    # translate(..., cancel=Event) destekler: önceden çeviri boşa çıkınca ekran kartı hemen boşalır.
    cancellable = True

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout_s: float = 10.0,
                 uncensored: bool = True, glossary: dict[str, str] | None = None, keep_alive: str = "60m",
                 num_ctx: int = 2048, use_context: bool = True, context_for_continuation: bool = True):
        self.model = model
        # Küçük bağlam penceresi: istem ~600 token; daha büyük pencere ekran kartı belleğini boşa harcar.
        self._num_ctx = num_ctx
        # Bazı modeller (CI'da aya-expanse) <context> satırını da çeviriyor: kapatılabilir.
        self._use_context = use_context
        self._context_for_continuation = context_for_continuation
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

    def ensure_running(self, log=print, wait_s: float = 20.0) -> bool:
        """Ollama sunucusu kapalıysa (tepsi uygulaması açılmamış) arka planda başlatır.
        Başlatabildiyse ya da zaten çalışıyorsa True."""
        if self._alive():
            return True
        exe = _find_ollama()
        if exe is None:
            return False
        import subprocess
        import time

        log("Ollama kapalı, arka planda başlatılıyor...")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        try:
            subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        except OSError:
            return False
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if self._alive():
                return True
            time.sleep(0.3)
        return False

    def _alive(self) -> bool:
        try:
            return self._http.get(f"{self._url}/api/version", timeout=1).ok
        except requests.RequestException:
            return False

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

    def _chat(self, user: str, timeout: float, num_predict: int = 120, history=(),
              cancel: threading.Event | None = None) -> str:
        # İstem sırası sabit önek + değişen kuyruk: sistem talimatı ve örnekler her istekte birebir aynı
        # (Ollama bu öneki önbellekten kullanır), sohbet geçmişi ve yeni satır en sonda.
        messages = [{"role": "system", "content": self._system}]
        for context, src, dst in self._examples:
            messages.append({"role": "user", "content": _user_message(context, src)})
            messages.append({"role": "assistant", "content": dst})
        # Sohbet geçmişi: önceki satırlar ve verdiğimiz çeviriler gerçek turlar olarak (model bağlamı
        # çevirmeye/tekrarlamaya çalışmaz, yalnızca tutarlı devam eder).
        for src, dst in history:
            messages.append({"role": "user", "content": _user_message(None, src)})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": user})
        body = {
            "model": self.model,
            "messages": messages,
            # İptal edilebilir istek akış olarak alınır: bağlantı kapatılınca Ollama üretimi durdurur.
            "stream": cancel is not None,
            "keep_alive": self._keep_alive,  # model GPU'da kalsın, her satırda yeniden yüklenmesin
            "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": self._num_ctx},
        }
        with self._http.post(f"{self._url}/api/chat", timeout=timeout, json=body,
                             stream=cancel is not None) as r:
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
            if cancel is None:
                return r.json().get("message", {}).get("content", "")
            parts: list[str] = []
            for raw in r.iter_lines():
                if cancel.is_set():
                    raise Cancelled("önceden çeviri iptal edildi")  # `with` bağlantıyı kapatır
                if not raw:
                    continue
                event = json.loads(raw)
                if "error" in event:
                    raise OllamaUnavailable(f"Ollama hatası: {str(event['error']).strip()[:300]}")
                parts.append(event.get("message", {}).get("content", ""))
                if event.get("done"):
                    break
            return "".join(parts)

    def translate(self, text: str, context: str | None, cancel: threading.Event | None = None) -> str:
        # Bağlam: her zaman (use_context) ya da yalnızca satır öncekinin devamıysa ("or lawyer she
        # fucking with." gibi yarım cümleler bağlamsız yanlış çevriliyordu).
        history = getattr(context, "history", ())
        if history:
            context = None  # önceki satır zaten sohbet geçmişinde
        elif not (self._use_context or (self._context_for_continuation and is_continuation(context, text))):
            context = None
        # Çıktı sınırı: Türkçe satır İngilizceden çok uzun olmaz; model açıklamaya başlarsa erken kesilir.
        limit = min(120, 24 + 4 * len(text.split()))
        out = clean_output(self._chat(_user_message(context, text), timeout=self._timeout, num_predict=limit,
                                      history=history, cancel=cancel))
        if not out:
            raise RuntimeError("Yerel LLM boş yanıt verdi")
        if is_meta_reply(out):  # hata sayılır: taslak çeviri ekranda kalır, önbelleğe girmez
            raise RuntimeError(f"Yerel LLM çeviri yerine açıklama yazdı: {out[:80]}")
        return out
