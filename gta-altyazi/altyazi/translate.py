"""DeepL çevirisi, diske kalıcı önbellek ile."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def translate(self, text: str, context: str | None) -> str: ...


GLOSSARY_PREFIX = "gta-altyazi-"


class DeepLBackend:
    name = "DeepL"

    # Bu kadar saniye istek yoksa bağlantıyı açık tutmak için ücretsiz kullanım sorgusu atılır;
    # böylece uzun sessizlikten sonraki ilk çeviri yeni TLS bağlantısı kurmayı beklemez.
    KEEPALIVE_IDLE_S = 20.0

    def __init__(self, auth_key: str, source_lang: str, target_lang: str,
                 model_type: str = "latency_optimized", timeout_s: float = 4.0,
                 instructions: list[str] | None = None, log=print, _translator=None):
        import deepl  # yalnızca gerçek kullanımda gerekir

        self._deepl = deepl
        # Oyun sırasında uzun yeniden deneme beklemeleri yerine hızlı hata: bir sonraki satır zaten gelecek.
        self._translator = _translator or deepl.Translator(auth_key, max_retries=1, timeout_s=timeout_s)
        self._source = source_lang
        self._target = target_lang
        self._log = log
        self._glossary = None
        self._glossary_fp = "0"
        model_type = model_type or None
        # Talimatlar (sansürsüz çeviri) DeepL'in her dil/model için desteklediği bir özellik
        # olmayabilir: reddedilirse sırayla bir sonraki ayara geçilir.
        self._attempts: list[tuple[str | None, list[str] | None]] = []
        if instructions:
            self._attempts.append((model_type, instructions))
            if model_type != "quality_optimized":
                self._attempts.append(("quality_optimized", instructions))
        self._attempts.append((model_type, None))
        self._attempt = 0
        self._attempt_lock = threading.Lock()
        self._last_used = time.monotonic()
        self._stop = threading.Event()
        if _translator is None:
            threading.Thread(target=self._keepalive, daemon=True, name="deepl-keepalive").start()

    @property
    def instructions_active(self) -> bool:
        return self._attempts[self._attempt][1] is not None

    def cache_namespace(self) -> str:
        """Çeviriyi gerçekten etkileyen o anki durum: kullanılan sözlük ve sansürsüz talimatın
        gerçekten gönderilip gönderilmediği. Geri çekilme olursa önbellek yanlış etiketlenmez."""
        return f"g{self._glossary_fp if self._glossary is not None else 0}-u{int(self.instructions_active)}"

    def setup_glossary(self, entries: dict[str, str], fingerprint: str) -> str:
        """Sözlüğü DeepL'de oluşturur (aynı içerik varsa yeniden kullanır). Durum mesajı döndürür."""
        if not entries:
            return "Sözlük boş"
        # Dil çifti de adda: dil değişirse eski (uyumsuz) sözlük yeniden kullanılmaz.
        name = f"{GLOSSARY_PREFIX}{fingerprint}-{self._source}-{self._target}".lower()
        try:
            found = None
            for info in self._translator.list_glossaries():
                if info.name == name and info.ready:
                    found = info
                elif info.name.startswith(GLOSSARY_PREFIX):
                    self._translator.delete_glossary(info)  # eski sürümler; DeepL hesabında yer kaplamasın
            if found is None:
                found = self._translator.create_glossary(name, self._source, self._target, entries)
            self._glossary = found
            self._glossary_fp = fingerprint
            return f"DeepL sözlüğü hazır: {len(entries)} terim"
        except Exception as exc:
            self._glossary = None
            return f"DeepL sözlüğü kullanılamıyor ({exc}); sözlüksüz devam ediliyor"

    def _rejected(self, exc: Exception) -> bool:
        """İstek parametreleri reddedildi mi? Yalnızca HTTP 400 (Bad request). 5xx/ağ/kota/anahtar
        hataları geçicidir; bunlar yüzünden sansürsüz talimat kapatılmamalı."""
        return isinstance(exc, self._deepl.DeepLException) and getattr(exc, "http_status_code", None) == 400

    def translate(self, text: str, context: str | None) -> str:
        self._last_used = time.monotonic()
        while True:
            attempt = self._attempt
            model_type, instructions = self._attempts[attempt]
            try:
                result = self._translator.translate_text(
                    text,
                    source_lang=self._source,
                    target_lang=self._target,
                    # Önceki satır bağlam olarak verilir; DeepL bağlamı ücretlendirmez.
                    context=context or None,
                    model_type=model_type,
                    glossary=self._glossary,
                    custom_instructions=instructions,
                )
                break
            except Exception as exc:
                if not self._rejected(exc) or attempt + 1 >= len(self._attempts):
                    raise
                with self._attempt_lock:
                    if self._attempt == attempt:
                        self._attempt = attempt + 1
                        nxt = self._attempts[self._attempt]
                        self._log(f"DeepL bu ayarı reddetti ({exc}); şununla devam: model={nxt[0] or 'varsayılan'}, "
                                  f"sansürsüz talimat={'açık' if nxt[1] else 'KAPALI'}")
        self._last_used = time.monotonic()
        return result.text

    def _keepalive(self) -> None:
        while not self._stop.wait(5.0):
            if time.monotonic() - self._last_used >= self.KEEPALIVE_IDLE_S:
                self._last_used = time.monotonic()
                try:
                    self._translator.get_usage()
                except Exception:
                    pass  # ağ hatası çeviri sırasında ayrıca raporlanır

    def close(self) -> None:
        self._stop.set()

    def usage_summary(self) -> str:
        usage = self._translator.get_usage()
        if usage.character.valid:
            return f"DeepL kullanımı: {usage.character.count:,} / {usage.character.limit:,} karakter"
        return "DeepL kullanımı: bilinmiyor"


CLAUDE_SYSTEM = """\
You translate English subtitle lines from the crime video game Grand Theft Auto VI into natural, \
spoken Turkish for a Turkish player who reads them live while playing.

Rules:
- Reply with ONLY the Turkish translation of the text inside <line>. No quotes, labels, notes or \
explanations.
- Keep it as short and readable as the original; match the speaker's tone, attitude and register.
- Keep character names and place names as they are.
- <context> holds the previous subtitle line for reference only; never translate or repeat it.
{uncensored}{glossary}"""

CLAUDE_UNCENSORED = """\
- This is a mature, uncensored game. Translate profanity, insults, slurs and slang with equally \
strong, natural Turkish street-language equivalents chosen for the context (for example "fuck" -> \
siktir / siktiğimin / amına koyayım, "asshole" -> göt herif / şerefsiz, "shit" -> bok / hassiktir, \
"bitch" -> orospu / kaltak). Never soften, censor, euphemize, bleep or omit swearing; if the original \
is vulgar, the Turkish must be just as vulgar.
"""


class TranslationRefused(RuntimeError):
    pass


class ClaudeBackend:
    """Son çeviri Claude ile: talimatları (sansürsüz, sözlük, ton) birebir uygular."""

    name = "Claude"

    def __init__(self, api_key: str, model: str, effort: str, timeout_s: float,
                 uncensored: bool, glossary: dict[str, str] | None = None, _client=None):
        import anthropic

        self._anthropic = anthropic
        # Oyun sırasında uzun yeniden deneme beklemeleri yerine hızlı hata: bir sonraki satır zaten gelecek.
        self._client = _client or anthropic.Anthropic(api_key=api_key or None, timeout=timeout_s, max_retries=1)
        self._model = model
        self._effort = effort
        self._uncensored = uncensored
        terms = ""
        if glossary:
            lines = "\n".join(f"  {src} = {dst}" for src, dst in list(glossary.items())[:300])
            terms = f"- Always use these glossary translations (English = Turkish):\n{lines}\n"
        self._system = CLAUDE_SYSTEM.format(uncensored=CLAUDE_UNCENSORED if uncensored else "", glossary=terms)
        self._fingerprint = hashlib.sha256(self._system.encode("utf-8")).hexdigest()[:10]

    instructions_active = property(lambda self: self._uncensored)

    def cache_namespace(self) -> str:
        # Model ve talimat metni (sansürsüz kuralı + sözlük) değişince eski çeviriler kullanılmaz.
        return f"claude-{self._model}-{self._fingerprint}"

    def check(self) -> str:
        """Anahtarı ve modelin erişilebilir olduğunu ücretsiz bir istekle doğrular."""
        info = self._client.models.retrieve(self._model)
        return f"Claude hazır: {getattr(info, 'display_name', self._model)} (effort: {self._effort})"

    def translate(self, text: str, context: str | None) -> str:
        user = (f"<context>{context}</context>\n" if context else "") + f"<line>{text}</line>"
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=2048,
            system=self._system,
            # Düşük effort: kısa çeviri işi, en düşük gecikme.
            output_config={"effort": self._effort},
            # Bir model satırı reddederse sunucu aynı istekte uygun bir modele geçer.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise TranslationRefused(f"Claude bu satırı çevirmedi ({getattr(details, 'category', None)})")
        out = "".join(block.text for block in response.content if block.type == "text").strip()
        if not out:
            raise TranslationRefused(f"Claude boş yanıt verdi ({response.stop_reason})")
        return out

    def usage_summary(self) -> str:
        return self.check()


class DryRunBackend:
    """DeepL anahtarı olmadan denemek için: metni olduğu gibi işaretleyip döndürür."""

    def translate(self, text: str, context: str | None) -> str:
        return f"[TR] {text}"

    def cache_namespace(self) -> str:
        return "dry-run"  # deneme çıktıları gerçek çevirilere karışmasın

    def usage_summary(self) -> str:
        return "Deneme modu: DeepL kullanılmıyor"


class CachedTranslator:
    def __init__(self, backend: Backend, cache_path: Path | None):
        self._backend = backend
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if cache_path is not None and cache_path.exists():
            with cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entry = json.loads(line)
                        self._cache[entry["src"]] = entry["tr"]
                    except (ValueError, KeyError, TypeError):
                        continue  # bozuk satırı atla, kalan önbellek kullanılabilir

    def _key(self, text: str, namespace: str | None = None) -> str:
        """Anahtar, arka ucun o anki ayarlarını (sözlük, sansürsüz talimat) içerir; ayar değişince
        eski çeviriler kullanılmaz."""
        if namespace is None:
            namespace = getattr(self._backend, "cache_namespace", lambda: "")()
        key = " ".join(text.split()).casefold()
        return f"{namespace}\x1f{key}" if namespace else key

    def cached(self, text: str) -> str | None:
        with self._lock:
            return self._cache.get(self._key(text))

    def translate(self, text: str, context: str | None = None) -> str:
        key = self._key(text)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        translated = self._backend.translate(text, context)
        # Çeviri sırasında ayar geri çekilmiş olabilir (ör. talimat reddedildi): çevirinin
        # gerçekte hangi ayarla yapıldığını kaydet.
        key = self._key(text)
        with self._lock:
            self._cache[key] = translated
            if self._cache_path is not None:
                with self._cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"src": key, "tr": translated}, ensure_ascii=False) + "\n")
        return translated

    def __len__(self) -> int:
        return len(self._cache)
