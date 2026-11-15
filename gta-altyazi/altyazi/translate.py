"""DeepL çevirisi, diske kalıcı önbellek ile."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def translate(self, text: str, context: str | None) -> str: ...


class DeepLBackend:
    def __init__(self, auth_key: str, source_lang: str, target_lang: str):
        import deepl  # yalnızca gerçek kullanımda gerekir

        self._translator = deepl.Translator(auth_key)
        self._source = source_lang
        self._target = target_lang

    def translate(self, text: str, context: str | None) -> str:
        result = self._translator.translate_text(
            text,
            source_lang=self._source,
            target_lang=self._target,
            # Önceki satır bağlam olarak verilir; DeepL bağlamı ücretlendirmez.
            context=context or None,
        )
        return result.text

    def usage_summary(self) -> str:
        usage = self._translator.get_usage()
        if usage.character.valid:
            return f"DeepL kullanımı: {usage.character.count:,} / {usage.character.limit:,} karakter"
        return "DeepL kullanımı: bilinmiyor"


class DryRunBackend:
    """DeepL anahtarı olmadan denemek için: metni olduğu gibi işaretleyip döndürür."""

    def translate(self, text: str, context: str | None) -> str:
        return f"[TR] {text}"

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

    @staticmethod
    def _key(text: str) -> str:
        return " ".join(text.split()).casefold()

    def translate(self, text: str, context: str | None = None) -> str:
        key = self._key(text)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        translated = self._backend.translate(text, context)
        with self._lock:
            self._cache[key] = translated
            if self._cache_path is not None:
                with self._cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"src": key, "tr": translated}, ensure_ascii=False) + "\n")
        return translated

    def __len__(self) -> int:
        return len(self._cache)
