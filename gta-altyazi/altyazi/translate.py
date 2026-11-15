"""DeepL çevirisi, diske kalıcı önbellek ile."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def translate(self, text: str, context: str | None) -> str: ...


class DeepLBackend:
    # Bu kadar saniye istek yoksa bağlantıyı açık tutmak için ücretsiz kullanım sorgusu atılır;
    # böylece uzun sessizlikten sonraki ilk çeviri yeni TLS bağlantısı kurmayı beklemez.
    KEEPALIVE_IDLE_S = 20.0

    def __init__(self, auth_key: str, source_lang: str, target_lang: str,
                 model_type: str = "latency_optimized", timeout_s: float = 4.0):
        import deepl  # yalnızca gerçek kullanımda gerekir

        # Oyun sırasında uzun yeniden deneme beklemeleri yerine hızlı hata: bir sonraki satır zaten gelecek.
        self._translator = deepl.Translator(auth_key, max_retries=1, timeout_s=timeout_s)
        self._source = source_lang
        self._target = target_lang
        self._model_type = model_type or None
        self._last_used = time.monotonic()
        self._stop = threading.Event()
        threading.Thread(target=self._keepalive, daemon=True, name="deepl-keepalive").start()

    def translate(self, text: str, context: str | None) -> str:
        self._last_used = time.monotonic()
        result = self._translator.translate_text(
            text,
            source_lang=self._source,
            target_lang=self._target,
            # Önceki satır bağlam olarak verilir; DeepL bağlamı ücretlendirmez.
            context=context or None,
            model_type=self._model_type,
        )
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
