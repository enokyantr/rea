"""CI: aday Ollama modellerinin örnek repliklerdeki Türkçe çevirisini karşılaştırır.

Kullanım: python tools/llm_eval.py MODEL [MODEL...]   (Ollama çalışıyor ve modeller indirilmiş olmalı)
Ekran kartı olmayan sunucuda süreler gerçek kullanımı yansıtmaz; amaç kalite karşılaştırması."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from altyazi import bench  # noqa: E402
from altyazi.llm import OllamaBackend  # noqa: E402

for model in sys.argv[1:]:
    backend = OllamaBackend(model, timeout_s=600)
    backend.check()
    bench.run(backend, f"=== {model} ===")
