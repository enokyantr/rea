"""Kullanım:  python -m altyazi [--dry-run] [--debug] [--ocr-test RESIM] [--list-windows] [--self-test]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as config_mod
from .translate import CachedTranslator, DeepLBackend, DryRunBackend

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


def main() -> int:
    # Konsol dışına (dosya/pipe) yazarken Windows varsayılanı cp1252 Türkçe karakterlerde çöker.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="altyazi", description="Remote Play için anlık Türkçe altyazı")
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    parser.add_argument("--dry-run", action="store_true", help="DeepL kullanmadan dene ([TR] metin)")
    parser.add_argument("--debug", action="store_true", help="OCR ve çeviri sürelerini yazdır")
    parser.add_argument("--ocr-test", type=Path, metavar="RESIM", help="Ekran görüntüsünde OCR+çeviri dene")
    parser.add_argument("--full", action="store_true", help="--ocr-test: bölgeyi değil tüm resmi oku")
    parser.add_argument("--list-windows", action="store_true", help="Açık pencere başlıklarını listele")
    parser.add_argument("--self-test", action="store_true", help="OCR, overlay ve DeepL'i bu bilgisayarda dene")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("Bu program Windows 10/11 üzerinde çalışır.")
        return 1

    from . import app, win32

    win32.enable_dpi_awareness()

    if args.list_windows:
        for _hwnd, title, rect in win32.visible_windows():
            print(f"{rect.width:>5}x{rect.height:<5} {title}")
        return 0

    cfg = config_mod.load(args.config)
    if args.self_test:
        from . import selftest

        key = cfg.resolved_auth_key()
        backend = DeepLBackend(key, cfg.source_lang, cfg.target_lang, cfg.deepl_model_type,
                               cfg.deepl_timeout_s) if key and not args.dry_run else None
        return selftest.run(cfg, backend)
    if args.dry_run:
        backend = DryRunBackend()
    else:
        key = cfg.resolved_auth_key()
        if not key:
            if not sys.stdin or not sys.stdin.isatty():
                print(f"DeepL anahtarı yok. {args.config} içinde \"deepl_auth_key\" alanına yazın.")
                return 2
            print("DeepL API anahtarı gerekli (ücretsiz: https://www.deepl.com/pro-api → Hesap → API Keys).")
            key = input("Anahtarı yapıştırıp Enter'a basın: ").strip()
            if not key:
                return 2
        backend = DeepLBackend(key, cfg.source_lang, cfg.target_lang, cfg.deepl_model_type, cfg.deepl_timeout_s)
        try:
            # Anahtarı doğrular ve DeepL bağlantısını oyundan önce ısıtır.
            app.log(backend.usage_summary())
        except Exception as exc:
            print(app.describe_error(exc))
            return 2
        if key != cfg.resolved_auth_key():
            cfg.deepl_auth_key = key
            config_mod.save(cfg, args.config)
            app.log(f"Anahtar {args.config} dosyasına kaydedildi")

    cache_path = args.config.parent / cfg.cache_file if cfg.cache_file else None
    translator = CachedTranslator(backend, cache_path)
    app.log(f"Önbellekte {len(translator)} çeviri var")

    if args.ocr_test:
        app.run_ocr_test(cfg, args.ocr_test, translator, args.full)
        return 0
    app.run_gui(cfg, args.config, translator, args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
