"""Kullanım:  python -m altyazi [--dry-run] [--debug] [--ocr-test RESIM] [--list-windows] [--self-test]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config as config_mod
from .textproc import needs_llm
from .hybrid import HybridTranslator
from .translate import CachedTranslator, DeepLBackend, DryRunBackend

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


# Menüden "Kaydet ve yeniden başlat": baslat.bat bu çıkış koduyla programı yeniden açar.
RESTART_CODE = 3


def main() -> int:
    # Konsol dışına (dosya/pipe) yazarken Windows varsayılanı cp1252 Türkçe karakterlerde çöker.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from .logbuf import install

    install()  # menüdeki "Kayıt" bölümü konsol çıktısını gösterebilsin
    parser = argparse.ArgumentParser(prog="altyazi", description="Remote Play için anlık Türkçe altyazı")
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    parser.add_argument("--dry-run", action="store_true", help="DeepL kullanmadan dene ([TR] metin)")
    parser.add_argument("--debug", action="store_true", help="OCR ve çeviri sürelerini yazdır")
    parser.add_argument("--guvenli", action="store_true",
                        help="Güvenli mod: yalnızca altyazı (dublaj, sokak, önceden çeviri kapalı)")
    parser.add_argument("--ocr-test", type=Path, metavar="RESIM", help="Ekran görüntüsünde OCR+çeviri dene")
    parser.add_argument("--ocr-compare", type=Path, nargs="?", const=APP_DIR / "dump", metavar="KLASOR",
                        help="dump/ kırpıntılarında tüm ön işleme seçeneklerini karşılaştır")
    parser.add_argument("--full", action="store_true", help="--ocr-test: bölgeyi değil tüm resmi oku")
    parser.add_argument("--list-windows", action="store_true", help="Açık pencere başlıklarını listele")
    parser.add_argument("--self-test", action="store_true", help="OCR, overlay ve DeepL'i bu bilgisayarda dene")
    parser.add_argument("--mt-bench", action="store_true", help="Yerel taslak çevirinin örnek repliklerdeki kalitesi/hızı")
    parser.add_argument("--llm-bench", action="store_true", help="Son çevirmenin (yerel LLM/DeepL) örnek repliklerdeki kalitesi/hızı")
    parser.add_argument("--ollama-model", metavar="MODEL", help="config.json'daki ollama_model yerine bunu kullan")
    parser.add_argument("--no-context", action="store_true", help="Yerel LLM'e önceki satırı gönderme")
    parser.add_argument("--dump", action="store_true", help="OCR'a giren görüntüleri dump/ klasörüne kaydet (ayar için)")
    parser.add_argument("--cpu-draft", action="store_true", help="Yerel taslak çeviriyi işlemcide çalıştır (GPU belleği boşalır)")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("Bu program Windows 10/11 üzerinde çalışır.")
        return 1

    from . import app, win32

    win32.enable_dpi_awareness()
    win32.raise_priority()

    if args.list_windows:
        for _hwnd, title, rect in win32.visible_windows():
            print(f"{rect.width:>5}x{rect.height:<5} {title}")
        return 0

    instance = None
    gui = not (args.self_test or args.mt_bench or args.llm_bench or args.ocr_test or args.ocr_compare)
    if gui:
        try:
            instance = win32.claim_single_instance()
        except Exception as exc:  # tek kopya denetimi çalışmazsa program yine açılsın
            print(f"Tek kopya denetimi yapılamadı ({exc}); devam ediliyor.")
            instance = ()
        if instance is None:
            print("GTA Altyazı zaten açık: açık olanın menüsü öne getirildi. (İki kopya ekran kartını ve "
                  "kısayolları paylaşır, gecikme artar.) Menü görünmezse Görev Yöneticisi'nde \"python\" "
                  "işlemini sonlandırıp yeniden başlatın.")
            return 0
    cfg = config_mod.load(args.config)
    if args.guvenli:
        # Güvenli mod (guvenli_baslat.bat): yalnızca altyazı; dublaj, sokak dinleme ve önceden çeviri
        # kapalı. Sorun çıkan bir özellik oyunu bozmasın; menüden tek tek yeniden açılabilir.
        cfg.tts_enabled = False
        cfg.street_enabled = False
        app.SAFE_MODE = True  # ayara yazılmaz: normal başlatmada önceden çeviri yine açık
        print("Güvenli mod: yalnızca altyazı (dublaj, sokak konuşmaları ve önceden çeviri kapalı).")
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.no_context:
        cfg.ollama_use_context = False
    if args.cpu_draft:
        cfg.local_device = "cpu"
    mode = cfg.translation_mode
    if mode not in ("hybrid", "final", "deepl", "local"):
        print(f"translation_mode geçersiz: {mode!r} (hybrid | final | local)")
        return 2
    if mode == "deepl":  # eski ad: yalnızca son çevirmen
        mode = "final"
    if cfg.final_translator not in ("ollama", "deepl"):
        print(f"final_translator geçersiz: {cfg.final_translator!r} (ollama | deepl)")
        return 2

    # Oyun aynı bilgisayardaysa (Performans: oyun) dil modeli ve OCR ekran kartını oyunla paylaşmasın.
    from .perf import game_policy, vram_free_mb

    free_mb = None
    if cfg.final_translator == "ollama" and mode in ("hybrid", "final"):
        free_mb = vram_free_mb()
        if free_mb is not None:  # önceki oturumdan yüklü kalan kendi modelimiz "dolu" sayılmasın
            free_mb += sum(size for _name, size in _ollama_resident(cfg)) / 2**20
    policy = game_policy(cfg, free_mb)
    if policy.note:
        app.log(policy.note)

    local = None
    if mode in ("hybrid", "local") or args.self_test or args.mt_bench:
        model_dir = Path(cfg.local_model_dir)
        if not model_dir.is_absolute():
            model_dir = args.config.parent / model_dir
        try:
            from .local_mt import LocalBackend

            device = cfg.local_device
            if device == "auto" and ((cfg.final_translator == "ollama" and mode != "local")
                                     or cfg.performance_mode == "oyun"):
                # RTX 4060 Ti 8 GB ölçümü: taslak model GPU'dayken Ollama satırları 1,9-6,6 sn sürdü
                # (GPU paylaşımı); taslak işlemcideyken 0,28-0,43 sn. Taslak işlemcide ~0,1 sn.
                # Oyun modu: ekran kartı belleği oyuna kalsın.
                device = "cpu"
            # Yayın/oyun modu: taslak çeviri 2 çekirdekle (yayın kodlaması ve oyun işlemciye kalsın).
            local = LocalBackend(model_dir, device, cfg.local_beam_size,
                                 threads=2 if cfg.performance_mode in ("yayin", "oyun") else 4)
            app.log(local.describe())
        except Exception as exc:
            if args.mt_bench or (mode == "local" and not args.self_test):
                print(f"Yerel çeviri başlatılamadı: {exc}")
                return 2
            if mode == "hybrid" or args.self_test:
                app.log(f"Yerel çeviri kullanılamıyor ({exc}); yalnızca DeepL kullanılacak")

    if args.mt_bench:
        from . import bench

        return bench.run(local, local.describe())
    if args.llm_bench:
        from . import bench

        backend = _final_backend(cfg, args, *_glossary(cfg, args), interactive=True, num_gpu=policy.num_gpu)
        if backend is None:
            return 2
        if hasattr(backend, "_timeout"):
            backend._timeout = 120  # ölçümde gerçek süreyi görmek için oyundaki zaman aşımı uygulanmaz
        title = f"{backend.name}: {getattr(backend, 'model', cfg.final_translator)}"
        route = None
        if local is not None and cfg.final_translator == "ollama" and cfg.ollama_only_profanity:
            # Oyundaki gibi: küfürlü satır LLM'e, diğerleri yerel taslak çeviriye.
            route = lambda line: (backend, "[LLM]    ") if needs_llm(line) else (local, "[taslak] ")  # noqa: E731
            title += "  (yalnızca küfürlü satırlar LLM'e gider)"
        return bench.run(backend, title, route=route)
    if args.self_test:
        from . import selftest

        backend = None
        if not args.dry_run:
            try:
                backend = _final_backend(cfg, args, *_glossary(cfg, args), interactive=False)
            except Exception as exc:
                app.log(f"Son çevirmen kullanılamıyor: {app.describe_error(exc)}")
        return selftest.run(cfg, backend, local)

    final = None
    if mode in ("hybrid", "final"):
        try:
            entries, fp = _glossary(cfg, args)
        except Exception as exc:  # hatalı sözlük satırı: kullanıcı düzeltsin
            print(exc)
            return 2
        llm_off = (policy.llm == "kapali" and cfg.final_translator == "ollama" and not args.dry_run)
        if llm_off and (mode != "hybrid" or local is None):
            app.log("Oyun modu: taslak çeviri yok, yerel dil modeli yine de yükleniyor (tek çevirmen o).")
            llm_off = False
        if llm_off:
            backend = None
            for name, _size in _ollama_resident(cfg):  # önceki oturumdan kalan model belleği bıraksın
                if _ollama_unload(cfg, name):
                    app.log(f"Ollama: '{name}' ekran kartı belleğinden boşaltıldı (oyun modu).")
        elif args.dry_run:
            backend = DryRunBackend()
        else:
            backend = _final_backend(cfg, args, entries, fp, interactive=True, num_gpu=policy.num_gpu)
            if backend is None and not (mode == "hybrid" and local is not None):
                return 2
            if backend is None:
                app.log("Son çevirmen yok: yalnızca yerel taslak çeviri kullanılacak")
            else:
                app.log("Sansürsüz çeviri: " + ("açık (küfür/argo yumuşatılmaz)" if cfg.uncensored else "kapalı"))
        if backend is not None:
            cache_path = args.config.parent / cfg.cache_file if cfg.cache_file else None
            final = CachedTranslator(backend, cache_path)
            app.log(f"Önbellekte {len(final)} çeviri var")

    from . import glossary as glossary_mod

    try:
        has_term = glossary_mod.needs_final(_glossary(cfg, args)[0])
    except Exception:
        has_term = lambda text: False  # noqa: E731

    def route_to_final(text: str) -> bool:
        # Sohbeti takip: her satır bağlamla yerel LLM'e (taslak önce görünür). Değilse yalnızca
        # küfür/argo ya da sözlükte özel karşılığı olan terim.
        return cfg.context_translation or needs_llm(text) or has_term(text)

    def make_translator() -> HybridTranslator:
        # Her bölgenin kendi kuyruğu; çeviri motorları (yerel model, DeepL, önbellek) ortak.
        only_profane = cfg.final_translator == "ollama" and cfg.ollama_only_profanity
        return HybridTranslator(final, local, final_for=route_to_final if only_profane else None)

    app.log(f"Çeviri modu: {make_translator().mode}")
    if args.ocr_compare:
        app.run_ocr_compare(cfg, args.ocr_compare)
        return
    if args.ocr_test:
        app.run_ocr_test(cfg, args.ocr_test, make_translator(), args.full)
        return 0
    info = [("Çeviri modu", make_translator().mode)]
    if local is not None:
        info.append(("Taslak çeviri", local.describe()))
    if final is not None:
        model = getattr(getattr(final, "_backend", None), "model", "")
        info.append(("Son çevirmen", f"{cfg.final_translator} {model}".strip()))
    if cfg.performance_mode == "oyun":
        ocr_at = "işlemcide" if policy.ocr_device == "cpu" else "ekran kartında"
        llm = ("kapalı (taslak son çeviri)" if final is None
               else "işlemcide" if policy.num_gpu == 0 else "ekran kartında")
        info.append(("Oyun modu", (f"dil modeli {llm} · " if cfg.final_translator == "ollama" else "")
                     + f"OCR {ocr_at}"))
    restart = app.run_gui(cfg, args.config, make_translator, args.debug,
                          dump_dir=(args.config.parent / "dump") if args.dump else None, engine_info=info,
                          instance=instance)
    return RESTART_CODE if restart else 0


def _glossary(cfg, args) -> tuple[dict[str, str], str]:
    from . import glossary

    if not cfg.glossary_file:
        return {}, "0"
    path = Path(cfg.glossary_file)
    if not path.is_absolute():
        path = args.config.parent / path
    entries = glossary.load(path)
    return entries, glossary.fingerprint(entries)


def _make_deepl(cfg, key: str) -> DeepLBackend:
    from . import app

    return DeepLBackend(key, cfg.source_lang, cfg.target_lang, cfg.deepl_model_type, cfg.deepl_timeout_s,
                        instructions=cfg.deepl_instructions if cfg.uncensored else None, log=app.log)


def _final_backend(cfg, args, entries: dict[str, str], fp: str, interactive: bool, num_gpu: int | None = None):
    """Taslağın yerine geçen son çevirmeni kurar. Başarısızsa None (etkileşimsiz modda hata fırlatır).
    `num_gpu`: Ollama'da ekran kartına konan katman sayısı (0: tamamen işlemcide; None: Ollama seçer)."""
    from . import app

    if cfg.final_translator == "deepl":
        if not interactive and not cfg.resolved_auth_key():
            raise RuntimeError("DeepL anahtarı yok")
        backend = _deepl_backend(cfg, args) if interactive else _make_deepl(cfg, cfg.resolved_auth_key())
        if backend is not None:
            app.log(backend.setup_glossary(entries, fp))
        return backend

    from .llm import OllamaBackend, OllamaUnavailable

    # num_gpu yalnızca oyun modunda (game_llm "cpu") verilir; None: Ollama'nın kendi seçimi (bugünkü davranış).
    backend = OllamaBackend(cfg.ollama_model, cfg.ollama_url, cfg.ollama_timeout_s, cfg.uncensored, entries,
                            num_ctx=cfg.ollama_num_ctx, use_context=cfg.ollama_use_context,
                            context_for_continuation=cfg.ollama_context_continuation, num_gpu=num_gpu)
    try:
        backend.ensure_running(app.log)
        if not backend.has_model():
            if not interactive:
                raise OllamaUnavailable(f"Ollama'da '{cfg.ollama_model}' modeli yok (ollama pull {cfg.ollama_model})")
            answer = input(f"'{cfg.ollama_model}' modeli yüklü değil (birkaç GB, bir kerelik). "
                           "Şimdi indirilsin mi? [E/h]: ").strip().lower()
            if answer not in ("", "e", "evet", "y", "yes"):
                return None
            backend.pull(lambda msg: print("\r" + msg.ljust(70), end="", flush=True))
            print()
        app.log("Yerel LLM " + ("işlemciye" if num_gpu == 0 else "ekran kartına")
                + " yükleniyor (ilk seferde 10-60 sn sürebilir)...")
        app.log(backend.check())
    except Exception as exc:
        if not interactive:
            raise
        print(app.describe_error(exc))
        return None
    return backend


def _ollama_resident(cfg) -> list[tuple[str, int]]:
    """Ollama'da şu an yüklü olan bizim modelimiz: [(ad, ekran kartındaki bayt)]. Ollama kapalıysa boş."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(cfg.ollama_url.rstrip("/") + "/api/ps", timeout=1.0) as resp:
            models = json.load(resp).get("models") or []
    except Exception:
        return []
    want = cfg.ollama_model if ":" in cfg.ollama_model else cfg.ollama_model + ":latest"
    return [(m.get("name") or m.get("model", ""), int(m.get("size_vram") or 0)) for m in models
            if want in (m.get("name"), m.get("model"))]


def _ollama_unload(cfg, name: str) -> bool:
    """Modeli Ollama'dan hemen boşaltır (keep_alive 0). Başarısızsa False."""
    import json
    import urllib.request

    body = json.dumps({"model": name, "keep_alive": 0}).encode()
    req = urllib.request.Request(cfg.ollama_url.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _deepl_backend(cfg, args):
    """Anahtarı alır (gerekirse sorar), doğrular ve kaydeder. Başarısızsa None."""
    from . import app

    key = cfg.resolved_auth_key()
    if not key:
        if not sys.stdin or not sys.stdin.isatty():
            print(f"DeepL anahtarı yok. {args.config} içinde \"deepl_auth_key\" alanına yazın.")
            return None
        print("DeepL API anahtarı gerekli (ücretsiz: https://www.deepl.com/pro-api → Hesap → API Keys).")
        key = input("Anahtarı yapıştırıp Enter'a basın: ").strip()
        if not key:
            return None
    backend = _make_deepl(cfg, key)
    try:
        # Anahtarı doğrular ve DeepL bağlantısını oyundan önce ısıtır.
        app.log(backend.usage_summary())
    except Exception as exc:
        print(app.describe_error(exc))
        return None
    if key != cfg.resolved_auth_key():
        cfg.deepl_auth_key = key
        config_mod.save(cfg, args.config)
        app.log(f"Anahtar {args.config} dosyasına kaydedildi")
    return backend


if __name__ == "__main__":
    code = main()
    # Arka plan iş parçacıkları (ses, çeviri havuzları) kapanışı bekletmesin: program kesin kapansın,
    # yoksa görünmez bir kopya kısayolları ve ekran kartını tutmaya devam eder.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code or 0)
