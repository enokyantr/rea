"""Karma çeviri: yerel model hemen taslak verir, DeepL gelince yerine geçer (platformdan bağımsız).

Sıralama garantisi: DeepL sonucu taslaktan önce gelirse taslak hiç gösterilmez; taslak
gösterildiyse DeepL sonucu her zaman ondan sonra iletilir (aynı kilit altında karar verilir).

Önceden çeviri (`prefetch`): satır doğrulanmadan, ilk güvenilir okumada son çeviri başlatılır.
Doğrulanan satır aynıysa (önbellek anahtarı ve bağlam aynı) sonuç yeniden kullanılır; farklıysa
hiç gösterilmez/dublajlanmaz ve önbelleğe de yazılmaz. Bu mantığın tek sahibi bu modüldür."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Protocol

from . import textproc

DRAFT, FINAL = "draft", "final"


class Backend(Protocol):
    def translate(self, text: str, context: str | None) -> str: ...


class FinalTranslator(Protocol):
    # `source`: satırın kaynağı ("ocr" ekrandaki altyazı, "asr" konuşma tanıma, "phone" telefon). İstemi değiştiren
    # arka uç (yerel LLM) kullanır ve anahtara ekler; diğerleri yok sayar (bkz. CachedTranslator).
    def translate(self, text: str, context: str | None, source: str = "ocr") -> str: ...
    def cached(self, text: str, source: str = "ocr") -> str | None: ...
    # Önceden çeviri için (CachedTranslator); yoksa prefetch yalnızca taslağı hazırlar.
    # def key(self, text: str, source: str = "ocr") -> str: ...
    # def translate_uncommitted(self, text: str, context, cancel, source) -> tuple[str, str]: ...
    # def commit(self, key: str, translated: str) -> None: ...


# (aşama, çeviri, süre_saniye)
ResultFn = Callable[[str, str, float], None]
# (hata, aşama, taslak_gösterildi_mi)
ErrorFn = Callable[[Exception, str, bool], None]


def _context_sig(context: Any) -> tuple:
    """İsteği etkileyen bağlam: önceki satır (str değeri) ve sohbet geçmişi. Dialogue bir str alt
    sınıfı olduğundan yalnızca == geçmişi karşılaştırmaz."""
    if context is None:
        return (None, ())
    return (str(context), tuple(getattr(context, "history", ())))


class _Speculation:
    """Bir bölgenin doğrulanmadan başlatılmış son çeviri isteği. Durum `_spec_lock` altında değişir."""

    __slots__ = ("key", "source", "context_fn", "context", "context_sig", "started", "adopted", "future",
                 "cancel")

    def __init__(self, key: str, source: str, context_fn: Callable[[], Any]):
        self.key = key
        self.source = source
        self.context_fn = context_fn
        self.context: Any = None
        self.context_sig: tuple | None = None  # istek başlayınca belli olur
        self.started = False
        self.adopted = False  # doğrulanan satır bu isteği sahiplendi: iptal edilmez
        self.future: Future = Future()  # sonuç: (önbellek anahtarı, çeviri)
        # Kurulunca sürmekte olan istek yarıda kesilir (arka uç destekliyorsa, ör. Ollama akışı):
        # boşa çıkan tahmin, doğrulanan satırın isteğini ekran kartında bekletmesin.
        self.cancel = threading.Event()

    def drop(self) -> None:
        """Bu istek artık kimsenin değil (`_spec_lock` altında çağrılır)."""
        if not self.adopted:
            self.cancel.set()

    def failed(self) -> bool:
        return self.future.done() and (self.future.cancelled() or self.future.exception() is not None)


class _Stale:
    """Satır bayatlayınca ("yeni satır geldi", "yazı kayboldu") kurulmuş görünen iptal bayrağı:
    LLM isteği akışla alınır ve her parçada buna bakılır; bayat satırın üretimi yarıda kesilir, ekran
    kartı sıradaki satıra hemen kalır (hızlı konuşmada eski satırın bitmesi beklenmez)."""

    def __init__(self, is_current: Callable[[], bool]):
        self._is_current = is_current

    def is_set(self) -> bool:
        return not self._is_current()


# Ana altyazının LLM isteği sürerken ek bölgelerin (görev, mesaj, telefon) istekleri bekler.
_PRIORITY_LOCK = threading.Lock()
_main_active = 0
LOW_PRIORITY_WAIT_S = 2.5


class HybridTranslator:
    # True: bu çevirici ana altyazı değil (görev/telefon...); ana altyazının LLM isteği varken
    # en çok LOW_PRIORITY_WAIT_S bekler (tek ekran kartında diyalog satırları öne geçer).
    low_priority = False

    def __init__(self, final: FinalTranslator | None, draft: Backend | None,
                 final_for: Callable[[str], bool] | None = None,
                 final_only: Callable[[str], bool] | None = None):
        # Not: `if final` kullanma — CachedTranslator boş önbellekte len()==0 olduğu için False sayılır.
        if final is None and draft is None:
            raise ValueError("en az bir çevirmen gerekli")
        self.final = final
        self.draft = draft
        # Son çevirmen yalnızca bu koşulu sağlayan satırlarda kullanılır (ör. küfürlü satırlar);
        # diğerlerinde taslak çeviri nihai sonuçtur. None: her satır.
        self.final_for = final_for
        # Bu satırlar yalnızca son çevirmene gider, taslak hiç çalışmaz (ör. İspanyolca satır: yerel
        # taslak model yalnızca İngilizce bilir, bozuk taslak göstermektense son çeviri beklenir).
        self.final_only = final_only
        # Ayrı işçiler: yavaş DeepL isteği hızlı yerel taslağı bekletmesin.
        self._draft_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-mt") if draft is not None else None
        self._final_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepl") if final is not None else None
        # Yerel taslak çeviri işlemcide çalışırken OCR işlemciyi paylaşmasın diye (bkz. draft_running).
        self._draft_active = threading.Event()
        # Taslak çeviri önbelleği (bellekte): tekrar eden satır (aynı replik, menü yazısı) anında.
        self._draft_cache: OrderedDict[str, str] = OrderedDict()
        self._draft_lock = threading.Lock()
        # Önceden çeviri: bölge başına (bu nesne) en fazla bir son çeviri adayı ve bir taslak adayı.
        self._spec_lock = threading.Lock()
        self._spec: _Speculation | None = None
        self._draft_spec: str | None = None

    DRAFT_CACHE_SIZE = 512

    def _draft_get(self, text: str) -> str | None:
        with self._draft_lock:
            hit = self._draft_cache.get(text)
            if hit is not None:
                self._draft_cache.move_to_end(text)
            return hit

    def _draft_put(self, text: str, translated: str) -> None:
        with self._draft_lock:
            self._draft_cache[text] = translated
            self._draft_cache.move_to_end(text)
            while len(self._draft_cache) > self.DRAFT_CACHE_SIZE:
                self._draft_cache.popitem(last=False)

    def draft_running(self) -> bool:
        """Yerel taslak çeviri şu an çalışıyor mu (işlemci yoğun; OCR kısa süre beklemeli)."""
        return self._draft_active.is_set()

    @property
    def mode(self) -> str:
        if self.final is not None and self.draft is not None:
            return "hybrid"
        return "deepl" if self.final is not None else "local"

    def _use_final(self, text: str) -> bool:
        return self.final is not None and (
            self.draft is None or self.final_for is None or self.final_for(text) or self._no_draft(text))

    def _no_draft(self, text: str) -> bool:
        return self.final is not None and self.final_only is not None and self.final_only(text)

    # --- önceden çeviri ------------------------------------------------------------------------

    def prefetch(self, text: str, context_fn: Callable[[], Any], draft: bool = False,
                 source: str = "ocr") -> bool:
        """Henüz doğrulanmamış (ama büyük olasılıkla doğru) okumanın çevirisini şimdiden başlatır.

        `context_fn` istek başlarken çağrılır (önceki satırın çevirisi kuyrukta önde bitince sohbet
        geçmişi güncel olsun); `submit` istek başlamadan gelirse onun bağlamı kullanılır. Yenisi
        eskisinin yerine geçer: başlamamış eski istek hiç gönderilmez, sürmekte olanın sonucu atılır.
        `draft`: yerel taslağı da hazırla (işlemciyi OCR ile paylaşmıyorsa). `source`: satır kaynağı
        ("ocr"/"asr"; anahtarın parçası, `submit` ile aynı verilmeli). Yeni son çeviri isteği
        başlatıldıysa True."""
        started = False
        if (self.final is not None and hasattr(self.final, "translate_uncommitted")
                and self._use_final(text) and self.final.cached(text, source=source) is None):
            key = self.final.key(text, source=source)
            with self._spec_lock:
                current = self._spec
                if current is None or current.key != key or current.failed():
                    if current is not None:
                        current.drop()
                    spec = _Speculation(key, source, context_fn)
                    self._spec = spec
                    started = True
            if started:
                self._final_exec.submit(self._run_spec, spec, text)
        if draft and self.draft is not None and not self._no_draft(text) and self._draft_get(text) is None:
            with self._spec_lock:
                queued = self._draft_spec == text
                self._draft_spec = text
            if not queued:
                self._draft_exec.submit(self._run_draft_spec, text, context_fn)
        return started

    def cancel_prefetch(self) -> None:
        """Satır kayboldu/bölge değişti: başlamamış önceden çeviriler gönderilmez."""
        with self._spec_lock:
            if self._spec is not None:
                self._spec.drop()
            self._spec = None
            self._draft_spec = None

    def _run_spec(self, spec: _Speculation, text: str) -> None:
        with self._spec_lock:
            if self._spec is not spec and not spec.adopted:
                spec.future.cancel()  # yerine yenisi geldi ya da satır farklı doğrulandı
                return
            if not spec.adopted:
                spec.context = spec.context_fn()
            spec.context_sig = _context_sig(spec.context)
            spec.started = True
            spec.future.set_running_or_notify_cancel()
        self._yield_to_main(lambda: not spec.cancel.is_set())
        self._enter()
        try:
            result = self.final.translate_uncommitted(text, spec.context, cancel=spec.cancel,
                                                      source=spec.source)
        except Exception as exc:
            spec.future.set_exception(exc)
        else:
            spec.future.set_result(result)
        finally:
            self._leave()

    def _run_draft_spec(self, text: str, context_fn: Callable[[], Any]) -> None:
        with self._spec_lock:
            if self._draft_spec != text:
                return
        try:
            if self._draft_get(text) is not None:
                return
            self._draft_active.set()
            try:
                translated = self.draft.translate(textproc.expand_slang(text), context_fn())
            except Exception:
                return  # gerçek istek yeniden dener ve hatayı o bildirir
            finally:
                self._draft_active.clear()
            self._draft_put(text, translated)
        finally:
            with self._spec_lock:
                if self._draft_spec == text:
                    self._draft_spec = None

    def _take_spec(self, text: str, context: Any, use_final: bool, source: str) -> _Speculation | None:
        """Doğrulanan satır önceden çeviriyle aynı istekse onu sahiplenir; her durumda bekleyen
        aday temizlenir (farklı satırın sonucu bu satırda asla kullanılmaz)."""
        with self._spec_lock:
            spec, self._spec = self._spec, None
            if self._draft_spec != text:
                self._draft_spec = None
            if spec is None:
                return None
            if not use_final or spec.key != self.final.key(text, source=source):
                spec.drop()
                return None
            if spec.started:
                # İstem birebir aynı olmalı: geçmiş bu arada değiştiyse yeni istek.
                if spec.context_sig != _context_sig(context) or spec.failed():
                    spec.drop()
                    return None
            else:
                spec.context = context  # henüz gönderilmedi: gerçek bağlamla ve kuyruktaki yeriyle
            spec.adopted = True
            return spec

    # --- çeviri --------------------------------------------------------------------------------

    def _route(self, text: str, draft: bool) -> bool:
        """Son çevirmen kullanılacak mı? `draft=False`: son çevirmen varsa yalnızca o (yönlendirme
        kuralından bağımsız, taslak hiç çalışmaz); yoksa taslak nihai sonuçtur."""
        return self._use_final(text) or (not draft and self.final is not None)

    def cached(self, text: str, source: str = "ocr", draft: bool = True) -> str | None:
        """`submit`'in bu satır için anında vereceği kesin çeviri (önbellekte), yoksa None."""
        if self._route(text, draft):
            return self.final.cached(text, source=source)
        return self._draft_get(text) if self.draft is not None else None

    def submit(self, text: str, context: str | None, is_current: Callable[[], bool],
               on_result: ResultFn, on_error: ErrorFn, source: str = "ocr", draft: bool = True) -> None:
        """`source`: satırın kaynağı ("ocr" ekrandaki altyazı, "asr" konuşma tanıma, "phone" telefon
        ekranı). Yalnızca son çevirmene geçer; yerel taslak çevirmen kaynağı kullanmaz.
        `draft=False` (telefon): taslak gösterilmez, yazı bir kez (son çeviriyle) gelir; son çevirmen
        yoksa taslak son çeviri olarak verilir."""
        use_final = self._route(text, draft)
        spec = self._take_spec(text, context, use_final, source)
        if use_final:
            hit = self.final.cached(text, source=source)
            if hit is not None:  # önbellek: kaliteli çeviri anında, taslağa gerek yok
                on_result(FINAL, hit, 0.0)
                return
            if spec is not None and spec.future.done():  # önceden çeviri hazır: anında
                key, translated = spec.future.result()
                self.final.commit(key, translated)
                on_result(FINAL, translated, 0.0)
                return

        if not use_final and self.draft is not None:
            hit = self._draft_get(text)
            if hit is not None:  # taslak nihai sonuç ve önbellekte: anında
                on_result(FINAL, hit, 0.0)
                return

        lock = threading.Lock()
        state = {"final_done": False, "draft_shown": False}
        # Son çevirmen kullanılmayacaksa taslak nihai sonuçtur.
        draft_stage = DRAFT if use_final else FINAL

        def run_draft() -> None:
            if not is_current():  # kuyrukta beklerken yeni satır geldi
                return
            started = time.perf_counter()
            cached = self._draft_get(text)
            self._draft_active.set()
            try:
                translated = cached if cached is not None else self.draft.translate(textproc.expand_slang(text), context)
            except Exception as exc:
                on_error(exc, DRAFT, False)
                return
            finally:
                self._draft_active.clear()
            self._draft_put(text, translated)
            took = time.perf_counter() - started
            with lock:
                if state["final_done"] or not is_current():
                    return
                state["draft_shown"] = True
                on_result(draft_stage, translated, took)

        submitted = time.perf_counter()

        def finish_final(translated: str, took: float) -> None:
            with lock:
                state["final_done"] = True
                if is_current():
                    on_result(FINAL, translated, took)

        def fail_final(exc: Exception) -> None:
            with lock:
                shown = state["draft_shown"]
            on_error(exc, FINAL, shown)

        def run_final() -> None:
            if not is_current():
                return
            self._yield_to_main(is_current)
            if not is_current():
                return
            started = time.perf_counter()
            self._enter()
            try:
                if hasattr(self.final, "translate_uncommitted"):
                    key, translated = self.final.translate_uncommitted(text, context, cancel=_Stale(is_current),
                                                                       source=source)
                    self.final.commit(key, translated)
                else:
                    translated = self.final.translate(text, context, source=source)
            except Exception as exc:
                if is_current():  # bayatlayan satırın iptali/hatası sessiz: yeni satır zaten ekranda
                    fail_final(exc)
                return
            finally:
                self._leave()
            finish_final(translated, time.perf_counter() - started)

        def spec_done(future: Future) -> None:
            # Sahiplenilen önceden çeviri aynı istektir. Başarısız olduysa (ya da iptal edildiyse,
            # olmamalı) satır normal yoldan bir kez daha istenir; hata o istekten bildirilir.
            if future.cancelled() or future.exception() is not None:
                try:
                    self._final_exec.submit(run_final)
                except RuntimeError:
                    pass  # kapatılıyor
                return
            key, translated = future.result()
            self.final.commit(key, translated)
            finish_final(translated, time.perf_counter() - submitted)

        if self._draft_exec is not None and not self._no_draft(text) and (draft or not use_final):
            self._draft_exec.submit(run_draft)
        if spec is not None:
            spec.future.add_done_callback(spec_done)
        elif use_final:
            self._final_exec.submit(run_final)

    def _enter(self) -> None:
        global _main_active
        if not self.low_priority:
            with _PRIORITY_LOCK:
                _main_active += 1

    def _leave(self) -> None:
        global _main_active
        if not self.low_priority:
            with _PRIORITY_LOCK:
                _main_active = max(0, _main_active - 1)

    def _yield_to_main(self, is_current: Callable[[], bool]) -> None:
        """Düşük öncelikli çevirici: ana altyazının isteği bitene kadar (en çok 2,5 sn) bekler."""
        if not self.low_priority:
            return
        deadline = time.monotonic() + LOW_PRIORITY_WAIT_S
        while _main_active > 0 and time.monotonic() < deadline and is_current():
            time.sleep(0.03)

    def shutdown(self) -> None:
        self.cancel_prefetch()
        for ex in (self._draft_exec, self._final_exec):
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)


def draft_hold_s(original: str, max_ms: int = 1000) -> float:
    """Taslağın bekletileceği süre (sn): son çevirinin tahmini gelişi (≈ 0,25 sn + 11 ms/harf, kayıtlarda
    40 harf ~0,7 sn, 100 harf ~1,3 sn) + 150 ms; en çok `max_ms`. Kısa/hızlı satırlarda gereksiz
    beklenmez; son çeviri gelince bekletme zaten biter."""
    expected = 0.25 + 0.011 * len(original) + 0.15
    return max(0.0, min(max_ms / 1000, expected)) if max_ms > 0 else 0.0
