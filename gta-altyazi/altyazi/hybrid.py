"""Karma çeviri: yerel model hemen taslak verir, DeepL gelince yerine geçer (platformdan bağımsız).

Sıralama garantisi: DeepL sonucu taslaktan önce gelirse taslak hiç gösterilmez; taslak
gösterildiyse DeepL sonucu her zaman ondan sonra iletilir (aynı kilit altında karar verilir)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

DRAFT, FINAL = "draft", "final"


class Backend(Protocol):
    def translate(self, text: str, context: str | None) -> str: ...


class FinalTranslator(Backend, Protocol):
    def cached(self, text: str) -> str | None: ...


# (aşama, çeviri, süre_saniye)
ResultFn = Callable[[str, str, float], None]
# (hata, aşama, taslak_gösterildi_mi)
ErrorFn = Callable[[Exception, str, bool], None]


class HybridTranslator:
    def __init__(self, final: FinalTranslator | None, draft: Backend | None):
        # Not: `if final` kullanma — CachedTranslator boş önbellekte len()==0 olduğu için False sayılır.
        if final is None and draft is None:
            raise ValueError("en az bir çevirmen gerekli")
        self.final = final
        self.draft = draft
        # Ayrı işçiler: yavaş DeepL isteği hızlı yerel taslağı bekletmesin.
        self._draft_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-mt") if draft is not None else None
        self._final_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepl") if final is not None else None

    @property
    def mode(self) -> str:
        if self.final is not None and self.draft is not None:
            return "hybrid"
        return "deepl" if self.final is not None else "local"

    def submit(self, text: str, context: str | None, is_current: Callable[[], bool],
               on_result: ResultFn, on_error: ErrorFn) -> None:
        if self.final is not None:
            hit = self.final.cached(text)
            if hit is not None:  # önbellek: kaliteli çeviri anında, taslağa gerek yok
                on_result(FINAL, hit, 0.0)
                return

        lock = threading.Lock()
        state = {"final_done": False, "draft_shown": False}
        # Yalnızca yerel model varsa onun sonucu nihai sonuçtur.
        draft_stage = DRAFT if self.final is not None else FINAL

        def run_draft() -> None:
            if not is_current():  # kuyrukta beklerken yeni satır geldi
                return
            started = time.perf_counter()
            try:
                translated = self.draft.translate(text, context)
            except Exception as exc:
                on_error(exc, DRAFT, False)
                return
            took = time.perf_counter() - started
            with lock:
                if state["final_done"] or not is_current():
                    return
                state["draft_shown"] = True
                on_result(draft_stage, translated, took)

        def run_final() -> None:
            if not is_current():
                return
            started = time.perf_counter()
            try:
                translated = self.final.translate(text, context)
            except Exception as exc:
                with lock:
                    shown = state["draft_shown"]
                on_error(exc, FINAL, shown)
                return
            took = time.perf_counter() - started
            with lock:
                state["final_done"] = True
                if is_current():
                    on_result(FINAL, translated, took)

        if self._draft_exec is not None:
            self._draft_exec.submit(run_draft)
        if self._final_exec is not None:
            self._final_exec.submit(run_final)

    def shutdown(self) -> None:
        for ex in (self._draft_exec, self._final_exec):
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
