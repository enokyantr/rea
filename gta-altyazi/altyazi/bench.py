"""Örnek oyun replikleriyle çeviri kalitesi/hız ölçümü (platformdan bağımsız; CI'da da çalışır)."""

from __future__ import annotations

import statistics
import time

BENCH_LINES = [
    "Get in the car, now!", "Where are we even going?", "I told you this was a bad idea.",
    "Relax, I've got a plan.", "Cops are right behind us!", "Take the next left, quick!",
    "We need to lose them first.", "Is that a helicopter?", "Just keep driving, Lucia.",
    "Hold on to something!", "That was way too close.", "You think this is a game, Jason? This is our life now.",
    # Sansürsüz çeviri kontrolü: küfür/argo yumuşatılmamalı.
    "What the fuck are you doing?", "Shut the fuck up, asshole!", "You stupid son of a bitch!",
    "Holy shit, that was close.", "This motherfucker owes me money.", "Get your ass in the car, bro.",
]


def run(backend, title: str, lines=BENCH_LINES, route=None) -> int:
    """route(satır) -> (çevirmen, etiket): oyundaki gibi satır başına çevirmen seçimi (isteğe bağlı)."""
    print(title, flush=True)
    times = []
    previous = None
    for line in lines:
        engine, label = route(line) if route else (backend, "")
        t0 = time.perf_counter()
        try:
            out = engine.translate(line, previous)
        except Exception as exc:  # tek satırın hatası ölçümü bitirmesin
            out = f"HATA: {type(exc).__name__}: {exc}"
        times.append(1000 * (time.perf_counter() - t0))
        print(f"{times[-1]:6.0f} ms  {label}{line}  →  {out}", flush=True)
        previous = line
    print(f"medyan {statistics.median(times):.0f} ms, en kötü {max(times):.0f} ms", flush=True)
    return 0
