"""Yerel (çevrimdışı) İngilizce→Türkçe çeviri: CTranslate2 + Opus-MT (Marian), NVIDIA GPU varsa GPU'da."""

from __future__ import annotations

import glob
import os
import sys
import threading
from pathlib import Path


def _add_cuda_dll_dirs() -> list[str]:
    """pip'in nvidia-* paketlerindeki CUDA DLL'lerini (cublas vb.) Windows'un bulmasını sağlar.

    PyInstaller paketinde bunlar _internal/nvidia/<paket>/bin altında, normal kurulumda
    site-packages/nvidia/<paket>/bin altında durur."""
    if sys.platform != "win32":
        return []
    roots = list(sys.path)
    if getattr(sys, "_MEIPASS", None):
        roots.insert(0, sys._MEIPASS)
    added = []
    for root in roots:
        for bin_dir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            if bin_dir not in added and os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                # ctranslate2 bazı DLL'leri PATH üzerinden de arar.
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                added.append(bin_dir)
    return added


class LocalBackend:
    """Taslak çevirmen. `translate` iş parçacığı güvenlidir (CTranslate2 kendi kuyruğunu tutar)."""

    def __init__(self, model_dir: Path, device: str = "auto", beam_size: int = 1):
        _add_cuda_dll_dirs()
        import ctranslate2
        import sentencepiece as spm

        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(f"Yerel çeviri modeli bulunamadı: {model_dir}")
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        try:
            self._tr = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type,
                                              inter_threads=1, intra_threads=0 if device == "cuda" else 4)
        except Exception:
            if device != "cuda":
                raise
            # CUDA kütüphanesi eksik/uyumsuz: işlemciyle devam et.
            device, compute_type = "cpu", "int8"
            self._tr = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8",
                                              inter_threads=1, intra_threads=4)
        self.device = device
        self._src = spm.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
        self._tgt = spm.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
        self._beam = max(1, beam_size)
        self._lock = threading.Lock()
        self.translate("Warm up.", None)  # ilk çağrıdaki gecikmeyi (çekirdek yükleme) oyundan önce öde

    def translate(self, text: str, context: str | None) -> str:
        tokens = self._src.encode(text, out_type=str) + ["</s>"]
        with self._lock:
            result = self._tr.translate_batch([tokens], beam_size=self._beam, max_decoding_length=256)
        hyp = [t for t in result[0].hypotheses[0] if t not in ("</s>", "<pad>")]
        return self._tgt.decode(hyp)

    def describe(self) -> str:
        return f"Yerel çeviri: {'NVIDIA GPU' if self.device == 'cuda' else 'işlemci'} ({self.device})"
