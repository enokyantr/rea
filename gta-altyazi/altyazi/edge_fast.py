"""Edge seslerine hazır (önceden açılmış) bağlantılarla istek: ilk ses ~0,1 sn'de gelir.

edge-tts her satır için yeni TLS/WebSocket bağlantısı kurar; ilk ses parçası ~0,8 sn sonra gelir.
Burada arka planda hep `POOL_SIZE` (4) kadar bağlantı açık ve ayarları (speech.config) gönderilmiş
tutulur; satır gelince yalnızca metin gönderilir. Her bağlantı tek satırda kullanılıp kapatılır
(aynı bağlantının yeniden kullanımı ölçümde düzensiz gecikti), yerine hemen yenisi açılır.
Dublaj bir satır için en çok 3 istek açabilir (konuşan tanıma bitmeden 2 olası sesle, tahmin
tutmazsa doğru sesle); 4 hazır bağlantı, art arda gelen satırda da yeni bağlantı kurmayı
(~0,3-0,7 sn) beklemesin diye. `on_audio` False döndürürse (ses artık istenmiyor) indirme bırakılır.

edge-tts'in iç yardımcılarını (DRM belirteci, SSML biçimi) kullanır; sürüm değişip bunlar
bulunamazsa ya da bağlantı açılamazsa `EdgeFastUnavailable` / hata verilir ve çağıran
normal edge-tts yoluna döner."""

from __future__ import annotations

import asyncio
import os
import ssl
import threading
import time
from typing import Callable
from xml.sax.saxutils import escape

POOL_SIZE = 4
MAX_IDLE_S = 60.0  # bundan eski hazır bağlantı kullanılmaz (sunucu kapatmış olabilir)

_CONFIG = ("X-Timestamp:{}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n"
           '{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{"sentenceBoundaryEnabled":"false",'
           '"wordBoundaryEnabled":"false"}},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}\r\n')


class EdgeFastUnavailable(RuntimeError):
    pass


class EdgeFast:
    """Kendi olay döngüsünü ayrı iş parçacığında çalıştırır; `request` her iş parçacığından çağrılabilir."""

    def __init__(self, log: Callable[[str], None] = print):
        try:
            import aiohttp
            import certifi
            from edge_tts import communicate as comm
            from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
            from edge_tts.data_classes import TTSConfig
            from edge_tts.drm import DRM
        except Exception as exc:  # edge-tts sürümü farklı: normal yol kullanılır
            raise EdgeFastUnavailable(f"edge-tts iç yapısı beklenenden farklı ({exc})") from exc
        self._aiohttp, self._comm, self._drm, self._config_cls = aiohttp, comm, DRM, TTSConfig
        self._url, self._headers, self._gec_version = WSS_URL, WSS_HEADERS, SEC_MS_GEC_VERSION
        self._ssl = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())
        self._log = log
        self._loop = asyncio.new_event_loop()
        self._ready: asyncio.Queue | None = None
        self._session = None
        self._closed = False
        self.failures = 0
        threading.Thread(target=self._run, name="edge-baglanti", daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._start(), self._loop).result(timeout=5)

    # --- dışarıdan ---------------------------------------------------------------------------

    def request(self, text: str, voice: str, rate: str, pitch: str, volume: str,
                on_audio: Callable[[bytes], None], on_end: Callable[[Exception | None], None],
                timeout: float) -> None:
        """Satırı istet; ses parçaları geldikçe `on_audio`, sonunda `on_end(hata ya da None)` çağrılır.
        `on_audio` False döndürürse istek bırakılır (bağlantı kapanır, `on_end(None)`). Hemen döner."""
        if self._closed:
            on_end(EdgeFastUnavailable("kapatıldı"))
            return
        coro = self._request(text, voice, rate, pitch, volume, on_audio, on_end, timeout)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=3)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # --- olay döngüsü ------------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    async def _start(self) -> None:
        self._ready = asyncio.Queue()
        self._session = self._aiohttp.ClientSession(trust_env=True)
        for _ in range(POOL_SIZE):
            asyncio.ensure_future(self._refill())

    async def _shutdown(self) -> None:
        while self._ready is not None and not self._ready.empty():
            _, ws = self._ready.get_nowait()
            await ws.close()
        if self._session is not None:
            await self._session.close()

    async def _open(self):
        comm, drm = self._comm, self._drm
        url = (f"{self._url}&ConnectionId={comm.connect_id()}&Sec-MS-GEC={drm.generate_sec_ms_gec()}"
               f"&Sec-MS-GEC-Version={self._gec_version}")
        try:
            ws = await self._session.ws_connect(url, compress=15, headers=drm.headers_with_muid(self._headers),
                                                ssl=self._ssl)
        except self._aiohttp.ClientResponseError as exc:
            if exc.status != 403:
                raise
            drm.handle_client_response_error(exc)  # bilgisayar saati kaymış: düzeltip bir kez daha
            ws = await self._session.ws_connect(url, compress=15, headers=drm.headers_with_muid(self._headers),
                                                ssl=self._ssl)
        await ws.send_str(_CONFIG.format(comm.date_to_string()))
        return ws

    async def _refill(self) -> None:
        """Havuza bir hazır bağlantı ekle (başarısızsa artan aralıkla yeniden dener)."""
        delay = 1.0
        while not self._closed:
            try:
                ws = await self._open()
            except Exception as exc:
                self.failures += 1
                if self.failures in (1, 10):
                    self._log(f"Edge hazır bağlantısı açılamadı ({exc}); normal bağlantıyla devam.")
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)
                continue
            self.failures = 0
            await self._ready.put((time.monotonic(), ws))
            return

    async def _take(self):
        """Taze bir hazır bağlantı; yoksa (ya da hepsi eskiyse) hemen yenisi açılır."""
        while not self._ready.empty():
            opened, ws = self._ready.get_nowait()
            asyncio.ensure_future(self._refill())
            if not ws.closed and time.monotonic() - opened < MAX_IDLE_S:
                return ws
            await ws.close()
        asyncio.ensure_future(self._refill())
        return await self._open()

    async def _request(self, text, voice, rate, pitch, volume, on_audio, on_end, timeout) -> None:
        error: Exception | None = None
        ws = None
        try:
            async with asyncio.timeout(timeout):
                ws = await self._take()
                await self._speak(ws, text, voice, rate, pitch, volume, on_audio)
        except Exception as exc:
            error = exc if not isinstance(exc, TimeoutError) else TimeoutError("Edge zaman aşımı")
        finally:
            if ws is not None:
                await ws.close()
        on_end(error)

    async def _speak(self, ws, text, voice, rate, pitch, volume, on_audio) -> None:
        comm, aiohttp = self._comm, self._aiohttp
        config = self._config_cls(voice, rate, volume, pitch, "SentenceBoundary")
        ssml = comm.mkssml(config, escape(comm.remove_incompatible_characters(text)))
        await ws.send_str(comm.ssml_headers_plus_data(comm.connect_id(), comm.date_to_string(), ssml))
        got_audio = False
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                if len(message.data) < 2:
                    continue
                header_length = int.from_bytes(message.data[:2], "big")
                _, data = comm.get_headers_and_data(message.data, header_length)
                if data:
                    got_audio = True
                    if on_audio(data) is False:
                        return  # ses artık istenmiyor (ör. tutmayan tahmin): bağlantı kapatılır
            elif message.type == aiohttp.WSMsgType.TEXT:
                if "Path:turn.end" in message.data:
                    break
            else:
                raise ConnectionError(f"Edge bağlantısı kapandı ({message.type})")
        else:
            raise ConnectionError("Edge bağlantısı beklenmedik şekilde kapandı")
        if not got_audio:
            raise ConnectionError("Edge ses göndermedi")
