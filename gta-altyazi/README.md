# GTA Altyazı — Remote Play için anlık Türkçe altyazı

PS Remote Play penceresindeki **İngilizce altyazıyı** ekrandan okur (Windows OCR),
**DeepL** ile Türkçeye çevirir ve orijinal altyazının üstünde gösterir.
Oyuna, konsola veya Remote Play'e dokunmaz; yalnızca ekranı okur.

```
Remote Play penceresi ─► altyazı bölgesini yakala ─► Windows OCR ─► kararlılık/tekrar süzgeci
                                                                     │
                     tıklanamaz şeffaf overlay ◄── DeepL (önbellekli) ◄┘
```

## Gereksinimler

- Windows 10/11
- Python 3.10+ (python.org; kurulumda **Add python.exe to PATH** işaretli)
- DeepL API anahtarı — <https://www.deepl.com/pro-api> (Free plan: ayda 500.000 karakter)
- Windows İngilizce OCR paketi. Windows'unuz Türkçeyse büyük ihtimalle eksiktir;
  **yönetici** PowerShell'de:
  ```powershell
  Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"
  ```

## Kurulum

1. `kurulum.bat` dosyasını çift tıklayın.
2. `baslat.bat --dry-run` ile bir kez çalıştırın; `config.json` oluşur (Ctrl+Alt+Q ile kapatın).
3. `config.json` içinde `"deepl_auth_key"` alanına anahtarınızı yazın
   (veya `DEEPL_AUTH_KEY` ortam değişkenini ayarlayın).

## Kullanım

1. Oyunda **İngilizce altyazıyı açın**, mümkünse altyazı boyutunu büyük yapın.
2. PS Remote Play'i açın; tam ekran yerine **ekranı kaplayan pencere** daha sorunsuzdur.
3. `baslat.bat` ile programı başlatın.
4. İlk seferde **Ctrl+Alt+R** ile altyazının çıktığı alanı fareyle seçin.
   Seçim Remote Play penceresine göre oransal kaydedilir; pencere taşınsa/boyutlansa da uyar.

| Kısayol | İşlev |
|---|---|
| Ctrl+Alt+R | Altyazı bölgesini seç |
| Ctrl+Alt+S | Duraklat / devam |
| Ctrl+Alt+O | Türkçenin üstünde İngilizceyi de göster |
| Ctrl+Alt+Q | Çıkış |

## Ayar yapma (oyun çıkınca)

Remote Play'de bir altyazı sahnesinin ekran görüntüsünü alın (Win+Shift+S → kaydet) ve:

```
baslat.bat --ocr-test ekran.png          (config'deki bölgeyi okur)
baslat.bat --ocr-test ekran.png --full   (tüm resmi okur)
```

Okunan satırlar ve Türkçesi yazdırılır, OCR'a giren görüntü `ekran_ocr_girdi.png` olarak kaydedilir.
Canlı kullanımda süreleri görmek için `baslat.bat --debug`. Pencere başlığı farklıysa
`baslat.bat --list-windows` ile bakıp `window_titles` listesine ekleyin.

`config.json` içindeki önemli ayarlar:

| Ayar | Varsayılan | Ne zaman değiştirilir |
|---|---|---|
| `preprocess` | `"none"` | Arka plan OCR'ı bozuyorsa `"bright"` (yalnızca beyaz yazıyı bırakır) |
| `bright_threshold` | `190` | `"bright"` modunda yazı eksik okunuyorsa düşürün |
| `upscale` | `2.0` | Küçük yazıda 2.5–3; yavaşsa 1.5 |
| `capture_interval_ms` | `120` | Daha hızlı tepki için düşürün (daha çok CPU) |
| `stable_reads` | `2` | Yanlış/yarım satırlar çevriliyorsa 3 |
| `clear_after_ms` | `700` | Türkçe altyazı çok erken/geç kayboluyorsa |
| `cover_original` | `true` | `false`: orijinal altyazının üstü kapatılmaz |
| `font_size`, `font_family` | `26`, `Segoe UI` | Görünüm |

## Notlar

- Gecikme yaklaşık: yakalama aralığı × `stable_reads` + OCR (~20–60 ms) + DeepL (~150–400 ms).
  Tekrar eden satırlar `ceviri_onbellek.jsonl` önbelleğinden anında gelir ve kota harcamaz.
- Önceki satır DeepL'e bağlam olarak gönderilir (ücretlendirilmez); çeviri daha tutarlı olur.
- Altyazısı olmayan konuşmalar (ör. yoldan geçenler, radyo) çevrilmez.
- Overlay, tam ekran özel (exclusive fullscreen) uygulamaların üstünde görünmeyebilir; Remote Play
  bu modu kullanmaz ama sorun olursa pencere moduna alın.

## Geliştirme

```
python -m pip install pytest numpy pillow
python -m pytest tests
```

Testler Windows olmadan çalışan kısımları (metin temizleme, kararlılık, bölge hesabı,
değişim tespiti, önbellek, ayarlar) kapsar. Ekran yakalama, Windows OCR ve overlay yalnızca
Windows'ta denenebilir.
