# GTA Altyazı — Remote Play için anlık Türkçe altyazı

PS Remote Play penceresindeki **İngilizce altyazıyı** ekrandan okur (Windows OCR),
**DeepL API** ile Türkçeye çevirir ve İngilizce altyazının hemen üstünde gösterir.
Oyuna, konsola veya Remote Play'e dokunmaz; yalnızca ekranı okur.

```
Remote Play penceresi ─► altyazı bölgesini yakala ─► Windows OCR ─► kararlılık/tekrar süzgeci
          │                                                          │
          └─► Türkçenin arkasındaki sahnenin parlaklığı              ▼
                    │                                  DeepL (hızlı model, önbellek, sıcak bağlantı)
                    ▼                                                │
          yazı rengi: açık sahnede siyah, koyu sahnede beyaz ──► tıklanamaz şeffaf overlay
```

**Karma çeviri (varsayılan):** Satır görünür görünmez bilgisayardaki yerel model (Opus-MT, NVIDIA
GPU'da) Türkçesini ~10–30 ms'de gösterir; aynı anda DeepL'e gönderilen daha kaliteli çeviri gelince
(~100–300 ms) yerine geçer. Daha önce görülmüş satırlar DeepL önbelleğinden anında gelir. DeepL'e
ulaşılamazsa yerel çeviri ekranda kalır. `config.json` → `translation_mode`:
`"hybrid"` (varsayılan), `"deepl"` (yalnızca DeepL) veya `"local"` (yalnızca yerel, internet/anahtar gerekmez).

**Anlık renk:** Türkçe yazının arkasındaki görüntü her karede (~50 ms) ölçülür. Kar, gökyüzü,
beyaz duvar gibi açık bir sahneye gelince yazı siyah olur, beyaz kontur alır; karanlıkta beyaza döner.
Yarısı açık yarısı koyu sahnede kontur kalınlaşır. Tek karelik patlama ve flaşlarda renk titremez.
Ölçüm Türkçe yazının hemen üstündeki şeritten yapılır; böylece kendi yazımız ölçüme karışmaz.

**Yayın ve ekran görüntüsü:** Türkçe altyazı OBS/Discord yayınında ve ekran görüntülerinde görünür
(`capture_visible: true`, varsayılan). Yazı İngilizce altyazının üstünde, okunan bölgenin dışında
durduğu için program kendi Türkçe yazısını okumaz; taşarsa o kısım OCR'dan önce karartılır.
`"cover"` modundaki kutu (görev/telefon bölgeleri dahil) okunan yazının üstünü örttüğü için
yakalamadan gizli kalır. Son satırlar paneli (Ctrl+Alt+H) kişiseldir, yayında görünmez.

## Gereksinimler

- Windows 10 (2004+) / Windows 11
- Hazır `.exe` kullanılmayacaksa Python 3.11+ (python.org; kurulumda **Add python.exe to PATH** işaretli)
- DeepL API anahtarı — <https://www.deepl.com/pro-api> (Free plan: ayda 500.000 karakter)
- Windows İngilizce OCR paketi. Windows'unuz Türkçeyse büyük ihtimalle eksiktir;
  **yönetici** PowerShell'de:
  ```powershell
  Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"
  ```

## Kurulum

**Hazır exe (önerilen):** GitHub'da *Actions → GTA Altyazi Windows build* altındaki son başarılı
çalıştırmadan `GTA-Altyazi-windows` (NVIDIA GPU destekli, büyük) veya `GTA-Altyazi-windows-cpu`
(GPU dosyaları yok, ~550 MB daha küçük; yerel çeviri işlemcide ~60–110 ms) dosyasını indirin, zip'i açın ve `GTA-Altyazi.exe`'yi çalıştırın.
Python gerekmez. Aşağıdaki `baslat.bat` komutlarının hepsi `.\GTA-Altyazi.exe` ile de çalışır
(PowerShell'de programın başına `.\` yazılmalıdır).
İlk iş olarak öz testi çalıştırın:

```
.\GTA-Altyazi.exe --self-test
```

OCR, overlay, ekran yakalamadan gizleme, beyaz sahnede renk seçimi ve (anahtar girildiyse) DeepL
gecikmesini bu bilgisayarda dener; her satır `TAMAM`/`HATA` yazar.

**Kaynak koddan (güncellemeler tek tıkla):** Python 3.12 ve Git kurulu olmalı.

```
git clone -b claude/gta6-auto-subtitle-tool-9uumiv https://github.com/enokyantr/rea.git gta
```

`gta\gta-altyazi` klasöründe `kurulum.bat`'ı çalıştırın (bir kerelik), exe paketindeki `model`
klasörünü buraya kopyalayın ve `baslat.bat` ile başlatın (`baslat.bat --self-test` vb. aynı
seçenekler geçerli). Yeni sürüm için yalnızca `guncelle.bat`.

**Python ile (eski yöntem):** (yerel model exe paketinde gelir; Python kurulumunda exe'deki `model` klasörünü
proje klasörüne kopyalayın ve `pip install -r requirements-gpu.txt` çalıştırın, yoksa yalnızca DeepL kullanılır)

1. `kurulum.bat` dosyasını çift tıklayın.
2. `baslat.bat` ile çalıştırın. İlk açılışta DeepL API anahtarı sorulur; yapıştırın.
   Anahtar doğrulanır ve `config.json` dosyasına kaydedilir
   (isterseniz `DEEPL_AUTH_KEY` ortam değişkenini de kullanabilirsiniz).
   Anahtarsız denemek için: `baslat.bat --dry-run`.

## Kullanım

1. Oyunda **İngilizce altyazıyı açın**, mümkünse altyazı boyutunu büyük yapın.
2. PS Remote Play'i açın; tam ekran yerine **ekranı kaplayan pencere** daha sorunsuzdur.
3. `baslat.bat` ile programı başlatın. **Ayar menüsü** açılır (sonra **Ctrl+Alt+M** ile aç/gizle).
4. İlk seferde menüden ya da **Ctrl+Alt+R** ile altyazının çıktığı alanı fareyle seçin.
   Seçim Remote Play penceresine göre oransal kaydedilir; pencere taşınsa/boyutlansa da uyar.

### Menü (Ctrl+Alt+M)

- **Durum:** çeviriyi duraklat/devam ettir; altyazı, görev ve telefon okumasının her biri için
  "Çalışıyor ✔ / Kapalı / Bölge seçilmedi" durumu, aç/kapat, bölge seç, hazır konum, kaldır;
  kullanılan OCR ve çeviri motorları; son satır ve gecikmesi.
- **Görünüm:** yazı boyutu, konumu, kontur, yazı tipi, İngilizceyi de göster, yayında görünürlük.
  Anında uygulanır.
- **Çeviri / OCR ve hız:** son çevirmen, model, sansürsüz çeviri, OCR motoru ve cihazı, hızlı kabul.
  Motor değiştiren ayarlar **Kaydet ve yeniden başlat** ile uygulanır.
- **Kısayollar:** her işlemin kısayolu ve durumu ("✔ çalışıyor", "✖ başka program kullanıyor",
  "— atanmamış"). İşlemi seçip **Değiştir**'e basın, yeni tuş birleşimine basın (Ctrl ve/veya Alt ile).
  **Boştaki kısayolları göster** hangi Ctrl+Alt birleşimlerinin kullanılabildiğini listeler.
- **Ses ve kayıt:** sesli okuma (dublaj) ve oturum kaydı.

Varsayılan kısayollar (menüden değiştirilebilir):

| Kısayol | İşlev |
|---|---|
| Ctrl+Alt+M | Menüyü aç / gizle |
| Ctrl+Alt+R | Altyazı bölgesini seç |
| Ctrl+Alt+G | Görev hedefi bölgesini seç (seçicide Delete: kaldır) |
| Ctrl+Alt+T | Telefon mesajı bölgesini seç (seçicide Delete: kaldır) |
| Ctrl+Alt+S | Duraklat / devam |
| Ctrl+Alt+PageUp / PageDown | Yazıyı büyüt / küçült |
| Ctrl+Alt+↑ / ↓ | Yazıyı yukarı / aşağı taşı |
| Ctrl+Alt+O | Türkçenin üstünde İngilizceyi de göster |
| Ctrl+Alt+H | Son satırlar paneli (aç/kapat) |
| Ctrl+Alt+D | Dublaj aç / kapat |
| Ctrl+Alt+N | Sokak konuşmalarını aç / kapat |
| Ctrl+Alt+Q | Çıkış |

### Dublaj (Ctrl+Alt+D ya da menüdeki "Dublaj" anahtarı; program durmadan açılıp kapanır)

Türkçe satır, konuşan karakterin sesine uygun bir sesle okunur:

1. Altyazı belirdiği andan itibaren oyun sesinin ilk ~0,6 saniyesi dinlenir (hoparlöre giden sesin
   kopyası; hiçbir yere kaydedilmez/gönderilmez).
2. Ses perdesinden kadın/erkek, ses renginden karakter bulunur. Hafızadaki bir karaktere benziyorsa
   aynı karakter sayılır ve **hep aynı sesle** okunur; yoksa yeni karakter eklenir
   (`karakter_sesleri.json`). Aynı cinsiyetteki karakterler farklı perde/hızla ayrılır.
3. Ses: **edge** (varsayılan) Microsoft Edge'in doğal Türkçe sesleri, kadın *Emel*, erkek *Ahmet*.
   Ücretsizdir ama internet gerekir ve Türkçe metin Microsoft'a gönderilir; resmi bir API değildir.
   Ulaşılamazsa satır **windows** (çevrimdışı) sesiyle okunur; Windows'ta Türkçe ses genelde tektir,
   kadın sesi perdesi yükseltilerek yapılır.
4. Yeni satır gelince önceki okuma hemen kesilir.
5. **Duygu:** konuşanın bu satırdaki ses gücü ve perdesi **kendi normaline göre** (bağırıyor mu,
   alçak sesle mi konuşuyor) ve metindeki ipuçları (ünlem, BÜYÜK HARF, küfür, korku/üzüntü/heyecan
   kelimeleri) birleştirilir: sakin, heyecanlı, sinirli, üzgün ya da korkmuş. Türkçe ses buna göre
   hızlanır/yavaşlar, incelir/kalınlaşır, yükselir/kısılır. Her karakterin bir "duygu gücü" vardır.
6. **Hızlı tanıma:** hafızadaki bir karakter genelde ilk ~0,3 sn'de tanınır; yalnızca emin olunamazsa
   0,6 sn dinlenir. Menü > Dublaj'da son konuşanın kim olduğu, duygusu ve kaç ms'de tanındığı görünür.

Ses havuzu (menü > Dublaj): **turkce** yalnızca Emel/Ahmet kullanır (karakterler perde/hızla ayrılır);
**genis** Microsoft'un çok dilli seslerini de kullanır: her karakter gerçekten farklı bir sesle konuşur
ama bu seslerin Türkçesi aksanlı olabilir. "Sesini dinle" ile deneyip karar verin.

**Windows (çevrimdışı) sesi:** Windows'ta Türkçe için yalnızca erkek ses (Tolga) vardır. Kurmak için:
Ayarlar > Saat ve dil > Konuşma > Sesleri yönet > Ses ekle > Türkçe, sonra bilgisayarı yeniden başlatın.

Konuşan tanıma oyun müziği ve efektler karıştığında yanılabilir: menü > Dublaj > Karakter hafızası'ndan
karakterin cinsiyetini düzeltebilir (kalıcıdır), sesini dinleyebilir, silebilir ya da hafızayı
sıfırlayabilirsiniz.

### Karma dublaj (Gemini + Edge)

Menü > Dublaj > Ses: **gemini**. Google AI Studio'dan alınan ücretsiz API anahtarını menüdeki alana yapıştırın.
Her satır Gemini TTS'ten, algılanan duyguya göre talimatla ("çok sinirli, bağırarak", "korkmuş, fısıltıyla")
ve karaktere özgü bir hazır sesle istenir. Aynı anda Edge sesi de hazırlanır: Gemini 1,2 sn içinde
gelmezse, kotası dolarsa ya da hata verirse **hazır Edge sesi beklemeden çalar**; o da yoksa Windows sesi.
Kota dolunca Gemini 10 dk dinlenir. Menü son satırı hangi motorun okuduğunu ve kaç kez Edge'e geçildiğini
gösterir. Ücretsiz kota ve Türkçe kalitesi Google'a bağlıdır; anahtar yoksa doğrudan Edge kullanılır.

### Sokak konuşmaları (Ctrl+Alt+N ya da menüdeki "Sokak" anahtarı)

Altyazısı olmayan NPC konuşmaları oyun sesinden yazıya çevrilir (faster-whisper, bilgisayarınızda,
işlemcide; ilk açılışta model bir kez indirilir), sonra çevrilip ekranın üstünde "Sokak: …" diye birkaç
saniye gösterilir. İsterseniz seslendirilir (menü > Çeviri > Sokak konuşmaları).

- Ekranda altyazı varken dinlenmez (o diyalog zaten çevriliyor), kendi dublajımız çalarken de.
- "Konuşma yok" olasılığı yüksek, düşük güvenli ya da müzik işaretli sonuçlar atılır; yine de araç
  radyosunu kısmak daha temiz sonuç verir.

### Sohbeti takip eden çeviri

Varsayılan olarak her satır yerel dil modeline, son 6 satırın İngilizcesi ve verdiğimiz Türkçesiyle
birlikte gider (menü > Çeviri). Model bunları sohbet geçmişi olarak görür: hitap (sen/siz), lakaplar,
"o / bu / onu" gibi göndermeler ve konu satırdan satıra tutarlı kalır. 6 satır yaklaşık son 20-30 sn
diyaloğu kapsar; daha fazlası 8 GB kartta yavaşlatır ve eski sahneyi karıştırır. 45 sn diyalog olmazsa
geçmiş sıfırlanır (yeni sahne). Hızlı taslak çeviri önce görünür, bağlamlı çeviri ~0,3-0,6 sn sonra
yerine geçer (dublaj açıkken ikisi birlikte çıkar). Ekran kartı daha çok çalışır; kapatınca dil modeli
yalnızca küfür/argo satırlarında kullanılır.

### Discord / OBS yayını

- Program bilgisayarın yükünü sürekli ölçer (menü > Durum > Motorlar > "Yük"). İşlemci 3 sn %80'i geçerse
  kendini kısar (ekranı daha seyrek yakalar, OCR'lar arasında kısa ara verir); %92'yi geçerse sokak
  dinlemesini de duraklatır. Rahatlayınca tam hıza döner. Altyazıya etkisi en fazla ~0,1-0,15 sn.
- Yayın gecesi menü > OCR ve hız > Performans: **yayin** seçip "Kaydet ve yeniden başlat": hep hafif
  çalışır; taslak çeviri 2 çekirdek, sokak dinleme hafif model (base.en) kullanır.
- Discord'da **Ayarlar > Gelişmiş > Donanım Hızlandırma** ve **Ses ve Görüntü > H.264 Donanım Hızlandırma**
  açık olsun: yayın görüntüsü ekran kartının ayrı kodlayıcısıyla (NVENC) sıkıştırılır, işlemci yorulmaz.
- Türkçe altyazının yayında görünmesi için **pencereyi değil ekranın tamamını** paylaşın (Türkçe yazı
  ayrı bir pencerede). Menüyü yayından önce Ctrl+Alt+M ile gizleyin ya da ikinci ekrana taşıyın.
- Sesli paylaşımda dublaj da arkadaşlarınıza gider.

### chiaki-ng ile

Program chiaki-ng'nin akış penceresini kendisi bulur (başlığı "chiaki" ile başlayan pencere; aynı
adı taşıyan tarayıcı sekmeleri seçilmez). Pencere 16:9 değilse görüntünün yanındaki/altındaki siyah
bantlar hesaba katılır; chiaki-ng'de görüntüyü pencereye **esneten** mod kullanıyorsanız `config.json`
içinde `"video_aspect": 0` yapın.

OCR doğruluğunu en çok akış kalitesi belirler (benzetim, 120 satır: 1080p'de karakter hatası %1,
720p'de %5, 540p'de %25). chiaki-ng ayarlarında mümkün olan en yüksek değerleri seçin:

- Çözünürlük **1080p** (PS5), kare hızı 60; bit hızını bağlantınızın kaldırdığı kadar yükseltin.
- Codec **H.265 (HEVC)** ve donanım çözücü (ekran kartı): aynı bit hızında daha net yazı.
- Mümkünse PS5'i kabloyla (Ethernet) bağlayın; Wi-Fi'de 5 GHz.

Akış yine de bulanık/bloklu kalıyorsa menü → **OCR ve hız → Akış kalitesi: Düşük** seçin
(daha soluk yazı da aranır ve emin olunmayan okumalar daha sık yeniden denenir: benzetimde 720p'de
tam doğru satır 55 -> 68/120, okuma başına ~+70 ms). Okuma
emin değilken program görüntünün keskinleştirilmiş ve kumlanması süzülmüş hâllerini de okuyup en
iyisini seçer (bu her zaman açık; iyi görüntüde ek iş yapılmaz).

### Oturum kaydı

Görülen her satır İngilizce + Türkçe olarak `kayitlar/` klasörüne (oturum başına bir dosya) yazılır:
kaçırdığınız sahneleri okuyabilir, kötü çevirileri toplu bildirebilirsiniz.

## Görev ve telefon yazıları

Altyazı dışında görev hedefleri ("Lose the cops") ve telefon mesajları da çevrilebilir.
Ctrl+Alt+G / Ctrl+Alt+T ile ilgili alanı bir kez seçin; her bölge ayrı okunur ve Türkçesi o alanın
üstünü kapatan kutuda gösterilir. Bölgeyi kaldırmak için seçiciyi açıp Delete'e basın.
Son satırlar panelinde (Ctrl+Alt+H) bu satırlar `[Görev]` / `[Telefon]` etiketiyle görünür.

## Yerel dil modeli (ücretsiz, sansürsüz son çeviri)

Taslağın yerine geçen son çeviri varsayılan olarak bilgisayarınızdaki bir dil modeliyle yapılır
(`final_translator: "ollama"`). Ücretsizdir, internet ve anahtar gerektirmez, küfür/argoyu
yumuşatmaz. NVIDIA ekran kartında çalışır (8 GB VRAM yeterli: RTX 4060 Ti 8 GB'ta model tamamen ekran kartında, satır başına ~0,3 sn).

1. Ollama'yı kurun: <https://ollama.com/download> (Windows, ücretsiz). Kurulunca arka planda çalışır.
2. Programı başlatın. Model (`ollama_model`, varsayılan `aya-expanse:8b`, ~5 GB) yüklü değilse
   indirilsin mi diye sorar; bir kerelik indirilir. İlk açılışta model ekran kartına yüklenir.
3. `.\GTA-Altyazi.exe --llm-bench` örnek repliklerin (küfürlüler dahil) çevirisini ve süresini gösterir.
   Başka bir model denemek için `config.json` → `ollama_model` değiştirip tekrar çalıştırın.

Varsayılan olarak (`ollama_only_profanity: true`) dil modeli yalnızca küfür/argo içeren satırlarda
ve sokak ağzı (y'all, ain't, callin', crib, loc...) içeren satırlarda kullanılır; diğer satırlarda
yerel taslak çeviri daha doğru ve hızlı olduğu için nihai sonuçtur.
12 GB+ ekran kartında daha iyi Türkçe için `"ollama_model": "gemma3:12b"` ve `"ollama_use_context": true`.

Ollama kapalıysa veya model yoksa program bunu söyler ve yalnızca yerel taslak çeviriyle devam eder.
DeepL'e dönmek için: `"final_translator": "deepl"` (ücretsiz plan; küfürleri kısmen yumuşatır).

## Sözlük ve sansürsüz çeviri

- **Sözlük (`sozluk.txt`):** her satır `İngilizce = Türkçe`. İsimler (Lucia, Vice City...) ve terimler
  DeepL çevirisinde hep aynı şekilde kullanılır. Dosya ilk çalıştırmada örnek isimlerle oluşturulur;
  değiştirip programı yeniden başlatın. Sözlük değişince eski çeviriler önbellekten kullanılmaz.
  Not: yerel taslak çeviri sözlüğü kullanmaz; DeepL çevirisi gelince sözlüklü hâli yerine geçer.
  Çok anlamlı tek kelimeleri (ör. "shit") sözlüğe eklemeyin; her bağlamda aynı karşılık kullanılır.
- **Sansürsüz (`uncensored: true`, varsayılan):** DeepL'e küfür ve argoyu aynı sertlikte çevirmesi,
  asla yumuşatmaması talimatı gönderilir (`deepl_instructions`). DeepL bu talimatı hızlı modelde kabul
  etmezse program otomatik olarak kalite modeline geçer; o da kabul etmezse talimatsız devam eder ve
  bunu konsola yazar. `--self-test` (anahtar girildiyse) talimatın kabul edilip edilmediğini ve örnek
  bir küfürlü satırın çevirisini gösterir. Yerel taslak model film/dizi altyazılarıyla eğitildiği için
  küfürleri olduğu gibi çevirme eğilimindedir. Program hiçbir metni filtrelemez.

## Ayar yapma (oyun çıkınca)

Remote Play'de bir altyazı sahnesinin ekran görüntüsünü alın (Win+Shift+S → kaydet) ve:

```
baslat.bat --ocr-test ekran.png          (config'deki bölgeyi okur)
baslat.bat --ocr-test ekran.png --full   (tüm resmi okur)
```

Okunan satırlar ve Türkçesi yazdırılır, OCR'a giren görüntü `ekran_ocr_girdi.png` olarak kaydedilir.
Canlı kullanımda süreleri görmek için `baslat.bat --debug`. Pencere başlığı farklıysa
`baslat.bat --list-windows` ile bakıp `window_titles` listesine ekleyin.

`config.json` yalnızca sizin değiştirdiğiniz ayarları içerir (diğerleri programın güncel varsayılanlarıdır). Bir ayarı değiştirmek için dosyaya `"ayar": değer` satırı ekleyin. Önemli ayarlar:

| Ayar | Varsayılan | Ne zaman değiştirilir |
|---|---|---|
| `translation_mode` | `"hybrid"` | `"final"` (yalnızca son çevirmen) veya `"local"` |
| `final_translator` | `"ollama"` | `"deepl"` |
| `ollama_model` | `"aya-expanse:8b"` | başka bir Ollama modeli (`--llm-bench` ile karşılaştırın) |
| `local_device` | `"auto"` | GPU sorun çıkarırsa `"cpu"` |
| `display_mode` | `"adaptive"` | `"cover"`: İngilizceyi koyu kutuyla kapatıp sabit renkte yazar |
| `deepl_model_type` | `"latency_optimized"` | Daha doğal ama daha yavaş çeviri için `"quality_optimized"` |
| `ocr_engine` | `"auto"` | RapidOCR (yerel, işlemcide; kontursuz/bulanık altyazıda Windows OCR'dan çok daha iyi). Kurulu değilse Windows OCR. Yalnızca Windows OCR için `"windows"`. İkisini kendi kayıtlarınızda `.\baslat.bat --ocr-compare` ile karşılaştırabilirsiniz |
| `ocr_device` | `"auto"` | RapidOCR ekran kartında (DirectML) çalışır, işlemci yerel taslak çeviriye kalır. Yerel dil modeli yavaşlarsa `"cpu"` |
| `preprocess` | `"auto"` | Yalnızca Windows OCR'da: kontur, parlak ve gri ton okumalarından en anlamlısını seçer. Tek yöntem için `"outline"`, `"bright"`, `"none"` (gri) ya da `"tophat"` |
| `tophat_radius` | `3` | Çok açık sahnelerde 2 deneyin |
| `outline_white_threshold` | `232` | Yazı eksik okunuyorsa düşürün; açık sahnede harfler kalınlaşıyorsa yükseltin |
| `bright_threshold` | `190` | Altyazı hiç okunmuyorsa düşürün (yazı bu parlaklığın altındaysa OCR atlanır) |
| `skip_without_bright_text` | `true` | Altyazı beyaz/parlak değilse `false` |
| `upscale` | `2.0` | Küçük yazıda 2.5–3; yavaşsa 1.5 |
| `capture_interval_ms` | `33` | CPU yükü fazlaysa 50–80 |
| `stable_reads` | `2` | Yanlış/yarım satırlar çevriliyorsa 3 |
| `fast_accept_confidence` | `0.93` | RapidOCR bu kadar eminse satır ikinci okuma beklenmeden çevrilir (~100-350 ms kazanç). Yanlış/yarım satırlar çevriliyorsa 0.97, her zaman doğrulamak için 1.0 |
| `confirm_hold_ms` | `40` | Geçiş karelerindeki bozuk satırlar hâlâ çevriliyorsa 60-80 (gecikme o kadar artar) |
| `clear_after_ms` | `700` | Türkçe altyazı çok erken/geç kayboluyorsa |
| `font_size`, `font_family`, `outline_px` | `26`, `Segoe UI`, `2` | Görünüm |

## Gecikme

Altyazının ekrana gelmesinden Türkçenin görünmesine kadar:

| Adım | Süre |
|---|---|
| Satırın fark edilmesi (33 ms aralık, ortalama yarısı) | ~0–33 ms |
| Windows OCR (yalnızca yazının kırpılmış kısmı) | ~15–50 ms |
| Doğrulama okuması (hemen yapılır, OCR genelde atlanır) | ~1–10 ms |
| Yerel taslak çeviri (RTX 4060 Ti; işlemcide ~20–60 ms) | ~10–30 ms |
| DeepL `latency_optimized` (taslağın yerine geçer) | ~100–300 ms (internete bağlı) |
| Önbellekteki satır | ~0 ms |

Sahte OCR (30 ms) ve sahte DeepL (150 ms) ile yapılan simülasyonda ortalama 197 ms ölçüldü:
DeepL dışındaki kısım ~47 ms. Kalan sürenin neredeyse tamamı DeepL'in ağ + çeviri süresidir;
bulut çeviri kullanıldıkça bunun altına inilemez. Bu yüzden karma modda ilk Türkçe metni yerel model
verir: ekrana gelme süresi tahminen ~60–80 ms'ye iner (GPU süresi ölçülmedi; `--self-test` ölçer).

`baslat.bat --debug` her satır için gerçek süreyi yazar (`ilk okumadan ekrana: … ms`).
Gecikmeyi düşürmek için yapılanlar:
- Görüntü değişmediyse OCR atlanır; bölgede parlak yazı yoksa OCR hiç çalışmaz.
- OCR'a bölgenin tamamı değil, yalnızca yazının bulunduğu kısım verilir.
- DeepL bağlantısı açık tutulur (sessiz anlarda ücretsiz kullanım sorgusu); uzun sessizlikten sonraki
  ilk satır yeniden bağlantı kurmayı beklemez. Hata olursa uzun yeniden denemeler yerine sonraki satıra geçilir.
- Çeviri kuyruğundaki eski satırlar atlanır; geç gelen eski çeviri yeni satırın üstüne yazılmaz.
- Tekrar eden satırlar `ceviri_onbellek.jsonl` önbelleğinden anında gelir ve kota harcamaz.
- Önceki satır DeepL'e bağlam olarak gönderilir (ücretlendirilmez); çeviri daha tutarlı olur.

## Notlar

- Altyazısı olmayan konuşmalar (ör. yoldan geçenler, radyo) çevrilmez.
- Overlay, tam ekran özel (exclusive fullscreen) uygulamaların üstünde görünmeyebilir; Remote Play
  bu modu kullanmaz ama sorun olursa pencere moduna alın.
- `"cover"` kutusunun ekran yakalamadan gizlenmesi Windows 10 (2004) veya Windows 11 gerektirir.
  Gizlenemezse başlangıçta uyarı yazılır.
- OBS'de "Ekran Yakalama" (Display Capture) kullanın; yalnızca Remote Play penceresini yakalayan
  "Pencere Yakalama" ayrı bir pencere olan Türkçe altyazıyı içermez.

## Geliştirme

```
python -m pip install pytest numpy pillow
python -m pytest tests
```

Testler Windows olmadan çalışan kısımları (metin temizleme, kararlılık, bölge hesabı,
değişim tespiti, renk seçimi, önbellek, ayarlar) kapsar. Ekran yakalama, Windows OCR ve overlay yalnızca
Windows'ta denenebilir.
