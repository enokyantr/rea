from altyazi.perf import decide


def test_throttles_up_fast_and_down_slowly():
    assert decide(0, [50, 60, 55], "auto") == 0
    assert decide(0, [85, 88, 90], "auto") == 1           # 3 sn %80+: hafif kısma
    assert decide(1, [95, 96, 97], "auto") == 2           # 3 sn %92+: yoğun
    assert decide(2, [95, 70, 60], "auto") == 2           # tek düşük ölçüm yetmez
    assert decide(2, [80, 75, 70, 72, 78, 74], "auto") == 1
    assert decide(1, [50, 55, 60, 40, 45, 50], "auto") == 0
    assert decide(0, [99, 99, 99], "normal") == 0         # normal: hiç kısmaz
    assert decide(0, [10, 10, 10, 10, 10, 10], "yayin") == 1  # yayın: en az hafif


from altyazi.config import Config
from altyazi.perf import GAME_LEVELS, LEVELS, LoadMonitor, game_policy


def test_game_mode_throttles_on_busy_gpu_or_low_vram():
    calm_cpu = [30] * 6
    assert decide(0, calm_cpu, "oyun") == 1                              # GPU ölçülemiyor: en az hafif
    assert decide(1, calm_cpu, "oyun", [95, 96, 97]) == 2                # 3 sn GPU %90+: yoğun
    assert decide(1, calm_cpu, "oyun", [95, 60, 97]) == 1                # tek dolu ölçüm yetmez
    assert decide(1, calm_cpu, "oyun", [40, 40, 40], vram_free_mb=300) == 2  # bellek bitiyor
    assert decide(2, calm_cpu, "oyun", [85, 70, 70, 70, 70, 70]) == 2   # GPU henüz rahatlamadı: düşmez
    assert decide(2, calm_cpu, "oyun", [70] * 6, vram_free_mb=2000) == 1
    assert decide(1, calm_cpu, "oyun", [99] * 6) == 2
    # Diğer modlarda GPU kısma kararına girmez (Remote Play'de GPU'yu dolduran bizim dil modelimiz).
    assert decide(0, calm_cpu, "auto", [99] * 6, vram_free_mb=100) == 0
    assert decide(0, calm_cpu, "yayin", [99] * 6, vram_free_mb=100) == 1


class FakeGpu:
    def __init__(self, samples):
        self.samples = list(samples)

    def sample(self):
        return self.samples.pop(0) if self.samples else None


def test_monitor_samples_gpu_and_uses_game_levels():
    cfg = Config(performance_mode="oyun")
    logs = []
    mon = LoadMonitor(cfg, logs.append, gpu=FakeGpu([(97.0, 7000.0, 8192.0)] * 3))
    assert mon.level == 1 and mon.capture_interval_s == GAME_LEVELS[1][0] / 1000
    for _ in range(3):
        mon.sample_once()
    assert mon.level == 2 and mon.min_ocr_gap_s == GAME_LEVELS[2][1] / 1000
    assert "ekran kartı %97" in logs[-1]
    text = mon.describe()
    assert "ekran kartı %97, bellek 6.8/8.0 GB" in text and "yoğun kısma" in text and "(oyun)" in text
    # GPU yok (nvidia-ml-py kurulu değil): eski davranış, GPU satırı yok.
    plain = LoadMonitor(Config(), logs.append, gpu=FakeGpu([]))
    plain.sample_once()
    assert plain.level == 0 and "ekran kartı" not in plain.describe()
    assert plain.capture_interval_s == LEVELS[0][0] / 1000


def test_game_policy_keeps_other_modes_unchanged():
    normal = game_policy(Config(), free_mb=7000)
    assert (normal.llm, normal.num_gpu, normal.ocr_device, normal.note) == ("gpu", None, "auto", "")
    low = game_policy(Config(), free_mb=1500)  # oyun açık ama mod "auto": yalnızca uyarı
    assert low.llm == "gpu" and low.num_gpu is None and "Performans: oyun" in low.note
    game = game_policy(Config(performance_mode="oyun"))
    assert (game.llm, game.num_gpu, game.ocr_device) == ("kapali", None, "cpu")
    on_cpu = game_policy(Config(performance_mode="oyun", game_llm="cpu", game_ocr_device="auto"))
    assert (on_cpu.llm, on_cpu.num_gpu, on_cpu.ocr_device) == ("cpu", 0, "auto")
    shared = game_policy(Config(performance_mode="oyun", game_llm="gpu"))
    assert shared.llm == "gpu" and shared.num_gpu is None
    deepl = game_policy(Config(performance_mode="oyun", final_translator="deepl"))
    assert deepl.llm == "gpu" and deepl.ocr_device == "cpu"  # DeepL ekran kartı kullanmaz
