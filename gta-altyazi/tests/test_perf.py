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
