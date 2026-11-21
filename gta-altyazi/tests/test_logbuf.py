"""Menüdeki canlı kayıt: konsol çıktısı aynen geçer, satırlar tamponda tutulur."""

import io

from altyazi.logbuf import LogBuffer, _Tee


def test_tee_passes_output_and_keeps_complete_lines():
    out, buf = io.StringIO(), LogBuffer(max_lines=3)
    tee = _Tee(out, buf)
    tee.write("01:00 bir\n01:01 iki")
    assert out.getvalue() == "01:00 bir\n01:01 iki" and buf.text() == "01:00 bir"
    tee.write(" devam\n\nüç\ndört\n")
    assert buf.text().split("\n") == ["01:01 iki devam", "üç", "dört"]  # en eski düştü


def test_since_returns_only_new_lines():
    buf = LogBuffer(max_lines=5)
    for i in range(3):
        buf.add(str(i))
    seen, lines = buf.since(0)
    assert (seen, lines) == (3, ["0", "1", "2"])
    for i in range(3, 10):
        buf.add(str(i))
    seen, lines = buf.since(seen)
    assert seen == 10 and lines == ["5", "6", "7", "8", "9"]  # tampondan düşenler hariç
    assert buf.since(seen) == (10, [])
