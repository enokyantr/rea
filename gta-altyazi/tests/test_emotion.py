from altyazi import emotion as e


def test_text_and_voice_decide_the_emotion():
    calm_voice, shouting, quiet = 0.0, e.arousal(-12, 160, -24, 120), e.arousal(-32, 110, -24, 120)
    assert shouting > 0.8 and quiet < -0.5
    assert e.classify("Get out of my car, you idiot!", shouting).label == e.ANGRY
    assert e.classify("Get out of my car, you idiot!", shouting).intensity > \
        e.classify("Get out of my car, you idiot!", calm_voice).intensity  # bağırınca daha yoğun
    assert e.classify("Hell yeah, we did it!", shouting).label == e.EXCITED
    assert e.classify("I'm sorry... he's gone.", quiet).label == e.SAD
    assert e.classify("Oh god, they're coming, run!", shouting).label == e.SCARED
    assert e.classify("The meeting is at noon.", calm_voice).label == e.NEUTRAL
    # Metinde işaret yokken yükselmiş ses: yine de tepki (heyecan)
    assert e.classify("Look at that over there.", shouting).label in (e.EXCITED, e.ANGRY)


def test_prosody_scales_with_intensity_and_character():
    angry = e.Emotion(e.ANGRY, 1.0)
    rate, pitch, volume = angry.prosody(1.0)
    assert rate > 0 and pitch > 0 and volume > 0
    assert angry.prosody(1.3)[2] > volume  # abartılı karakter daha yüksek bağırır
    assert e.Emotion(e.SAD, 1.0).prosody()[0] < 0  # üzgün: yavaş
    assert e.Emotion(e.NEUTRAL, 0.0).prosody() == (0, 0, 0)
