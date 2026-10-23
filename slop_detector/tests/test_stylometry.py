"""Тесты стилометрических сигналов (signals/text_stylometry.py)."""
from slop_detector.schema import EmailRecord
from slop_detector.signals.text_stylometry import text_stylometry_signals


def _rec(body, subject="Test"):
    return EmailRecord(msg_id="t1", subject=subject, text_body=body)


def test_short_text_no_crash():
    f, e = text_stylometry_signals(_rec("hi"))
    assert f["body_word_count"] == 1.0
    assert "ttr" not in f  # слишком коротко — метрики не считаются


def test_ttr_bounds():
    body = " ".join(["word"] * 50)
    f, _ = text_stylometry_signals(_rec(body))
    assert 0 < f["ttr"] <= 1.0
    assert f["ttr"] == 1 / 50 + 0 or f["ttr"] < 0.1  # одно уникальное слово на 50


def test_cliche_detection_en():
    body = ("I hope this message finds you well. Furthermore, it is important to note "
            "that our team is committed to delivering a seamless experience for you today.")
    f, expl = text_stylometry_signals(_rec(body))
    assert f["cliche_hits"] >= 3
    assert any("клише" in x.lower() or "LLM" in x for x in expl)


def test_uniform_sentences_flagged():
    s = "The system processes the request quickly and efficiently."
    body = " ".join([s] * 6)
    f, expl = text_stylometry_signals(_rec(body))
    assert f["sentence_len_cv"] < 0.28
    assert any("одинаковой длины" in x for x in expl)


def test_typo_free_long_text_explained():
    words = ["alpha", "bravo", "charlie", "delta", "echo"] * 20
    body = ", ".join(words) + "."
    f, expl = text_stylometry_signals(_rec(body))
    assert f["typo_hits"] == 0
    assert any("опечатк" in x for x in expl)


def test_script_mix_detected():
    body = ("Please review the документ attached ниже и подтвердите receipt of your "
            "запрос on the deadline urgently today please")
    f, expl = text_stylometry_signals(_rec(body))
    assert 0.02 < f["script_mix_score"] < 0.5
    assert any("Смешение" in x for x in expl)


def test_typos_counted():
    body = "Please teh verfy your acount immediatly before the definately deadline passes us"
    f, _ = text_stylometry_signals(_rec(body))
    assert f["typo_hits"] >= 4
