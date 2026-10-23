"""Интеграционные тесты: полный пайплайн SlopDetector на синтетическом корпусе."""
import json
from pathlib import Path

import pytest

from slop_detector.core import SlopDetector, load_records
from slop_detector.explain.explainer import classify_origin
from slop_detector.parser import record_from_dict

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_corpus.jsonl"


@pytest.fixture(scope="module")
def corpus():
    assert DATA.exists(), f"нет датасета {DATA}"
    return load_records(DATA)


def test_corpus_labels_present(corpus):
    assert len(corpus) >= 50
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {"y_spam", "y_ai", "origin"} <= set(rows[0])


def test_classify_origin_branches():
    assert classify_origin(0.1, {}, {}) == "human"
    assert classify_origin(0.9, {"cliche_hits": 3, "typo_hits": 0}, {}) == "fully_generated"
    assert classify_origin(0.7, {"style_shift_ttr": 0.3}, {}) == "partially_edited"
    assert classify_origin(0.7, {}, {"ai_variation_flag": 1}) == "auto_variants"
    assert classify_origin(0.4, {}, {}) == "unclear"


def test_end_to_end_probabilities_in_range(corpus):
    det = SlopDetector()  # без модели — эвристики
    verdicts = det.analyze(corpus[:60])
    assert len(verdicts) == 60
    for v in verdicts:
        assert 0.0 <= v.spam_probability <= 1.0
        assert 0.0 <= v.ai_assisted_probability <= 1.0
        assert v.origin_type in {
            "human", "fully_generated", "partially_edited", "auto_variants", "unclear"}


def test_explanations_are_populated_for_suspicious(corpus):
    det = SlopDetector()
    spam_rows = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("y_spam") == 1 and row.get("y_ai") == 1:
            spam_rows.append(record_from_dict(row))
    verdicts = det.analyze(spam_rows[:20])
    n_with_expl = sum(1 for v in verdicts
                      if any(items for items in v.explanations.values()))
    assert n_with_expl >= 0.8 * len(verdicts), (
        "объяснения должны выдаваться для большинства AI-спама")


def test_campaign_grouping_on_corpus(corpus):
    """Письма одной синтетической кампании не должны рассыпаться по одиночкам."""
    camped = [r for r in corpus if r.extra.get("_campaign")]
    if len(camped) < 4:
        pytest.skip("в корпусе нет достаточной кампании")
    det = SlopDetector(sim_threshold=0.45)
    campaigns = det.analyzer.group(camped)
    true_ids = {r.msg_id: r.extra["_campaign"] for r in camped}
    # каждая кампания должна попадать в предсказанные группы размером >1
    grouped = {m for c in campaigns if len(c.msg_ids) > 1 for m in c.msg_ids}
    covered = sum(1 for mid, cid in true_ids.items() if mid in grouped)
    assert covered / len(true_ids) >= 0.7, (
        f"покрытие кампаний {covered}/{len(true_ids)} слишком низкое — "
        "проверить sim_threshold")


def test_featurize_dimension_consistency(corpus):
    X, names = SlopDetector.featurize(corpus[:10])
    assert X.shape == (10, len(names))
    assert len(names) > 20
