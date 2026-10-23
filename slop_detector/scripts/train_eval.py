"""Обучение модели и оценка прототипа (стандартные + прикладные метрики).

Запуск:  python scripts/train_eval.py --data data/synthetic_corpus.jsonl
Результаты: artifacts/model.pkl, artifacts/metrics.json, артефакты примеров.

Прикладные критерии из ТЗ:
  - устойчивость к маскировке AI-признаков (recall на подмножестве ai_masked);
  - работа с частично сгенерированными письмами (recall на ai_partial);
  - выявление кампаний (precision/recall группировки против ground truth _campaign);
  - ложные срабатывания на легитимные письма (FPR на ham);
  - полнота объяснений (доля вердиктов с непустым списком suspicious_elements
    среди TP по AI-классу).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from slop_detector.core import SlopDetector, load_records
from slop_detector.parser import record_from_dict


def evaluate_grouping(det: SlopDetector, records, gt_campaigns):
    """Сравнение кампейн-группировки с ground truth (по парам писем)."""
    report = det.campaign_report(records)
    pred_pair = {}
    for c in report.campaigns:
        ids = sorted(c.msg_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pred_pair[(ids[i], ids[j])] = True
    true_pair = set()
    groups = defaultdict(list)
    for r, gt in zip(records, gt_campaigns):
        if gt:
            groups[gt].append(r.msg_id)
    for g in groups.values():
        g = sorted(g)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                true_pair.add((g[i], g[j]))
    tp = len(true_pair & set(pred_pair))
    fp = len(set(pred_pair) - true_pair)
    fn = len(true_pair - set(pred_pair))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pair_precision": round(prec, 3), "pair_recall": round(rec, 3),
            "pair_f1": round(f1, 3), "n_true_pairs": len(true_pair),
            "n_pred_pairs": len(pred_pair)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_corpus.jsonl")
    ap.add_argument("--out-model", default="artifacts/model.pkl")
    ap.add_argument("--out-metrics", default="artifacts/metrics.json")
    ap.add_argument("--cv", action="store_true", help="полный CV-отчёт (медленнее)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    records = [record_from_dict(r, i) for i, r in enumerate(rows)]
    y_spam = np.array([r["y_spam"] for r in rows])
    y_ai = np.array([r["y_ai"] for r in rows])
    origin = np.array([r["origin"] for r in rows])
    gt_camp = [r.get("_campaign") for r in rows]

    det = SlopDetector()
    X, names = det.featurize(records)
    # train/test split stratified by spam&ai combined
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(X))
    n_tr = int(0.75 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    det.fit([records[i] for i in tr], list(y_spam[tr]), list(y_ai[tr]))
    assert det.model is not None

    p_spam_te = np.array([det.model.predict({n: v for n, v in zip(names, X[i])})[0]
                          for i in te])
    p_ai_te = np.array([det.model.predict({n: v for n, v in zip(names, X[i])})[1]
                        for i in te])
    pred_spam = (p_spam_te >= 0.5).astype(int)
    pred_ai = (p_ai_te >= 0.5).astype(int)
    y_s_te, y_a_te, o_te = y_spam[te], y_ai[te], origin[te]

    metrics = {
        "spam": {
            "precision": round(precision_score(y_s_te, pred_spam, zero_division=0), 3),
            "recall": round(recall_score(y_s_te, pred_spam, zero_division=0), 3),
            "f1": round(f1_score(y_s_te, pred_spam, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_s_te, p_spam_te), 3),
            "confusion_matrix": confusion_matrix(y_s_te, pred_spam).tolist(),
        },
        "ai_assisted": {
            "precision": round(precision_score(y_a_te, pred_ai, zero_division=0), 3),
            "recall": round(recall_score(y_a_te, pred_ai, zero_division=0), 3),
            "f1": round(f1_score(y_a_te, pred_ai, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_a_te, p_ai_te), 3),
            "confusion_matrix": confusion_matrix(y_a_te, pred_ai).tolist(),
        },
    }

    # ---------- прикладные метрики ----------
    def subset_recall(mask, preds, ys):
        if mask.sum() == 0:
            return None
        return round(recall_score(ys[mask], preds[mask], zero_division=0), 3)

    m_masked = o_te == "ai_masked"
    m_partial = o_te == "ai_partial"
    m_ham = o_te == "ham"
    tn_ham = ((p_spam_te < 0.5) & m_ham).sum()
    fp_ham = ((p_spam_te >= 0.5) & m_ham).sum()
    applied = {
        "robustness_to_masking_ai_recall": subset_recall(m_masked, pred_ai, y_a_te),
        "partially_generated_ai_recall": subset_recall(m_partial, pred_ai, y_a_te),
        "legit_ham_false_positive_rate": round(float(fp_ham) / max(1, int(m_ham.sum())), 3),
        "legit_ham_spam_correct_rejection": round(
            float(tn_ham) / max(1, int(m_ham.sum())), 3),
    }

    # grouping eval on full corpus (детерминированно)
    grouping = evaluate_grouping(det, records, gt_camp)

    # explanation completeness: доля TP-AI вердиктов с объяснениями
    verdicts = det.analyze(records)
    tp_ids = {records[i].msg_id for pos, i in enumerate(te)
              if y_a_te[pos] == 1 and pred_ai[pos] == 1}
    with_expl = sum(1 for v in verdicts if v.msg_id in tp_ids and any(v.explanations.values()))
    applied["explanation_completeness_on_TP"] = round(
        with_expl / max(1, len(tp_ids)), 3)

    metrics["applied_criteria"] = applied
    metrics["campaign_grouping_vs_ground_truth"] = grouping
    metrics["n_train"], metrics["n_test"] = int(n_tr), int(len(te))
    metrics["model_cv"] = det.model.metrics if args.cv else "skipped (--cv to enable)"

    det.model.save(args.out_model)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # демо explainable-вывода на одном письме кампании
    demo_rows = [r for r in rows if r.get("_campaign")]
    if demo_rows:
        demo_recs = [record_from_dict(r, i) for i, r in enumerate(demo_rows[:6])]
        enrich = det.enrichment_json(demo_recs)
        Path("artifacts/sample_enrichment.json").write_text(
            json.dumps(enrich[0], ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n--- Sample enrichment (first campaign message) ---")
        print(json.dumps(enrich[0], ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
