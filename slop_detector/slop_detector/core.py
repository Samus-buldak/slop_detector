"""Ядро модуля: оркестрация сигналов, моделей и кампейн-анализа.

Использование:
    det = SlopDetector(model_path="artifacts/model.pkl")   # или det.fit(...)
    verdicts = det.analyze(records)                        # список MessageVerdict
    report   = det.campaign_report(records)                # CampaignReport
    json_out = det.enrichment_json(records)                # для SOC-интеграции
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .schema import EmailRecord
from .parser import parse_raw, record_from_dict
from .signals import (text_stylometry_signals, ai_liar_score,
                      html_structure_signals, url_signals, header_signals)
from .campaign.analyzer import CampaignAnalyzer, Campaign
from .models.classifier import SignalModel
from .explain.explainer import (build_explanation, classify_origin, group_of,
                                ORIGIN_DESCRIPTIONS)


@dataclass
class MessageVerdict:
    msg_id: str
    spam_probability: float
    ai_assisted_probability: float
    origin_type: str
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    explanations: Dict[str, List[str]] = field(default_factory=dict)
    campaign_id: Optional[str] = None


@dataclass
class CampaignReport:
    campaigns: List[Campaign]
    message_to_campaign: Dict[str, str]


def extract_all_features(rec: EmailRecord) -> Dict[str, float]:
    """Собирает признаки всех слоёв в один словарь."""
    feats: Dict[str, float] = {}
    f1, _ = text_stylometry_signals(rec)
    _, f2, _ = ai_liar_score(rec)
    f3, _ = html_structure_signals(rec)
    f4, _ = url_signals(rec)
    f5, _ = header_signals(rec)
    for d in (f1, f2, f3, f4, f5):
        feats.update(d)
    return feats


def extract_all_explanations(rec: EmailRecord) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    _, e1 = text_stylometry_signals(rec)
    _, _, e2 = ai_liar_score(rec)
    _, e3 = html_structure_signals(rec)
    _, e4 = url_signals(rec)
    _, e5 = header_signals(rec)
    if e1: out["text"] = e1
    if e2: out["ai_liar"] = e2
    if e3: out["html"] = e3
    if e4: out["url"] = e4
    if e5: out["header"] = e5
    return out


class SlopDetector:
    def __init__(self, model: Optional[SignalModel] = None,
                 sim_threshold: float = 0.45):
        self.model = model
        self.analyzer = CampaignAnalyzer(sim_threshold=sim_threshold)

    # ---------- fit ----------
    @staticmethod
    def featurize(records: List[EmailRecord]) -> tuple:
        rows = [extract_all_features(r) for r in records]
        names = sorted({k for row in rows for k in row})
        X = np.array([[row.get(n, 0.0) for n in names] for row in rows], dtype=float)
        return X, names

    def fit(self, records: List[EmailRecord], y_spam: List[int], y_ai: List[int]):
        X, names = self.featurize(records)
        self.model = SignalModel().fit(X, np.array(y_spam), np.array(y_ai), names)
        return self

    # ---------- analyze ----------
    def analyze(self, records: List[EmailRecord]) -> List[MessageVerdict]:
        campaigns = self.analyzer.group(records)
        m2c: Dict[str, Campaign] = {}
        for c in campaigns:
            for m in c.msg_ids:
                m2c[m] = c
        verdicts: List[MessageVerdict] = []
        for rec in records:
            feats = extract_all_features(rec)
            camp = m2c.get(rec.msg_id)
            camp_feats = camp.features if camp else {}
            p_spam, p_ai = self._predict(feats, camp_feats)
            origin = classify_origin(p_ai, feats, camp_feats)
            verdicts.append(MessageVerdict(
                msg_id=rec.msg_id,
                spam_probability=round(p_spam, 3),
                ai_assisted_probability=round(p_ai, 3),
                origin_type=origin,
                confidence=self._confidence(p_spam, p_ai, feats),
                features=feats,
                explanations=extract_all_explanations(rec),
                campaign_id=camp.campaign_id if camp else None,
            ))
        return verdicts

    def _predict(self, feats: Dict[str, float],
                 camp_feats: Dict[str, float]) -> tuple:
        if self.model is None:
            # fallback без модели: эвристические вероятности
            p_spam = min(0.98, max(0.02, self._heuristic_spam(feats)))
            p_ai = min(0.98, max(0.02, self._heuristic_ai(feats, camp_feats)))
            return p_spam, p_ai
        p_spam, p_ai = self.model.predict(feats)
        # кампейн-бустер: вариативность кампании повышает AI-вероятность
        if camp_feats.get("ai_variation_flag", 0) > 0:
            p_ai = min(0.99, p_ai + 0.15)
        return p_spam, p_ai

    @staticmethod
    def _heuristic_spam(f: Dict[str, float]) -> float:
        s = 0.0
        s += 0.30 * f.get("brand_mismatch_ratio", 0)
        s += 0.15 * f.get("suspicious_tld_ratio", 0)
        s += 0.10 * f.get("shortener_ratio", 0)
        s += 0.10 * f.get("ip_literal_ratio", 0)
        s += 0.10 * min(1.0, f.get("cta_phrase_count", 0) / 3)
        s += 0.10 * f.get("hidden_text_markers", 0) / 3
        s += 0.05 * f.get("qr_reference_flag", 0)
        s += 0.05 * f.get("script_mailer_flag", 0)
        s += 0.05 * f.get("local_is_random", 0)
        return s

    @staticmethod
    def _heuristic_ai(f: Dict[str, float], cf: Dict[str, float]) -> float:
        s = 0.0
        s += 0.30 * f.get("ai_liar_score", 0)
        s += 0.15 * min(1.0, f.get("cliche_hits", 0) / 3)
        s += 0.12 * (1.0 if f.get("typo_hits", 1) == 0 and f.get("body_word_count", 0) > 60 else 0)
        s += 0.10 * (1.0 if 0 < f.get("sentence_len_cv", 1) < 0.28 else 0)
        s += 0.10 * min(1.0, f.get("phrase_repeat_ratio", 0) / 0.12)
        s += 0.08 * (1.0 if f.get("indent_uniformity", 0) > 0.7 and f.get("has_html", 0) else 0)
        s += 0.15 * cf.get("ai_variation_flag", 0)
        return s

    @staticmethod
    def _confidence(p_spam: float, p_ai: float, f: Dict[str, float]) -> float:
        margin = abs(p_ai - 0.5) + abs(p_spam - 0.5)
        layers = sum(1 for k in ("ai_liar_score", "cliche_hits", "html_tag_density",
                                 "url_count", "local_is_random") if f.get(k, 0))
        return round(min(0.99, 0.45 + margin * 0.5 + layers * 0.05), 3)

    # ---------- explainable output ----------
    def enrichment_json(self, records: List[EmailRecord], top_k: int = 6) -> List[Dict]:
        """Полный объяснимый вывод для SOC/антиспам-системы."""
        campaigns = self.analyzer.group(records)
        m2c = {m: c for c in campaigns for m in c.msg_ids}
        verdicts = self.analyze(records)
        importance = getattr(self.model, "spam_model", None)
        global_imp = None
        if self.model and self.model.spam_model is not None:
            gimp = dict(zip(self.model.feature_names,
                            self.model.spam_model.feature_importances_))
            global_imp = gimp
        out = []
        for v in verdicts:
            top_feats = self._top_contributions(v, global_imp, top_k)
            camp = m2c.get(v.msg_id)
            camp_info = {
                "campaign_id": camp.campaign_id if camp else None,
                "size": len(camp.msg_ids) if camp else 1,
                "features": {k: round(x, 3) for k, x in (camp.features.items()
                             if camp else [])},
                "explanations": camp.explanations if camp else [],
            }
            out.append(build_explanation(
                v.msg_id, v.spam_probability, v.ai_assisted_probability,
                v.origin_type, v.explanations, top_feats, camp_info))
        return out

    @staticmethod
    def _top_contributions(v: MessageVerdict, global_imp: Optional[Dict[str, float]],
                           top_k: int) -> List[Dict[str, object]]:
        """Локальный вклад: |значение| × глобальная важность признака."""
        scored = []
        for name, val in v.features.items():
            w = (global_imp or {}).get(name, 0.0)
            local = abs(float(val)) * (0.5 + w)
            if local > 0:
                scored.append((local, name, float(val), w))
        scored.sort(reverse=True)
        res = []
        for local, name, val, w in scored[:top_k]:
            res.append({"feature": name, "value": round(val, 4),
                        "group": group_of(name), "model_weight": round(w, 4)})
        return res

    # ---------- convenience ----------
    def campaign_report(self, records: List[EmailRecord]) -> CampaignReport:
        campaigns = self.analyzer.group(records)
        m2c = {m: c.campaign_id for c in campaigns for m in c.msg_ids}
        return CampaignReport(campaigns=campaigns, message_to_campaign=m2c)


def load_records(path: str | Path) -> List[EmailRecord]:
    """Загрузка корпуса из JSONL (строки датасета)."""
    import json
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                recs.append(record_from_dict(json.loads(line), idx=i))
    return recs
