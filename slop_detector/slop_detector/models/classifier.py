"""Модельный слой: два классификатора поверх multi-signal признаков.

  spam_model   — вероятность спама/фишинга (двухклассовая);
  ai_model     — вероятность AI-assisted происхождения + тип происхождения
                 (fully_generated / partially_edited / auto_variants / human).

Реализация: GradientBoostingClassifier (scikit-learn) на табличных признаках.
Обучение — синтетический корпус (scripts/make_dataset.py), который моделирует
реалистичное распределение, включая маскировку AI-признаков и легитимные
корпоративные письма для контроля ложных срабатываний.

Модель сохраняет feature names; предсказание сопровождается per-feature
вкладом через difference-подобную аппроксимацию (fast permutation importance
для одного примера) — см. explain/explainer.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _fit(X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9,
        random_state=42)
    clf.fit(X, y)
    return clf


class SignalModel:
    """Обёртка над двумя моделями с отчётностью метрик."""

    def __init__(self):
        self.spam_model: Optional[GradientBoostingClassifier] = None
        self.ai_model: Optional[GradientBoostingClassifier] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, object] = {}

    # ---------- training ----------
    def fit(self, X: np.ndarray, y_spam: np.ndarray, y_ai: np.ndarray,
            feature_names: List[str], cv_report: bool = True):
        self.feature_names = list(feature_names)
        self.spam_model = _fit(X, y_spam)
        self.ai_model = _fit(X, y_ai)
        if cv_report:
            self.metrics = self._crossval_metrics(X, y_spam, y_ai)
        return self

    def _crossval_metrics(self, X, y_spam, y_ai) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for name, y in (("spam", y_spam), ("ai", y_ai)):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            est = _fit(X, y)
            proba = cross_val_predict(est, X, y, cv=skf, method="predict_proba")[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = {
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y, pred)),
            }
            try:
                m["roc_auc"] = float(roc_auc_score(y, proba))
            except ValueError:
                m["roc_auc"] = None
            out[name] = m
        return out

    # ---------- inference ----------
    def predict(self, feats: Dict[str, float]):
        x = self._vector(feats)
        p_spam = float(self.spam_model.predict_proba(x)[0, 1])
        p_ai = float(self.ai_model.predict_proba(x)[0, 1])
        return p_spam, p_ai

    def _vector(self, feats: Dict[str, float]) -> np.ndarray:
        return np.array([[float(feats.get(name, 0.0)) for name in self.feature_names]])

    # ---------- persistence ----------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"spam": self.spam_model, "ai": self.ai_model,
                         "names": self.feature_names, "metrics": self.metrics}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "SignalModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls()
        obj.spam_model, obj.ai_model = d["spam"], d["ai"]
        obj.feature_names = d["names"]
        obj.metrics = d.get("metrics", {})
        return obj
