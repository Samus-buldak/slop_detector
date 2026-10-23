"""Сигнал 2: эвристики семейства AI-Liar / DetectGPT (без LLM в проде).

Идея (Markov et al., 2024; Mitchell et al., 2023): машинно-сгенерированный текст
отличается статистикой n-грамм и «предсказуемостью» продолжений. Полноценный
перплексий требует языковой модели; здесь — лёгкая прокси-реализация на основе
char/word n-gram entropy, которую можно заменить на локальную LM при наличии.

Дополнительно реализованы маркеры из статьи AI-Liar:
  - повторы фраз в пределах письма;
  - аномально «ровная» структура абзацев;
  - канцеляризмы и пассивные конструкции LLM.
Возвращает score в [0,1] + признаки + объяснения.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from ..schema import EmailRecord

PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(it is (widely )?known|important|noted|"
    r"recommended|advised|required|believed|assumed)", re.I)
WEASEL = ["it is important", "please note", "kindly", "should you require",
          "for your convenience", "we kindly ask", "your attention is required"]


def _ngram_entropy(words: List[str], k: int) -> float:
    if len(words) <= k:
        return 0.0
    grams = Counter(tuple(words[i:i + k]) for i in range(len(words) - k))
    total = sum(grams.values())
    ent = -sum((c / total) * math.log2(c / total) for c in grams.values())
    # нормировка на максимум лог2(total)
    return ent / max(1e-9, math.log2(total)) if total > 1 else 0.0


def _phrase_repeats(text: str) -> float:
    phrases = re.findall(r"\b(\w+ \w+ \w+)\b", text.lower())
    if not phrases:
        return 0.0
    c = Counter(phrases)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return rep / len(phrases)


def ai_liar_score(rec: EmailRecord) -> Tuple[float, Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    expl: List[str] = []
    text = rec.text_body or ""
    words = re.findall(r"[A-Za-zА-Яа-я']+", text.lower())
    n = len(words)
    if n < 20:
        return 0.5, {"ai_liar_ngram_entropy": 0.5}, []

    e2 = _ngram_entropy(words, 2)
    e3 = _ngram_entropy(words, 3)
    f["ai_liar_bigram_norm_entropy"] = e2
    f["ai_liar_trigram_norm_entropy"] = e3
    # у LLM-текста энтропия n-грамм ниже при том же объёме слов
    low_entropy = e3 < 0.62
    f["ai_liar_low_entropy_flag"] = float(low_entropy)

    rep = _phrase_repeats(text)
    f["phrase_repeat_ratio"] = rep
    if rep > 0.08:
        expl.append("Повторяющиеся трёхсловные фразы (%.1f%%) — шаблонная генерация "
                    "или автовариации." % (rep * 100))

    low = text.lower()
    weasel_hits = sum(1 for w in WEASEL if w in low)
    f["weasel_phrase_count"] = float(weasel_hits)
    f["passive_formal_count"] = float(len(PASSIVE_RE.findall(text)))
    if weasel_hits >= 2:
        expl.append("Канцелярские «LLM-оборонительные» обороты (%d шт.) — стиль "
                    "машинной вежливости." % weasel_hits)

    # баланс абзацев: LLM выдает ровные блоки
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 3:
        plens = [len(p.split()) for p in paras]
        mean_p = sum(plens) / len(plens) or 1
        cv = math.sqrt(sum((x - mean_p) ** 2 for x in plens) / len(plens)) / mean_p
        f["paragraph_len_cv"] = cv
        if cv < 0.35:
            expl.append("Абзацы почти одинакового размера (CV=%.2f) — автоматическая "
                        "компоновка." % cv)
    else:
        f["paragraph_len_cv"] = 1.0

    # сводный score: взвешенная комбинация (кализуется модельным слоем)
    s = 0.0
    s += 0.45 * (1.0 - min(1.0, e3 / 0.75))          # низкая trigram-энтропия
    s += 0.25 * min(1.0, rep / 0.12)                  # повторы
    s += 0.20 * min(1.0, weasel_hits / 3.0)           # канцеляризмы
    s += 0.10 * (1.0 if f["paragraph_len_cv"] < 0.35 else 0.0)
    score = max(0.0, min(1.0, s))
    f["ai_liar_score"] = score
    if score > 0.6:
        expl.append("AI-Liar-подобный скор %.2f: статистика текста близка к "
                    "машинно-генерированной." % score)
    return score, f, expl
