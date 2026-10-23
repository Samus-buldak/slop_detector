"""Сигнал 1: стилометрия и лингвистические признаки AI-ассистированного текста.

Признаки, отмеченные в литературе (IMC'25 "Do Spammers Dream of Electric Sheep?",
SpearBot, Phish-Master):
  - низкая лексическое разнообразие при высокой «гладкости» синтаксиса;
  - избыточное использование связок и клише LLM ("I hope this message finds you well",
    "Please do not hesitate to contact us", "It is important to note that");
  - аномально равномерная длина предложений (низкий коэффициент вариации);
  - высокая плотность пунктуации, em-dash, «вежливые» конструкции;
  - смешение регистров/языков, отсутствие опечаток (у людей они есть);
  - TTR (type-token ratio), hapax legomena ratio;
  - частотность слов (Zipf) — LLM-текст смещён к частым словам;
  - указание на частичную генерацию: резкие смены стиля между блоками.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Dict, List, Tuple

from ..schema import EmailRecord

SENT_SPLIT = re.compile(r"[.!?…]+[\s\"')\]]*|\n+")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']{1,}")

LLM_CLICHES_EN = [
    "i hope this message finds you well", "hope you are doing well",
    "do not hesitate to", "it is important to note", "in today's digital age",
    "i am writing to inform you", "please be advised", "your satisfaction is our priority",
    "at no additional cost", "don't miss out", "rest assured", "furthermore",
    "moreover", "additionally", "in conclusion", "it is worth noting",
    "look no further", "we are committed to", "streamline", "leverage",
    "seamless experience", "cutting-edge", "unlock", "elevate your",
]
LLM_CLICHES_RU = [
    "надеемся, что это письмо застанет вас", "не стесняйтесь обращаться",
    "важно отметить", "мы стремимся", "без дополнительных затрат",
    "цифровая эпоха", "передовые технологии", "индивидуальный подход",
    "высококвалифицированные специалисты", "оптимизировать процессы",
]
FORMAL_MARKERS = [
    "furthermore", "moreover", "additionally", "therefore", "consequently",
    "however", "nevertheless", "subsequently", "accordingly",
]
EXCLAM = re.compile(r"!{1,}")
EM_DASH = re.compile(r"\u2014| -- ")
CAPS_WORD = re.compile(r"\b[A-Z]{2,}(?![a-z])")
TYPO_PRONE_PATTERNS = re.compile(
    r"\b(teh|recieve|seperate|definately|occured|wich|adn|thier|purchse|verfy|acount)\b", re.I)


def _sentences(text: str) -> List[str]:
    s = [x.strip() for x in SENT_SPLIT.split(text) if x and len(x.strip()) > 1]
    return s


def _words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def text_stylometry_signals(rec: EmailRecord) -> Tuple[Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    expl: List[str] = []
    text = rec.text_body or ""
    subj = rec.subject or ""
    words = _words(text)
    n = len(words)
    f["body_word_count"] = float(n)
    f["subject_length"] = float(len(subj))
    f["subject_all_caps_ratio"] = (
        sum(w.isupper() for w in subj.split()) / max(1, len(subj.split())))

    if n < 3:
        return f, expl

    uniq = set(words)
    f["ttr"] = len(uniq) / n                                  # type-token ratio
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1)
    f["hapax_ratio"] = hapax / len(uniq) if uniq else 0.0

    # Zipf-подобная кривая: доля топ-10% самых частых слов в объёме текста
    counts = sorted(freq.values(), reverse=True)
    top_k = max(1, len(counts) // 10)
    f["zipf_top10_coverage"] = sum(counts[:top_k]) / n

    sents = _sentences(text)
    lens = [len(_words(s)) for s in sents] or [0]
    f["mean_sentence_len"] = statistics.fmean(lens)
    f["sentence_len_std"] = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    mean_ml = f["mean_sentence_len"] or 1.0
    f["sentence_len_cv"] = f["sentence_len_std"] / mean_ml     # coeff. variation
    f["long_sentences_ratio"] = sum(1 for l in lens if l >= 20) / len(lens)

    # равномерность длины предложений — признак шаблонной генерации
    if len(lens) >= 4 and f["sentence_len_cv"] < 0.28:
        expl.append("Предложения аномально одинаковой длины (CV=%.2f) — похоже на "
                    "шаблонную генерацию." % f["sentence_len_cv"])

    tokens = re.findall(r"\S+", text)
    f["punct_density"] = sum(1 for t in tokens if re.search(r"[.,;:!?\u2014]", t)) / len(tokens)
    f["comma_per_100w"] = text.count(",") / max(1, n) * 100
    f["semicolon_per_100w"] = text.count(";") / max(1, n) * 100
    f["exclamation_density"] = len(EXCLAM.findall(text)) / max(1, n) * 100
    f["em_dash_density"] = len(EM_DASH.findall(text)) / max(1, n) * 100
    f["caps_word_ratio"] = len(CAPS_WORD.findall(text)) / n
    f["digit_ratio"] = sum(1 for w in words if any(c.isdigit() for c in w)) / n

    low = text.lower()
    cliches_hit = [c for c in LLM_CLICHES_EN + LLM_CLICHES_RU if c in low]
    f["cliche_hits"] = float(len(cliches_hit))
    f["cliche_per_100w"] = len(cliches_hit) / max(1, n) * 100
    f["formal_marker_density"] = sum(low.count(m) for m in FORMAL_MARKERS) / max(1, n) * 100
    if cliches_hit:
        expl.append("Обнаружены клише, характерные для LLM-генерации: "
                    + ", ".join(f"'{c}'" for c in cliches_hit[:4]) + ".")

    f["typo_hits"] = float(len(TYPO_PRONE_PATTERNS.findall(text)))
    if n > 60 and f["typo_hits"] == 0:
        expl.append("Ни одной типичной человеческой опечатки на %d слов — нетипично "
                    "для ручной рассылки." % n)

    # смешение языков (латиница+кириллица в одном теле) — след сборки из разных источников
    lat = len(re.findall(r"[A-Za-z]", text))
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    f["script_mix_score"] = min(lat, cyr) / max(1, (lat + cyr))
    if 0.02 < f["script_mix_score"] < 0.5:
        expl.append("Смешение латиницы и кириллицы в тексте (%.1f%%) — вероятна "
                    "автоматическая сборка/перевод." % (f["script_mix_score"] * 100))

    # смена стиля между началом и концом письма — возможная частичная генерация
    if n > 120:
        half = len(text) // 2
        a, b = _words(text[:half]), _words(text[half:])
        ta = len(set(a)) / max(1, len(a))
        tb = len(set(b)) / max(1, len(b))
        f["style_shift_ttr"] = abs(ta - tb)
        if f["style_shift_ttr"] > 0.18:
            expl.append("Резкий скачок лексического стиля между первой и второй "
                        "половиной письма (ΔTTR=%.2f) — признак частичного "
                        "переписывания/склейки." % f["style_shift_ttr"])
    else:
        f["style_shift_ttr"] = 0.0

    return f, expl
