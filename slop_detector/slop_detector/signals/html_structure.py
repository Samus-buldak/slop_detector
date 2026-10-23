"""Сигнал 3: HTML/структурные признаки письма.

AI-assisted спам часто генерирует однотипный HTML (шаблоны, сгенерированные LLM
или конструкторами): одинаковая вложенность, table-based layout без семантики,
инлайн-стили вместо CSS, «бесконечные» пустые ячейки, base64-изображения,
минификация/автоформатирование. Признаки структуры — важный слой multi-signal
анализа, т.к. текст может быть переписан человеком, а разметка — нет.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from ..schema import EmailRecord


def _max_depth(html: str) -> int:
    depth = cur = 0
    for tag in re.findall(r"<(/?)(\w+)[^>]*>", html):
        closing, name = tag
        if name.lower() in ("br", "img", "hr", "meta", "link", "input"):
            continue
        if closing:
            cur = max(0, cur - 1)
        else:
            cur += 1
            depth = max(depth, cur)
    return depth


def html_structure_signals(rec: EmailRecord) -> Tuple[Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    expl: List[str] = []
    html = rec.html_body or ""
    f["has_html"] = float(bool(html))
    if not html:
        return f, expl

    low = html.lower()
    total_len = len(low) or 1
    text_only = re.sub(r"<[^>]+>", "", low)
    f["html_tag_density"] = len(re.findall(r"<\w+", low)) / max(1, len(text_only.split()))
    f["inline_style_ratio"] = low.count("style=") / max(1, len(re.findall(r"<\w+", low)))
    f["table_count"] = float(low.count("<table"))
    f["td_count"] = float(low.count("<td"))
    f["tr_emptyish_ratio"] = (
        low.count("<td></td>") + low.count("<td>&nbsp;</td>") +
        low.count("<td > </td>") ) / max(1, f["td_count"])
    f["div_count"] = float(low.count("<div"))
    f["max_nesting_depth"] = float(_max_depth(html))
    f["base64_img_count"] = float(len(re.findall(r'src=["\']data:image', low)))
    f["script_count"] = float(low.count("<script"))
    f["iframe_count"] = float(low.count("<iframe"))
    f["comment_count"] = float(len(re.findall(r"<!--", html)))
    # автогенерированные комментарии шаблонов
    template_comments = len(re.findall(
        r"<!--\s*(?:if mso|end if|BEGIN|END|generated|template|module)", html, re.I))
    f["template_comment_count"] = float(template_comments)

    # whitespace-подпись: LLM-генераторы выдают характерное форматирование
    lines = [l for l in html.splitlines() if l.strip()]
    if lines:
        indents = [len(l) - len(l.lstrip()) for l in lines]
        f["indent_uniformity"] = 1.0 - (Counter(indents).most_common(1)[0][1] / len(lines))
    else:
        f["indent_uniformity"] = 0.0

    # скрытый текст (white-on-white) — частый приём фишинга
    hidden = len(re.findall(r"color\s*:\s*(#fff|white)\b[^>]*background(?:-color)?\s*:\s*\1",
                            low))
    tiny = len(re.findall(r"font-size\s*:\s*[0-5]px", low)) + \
        len(re.findall(r"display\s*:\s*none", low))
    f["hidden_text_markers"] = float(hidden + tiny)
    if hidden + tiny >= 2:
        expl.append("В HTML есть скрытый/невидимый текст (%d маркеров) — типично для "
                    "фишинговых шаблонов." % (hidden + tiny))

    if f["base64_img_count"] >= 1 and f["script_count"] == 0 and f["table_count"] >= 2:
        expl.append("Изображения встроены как base64 внутри табличного шаблона — "
                    "признак автоматической сборки письма.")
    if f["html_tag_density"] > 1.8:
        expl.append("Аномально плотная HTML-разметка на слово текста — письмо собрано "
                    "генератором, а не сверстано человеком.")
    if template_comments >= 2:
        expl.append("Комментарии автогенерируемого шаблона в HTML (%d шт.)."
                    % template_comments)
    return f, expl
