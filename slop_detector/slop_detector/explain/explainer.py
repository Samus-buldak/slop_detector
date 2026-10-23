"""Explainable AI-assisted spam enrichment: формирование вывода для аналитика.

Выдаёт не бинарный класс, а структурированное объяснение:
  - вердикты: spam_probability, ai_assisted_probability;
  - происхождение: fully_generated / partially_edited / auto_variants / human / unclear;
  - «подозрительные элементы» письма (текст, HTML, ссылки, заголовки, QR) с причинами;
  - вклад сигнальных групп в решение (аппроксимация через feature importance модели
    и локальный signs-of-contribution по значениям признаков);
  - уверенность (confidence) на основе расстояния вероятности от порога и согласия
    между слоями сигналов.
"""
from __future__ import annotations

from typing import Dict, List

GROUP_LABELS = {
    "text": "Текст/стилометрия",
    "ai_liar": "Статистика генеративности (AI-Liar)",
    "html": "HTML-структура",
    "url": "Ссылки/домены/CTA/QR",
    "header": "Заголовки/инфраструктура",
}

# какие признаки какой группе принадлежат (префиксы имён)
FEATURE_GROUP = {
    "ttr": "text", "hapax": "text", "zipf": "text", "sentence": "text",
    "cliche": "text", "formal_marker": "text", "typo": "text", "script_mix": "text",
    "style_shift": "text", "punct_density": "text", "comma": "text", "semicolon": "text",
    "exclamation": "text", "em_dash": "text", "caps_word": "text", "digit_ratio": "text",
    "subject": "text", "body_word": "text", "long_sentences": "text",
    "ai_liar": "ai_liar", "phrase_repeat": "ai_liar", "weasel": "ai_liar",
    "passive_formal": "ai_liar", "paragraph_len": "ai_liar",
    "html": "html", "inline_style": "html", "table": "html", "td": "html",
    "tr_empty": "html", "div": "html", "nesting": "html", "base64": "html",
    "script_count": "html", "iframe": "html", "comment": "html",
    "template_comment": "html", "indent_uniformity": "html", "hidden_text": "html",
    "url": "url", "suspicious_tld": "url", "ip_literal": "url", "shortener": "url",
    "brand_mismatch": "url", "homoglyph": "url", "deep_path": "url",
    "digits_in_domain": "url", "max_url": "url", "domain_entropy": "url",
    "qr_reference": "url", "cta": "url",
    "local_": "header", "replyto": "header", "list_unsubscribe": "header",
    "has_list": "header", "auto_submitted": "header", "precedence": "header",
    "msgid": "header", "recipient": "header", "script_mailer": "header",
}


def group_of(feature: str) -> str:
    for pref, g in FEATURE_GROUP.items():
        if feature.startswith(pref):
            return g
    return "text"


def classify_origin(p_ai: float, feats: Dict[str, float],
                    campaign_feats: Dict[str, float]) -> str:
    """Эвристика типа AI-assisted происхождения."""
    high_cliche = feats.get("cliche_hits", 0) >= 2 or feats.get("ai_liar_score", 0) > 0.55
    style_shift = feats.get("style_shift_ttr", 0) > 0.18
    mixed_scripts = feats.get("script_mix_score", 0) > 0.02
    has_typo = feats.get("typo_hits", 0) > 0
    auto_variants = campaign_feats.get("ai_variation_flag", 0) > 0

    if p_ai < 0.35:
        return "human"
    if auto_variants and p_ai >= 0.5:
        return "auto_variants"
    if style_shift or mixed_scripts or (high_cliche and has_typo):
        return "partially_edited"
    if high_cliche and not has_typo:
        return "fully_generated"
    return "unclear"


ORIGIN_DESCRIPTIONS = {
    "fully_generated": "Письмо целиком похоже на машинно-сгенерированное.",
    "partially_edited": "Следы частичного редактирования: склейка стилей/языков при "
                        "общей LLM-базе текста.",
    "auto_variants": "Сообщение — вариация в рамках автоматически масштабированной "
                     "кампании (парафраз шаблона).",
    "human": "Явных признаков AI-ассистирования не обнаружено.",
    "unclear": "Признаки смешаны; требуется ручная проверка.",
}


def build_explanation(rec_id: str, p_spam: float, p_ai: float, origin: str,
                      signal_explanations: Dict[str, List[str]],
                      top_features: List[Dict[str, object]],
                      campaign_info: Dict[str, object]) -> Dict[str, object]:
    # confidence: насколько уверенно (дальше от 0.5 + согласие слоёв)
    margin = abs(p_ai - 0.5) + abs(p_spam - 0.5)
    agreement = 1.0 if (p_spam > 0.5) == (p_ai > 0.4) else 0.55
    confidence = round(min(0.99, 0.5 + margin * 0.6 * agreement), 3)

    suspicious_elements = []
    for layer, items in signal_explanations.items():
        for it in items:
            suspicious_elements.append({"layer": GROUP_LABELS.get(layer, layer),
                                        "observation": it})

    verdict_words = []
    if p_spam >= 0.7:
        verdict_words.append("высокая вероятность спама/фишинга")
    elif p_spam >= 0.4:
        verdict_words.append("умеренная вероятность нежелательной рассылки")
    else:
        verdict_words.append("низкая вероятность спама")
    if p_ai >= 0.6:
        verdict_words.append("вероятно использовался генеративный ИИ (%s)" % origin)
    elif p_ai >= 0.4:
        verdict_words.append("возможны отдельные признаки AI-ассистирования")
    else:
        verdict_words.append("признаков AI-генерации не выявлено")

    return {
        "msg_id": rec_id,
        "spam_probability": round(p_spam, 3),
        "ai_assisted_probability": round(p_ai, 3),
        "origin_type": origin,
        "origin_description": ORIGIN_DESCRIPTIONS[origin],
        "confidence": confidence,
        "verdict_summary": "; ".join(verdict_words).capitalize() + ".",
        "suspicious_elements": suspicious_elements,
        "top_contributing_signals": top_features,
        "campaign": campaign_info,
    }
