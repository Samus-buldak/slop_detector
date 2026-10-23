"""Сигнал 4: ссылки, домены, CTA и QR-коды.

Опора на наблюдения из статей: AI-assisted кампании активно используют
свежерегистрированные домены, URL-шортенеры, deep-link пути со случайными
сегментами, несоответствие отображаемого и реального href, QR-коды как
обход текстовых фильтров (QR → лендинг с LLM-сгенерированным контентом).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..schema import EmailRecord

CTA_WORDS = ["click here", "verify now", "secure your account", "claim", "act now",
             "открыть", "подтвердить", "получить", "перейти по ссылке", "активировать",
             "unlock", "get started", "view offer", "sign in", "log in", "reschedule"]


def url_signals(rec: EmailRecord) -> Tuple[Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    expl: List[str] = []
    urls = rec.urls
    n_urls = len(urls)
    f["url_count"] = float(n_urls)
    if n_urls == 0:
        f["qr_reference_flag"] = float(rec.has_qr_reference)
        if rec.has_qr_reference:
            expl.append("Упоминание QR-кода без текстовой ссылки — попытка обойти "
                        "анализ URL (AI-лендинги часто открываются через QR).")
        return f, expl

    f["suspicious_tld_ratio"] = sum(u.suspicious_tld for u in urls) / n_urls
    f["ip_literal_ratio"] = sum(u.is_ip_literal for u in urls) / n_urls
    f["shortener_ratio"] = sum(u.url_shortener for u in urls) / n_urls
    f["brand_mismatch_ratio"] = sum(u.has_brand_mismatch for u in urls) / n_urls
    f["homoglyph_ratio"] = sum(u.punycode_or_homoglyph for u in urls) / n_urls
    f["deep_path_ratio"] = sum(u.deep_path for u in urls) / n_urls
    f["digits_in_domain_ratio"] = sum(u.digits_in_domain for u in urls) / n_urls
    f["max_url_length"] = float(max((len(u.raw) for u in urls), default=0))
    # энтропия доменов: сгенерированные домены псевдослучайны
    ent = _domain_entropy([u.domain for u in urls])
    f["domain_entropy"] = ent

    bad = [u for u in urls if u.has_brand_mismatch]
    if bad:
        expl.append(f"Домен не совпадает с упомянутым брендом: {bad[0].raw[:60]}")
    if f["suspicious_tld_ratio"] >= 0.5:
        expl.append("Большинство ссылок ведут на подозрительные TLD (%s)."
                    % ", ".join(sorted({u.domain.rsplit('.', 1)[-1] for u in urls
                                         if u.suspicious_tld})))
    if f["shortener_ratio"] > 0:
        expl.append("Используются URL-шортенеры — сокрытие конечного адреса.")
    if f["ip_literal_ratio"] > 0:
        expl.append("Прямые IP-ссылки вместо домена.")
    if f["homoglyph_ratio"] > 0:
        expl.append("Punycode/гомоглифовые домены — имитация легитимных брендов.")
    f["qr_reference_flag"] = float(rec.has_qr_reference)
    if rec.has_qr_reference and n_urls <= 1:
        expl.append("QR-код + минимум ссылок в тексте — приём обхода фильтрации.")

    low = rec.full_text.lower()
    cta_hits = sum(1 for c in CTA_WORDS if c in low)
    f["cta_phrase_count"] = float(cta_hits)
    if cta_hits >= 2:
        expl.append("Агрессивные CTA-фразы (%d шт.) — характерно для шаблонных "
                    "промо/фишинговых кампаний." % cta_hits)
    return f, expl


def _domain_entropy(domains: List[str]) -> float:
    import math
    from collections import Counter
    chars = "".join(domains)
    if not chars:
        return 0.0
    c = Counter(chars)
    tot = len(chars)
    return -sum((v / tot) * math.log2(v / tot) for v in c.values()) / 4.5
