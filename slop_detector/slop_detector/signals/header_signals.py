"""Сигнал 5: заголовки, отправитель, инфраструктура.

Признаки автоматизированной рассылки и AI-assisted кампаний из литературы:
  - X-Mailer/User-Agent, указывающие на скрипты/библиотеки (python, curl, bulk);
  - несовпаждение From-домена с Reply-To или display-name;
  - случайные локальные части адреса (12 символов, цифры);
  - отсутствие List-Unsubscribe у «маркетингового» письма;
  - аномальные Date / отсутствующие Received цепочки;
  - поддельный Message-ID (не совпадает с доменом).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..schema import EmailRecord

SCRIPT_MAILERS = ["python", "curl", "http", "bulk", "sendgrid-api", "mailgun",
                  "phpmailer", "manual send", "smtp", "node", "java"]


def header_signals(rec: EmailRecord) -> Tuple[Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    expl: List[str] = []
    h = rec.headers
    local = rec.from_addr.split("@")[0] if "@" in rec.from_addr else rec.from_addr
    f["local_part_len"] = float(len(local))
    f["local_digit_ratio"] = sum(c.isdigit() for c in local) / max(1, len(local))
    f["local_is_random"] = float(bool(re.fullmatch(r"[a-z0-9]{10,}", local.lower()))
                                 and f["local_digit_ratio"] > 0.3)
    if f["local_is_random"]:
        expl.append("Локальная часть адреса отправителя выглядит сгенерированной "
                    "(%s)." % local)

    xmailer = (h.get("x-mailer", "") + " " + h.get("user-agent", "")).lower()
    script_hit = any(s in xmailer for s in SCRIPT_MAILERS)
    f["script_mailer_flag"] = float(script_hit)
    if script_hit:
        expl.append("X-Mailer/User-Agent указывает на программную отправку: %s."
                    % xmailer.strip()[:60])

    reply_to = h.get("reply-to", "")
    rt_dom = reply_to.split("@")[-1].strip().lower() if "@" in reply_to else ""
    f["replyto_mismatch"] = float(bool(rt_dom) and rt_dom != rec.sender_domain)
    if f["replyto_mismatch"]:
        expl.append("Reply-To (%s) не совпадает с доменом отправителя (%s)."
                    % (rt_dom, rec.sender_domain))

    unsub = h.get("list-unsubscribe", "")
    f["has_list_unsubscribe"] = float(bool(unsub))
    marketingish = bool(re.search(r"unsubscribe|offer|deal|promo", rec.full_text.lower()))
    if marketingish and not unsub:
        expl.append("«Маркетинговый» текст без заголовка List-Unsubscribe — "
                    "несоответствие легитимным рассылкам.")

    f["auto_submitted"] = float(bool(h.get("auto-submitted", "")))
    f["precedence_bulk"] = float("bulk" in h.get("precedence", "").lower())

    # Message-ID vs домен отправителя
    mid = rec.msg_id
    m = re.search(r"@([\w.-]+)>?", mid or "")
    if m and rec.sender_domain and m.group(1).lower() != rec.sender_domain:
        f["msgid_domain_mismatch"] = 1.0
        expl.append("Message-ID содержит домен (%s), не совпадающий с доменом "
                    "отправителя (%s) — след автоподстановки."
                    % (m.group(1), rec.sender_domain))
    else:
        f["msgid_domain_mismatch"] = 0.0

    # to_addrs count as proxy of mass mailing (per-record)
    f["recipient_count"] = float(len(rec.to_addrs))
    return f, expl
