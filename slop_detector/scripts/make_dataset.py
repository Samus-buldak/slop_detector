"""Генератор синтетического корпуса для обучения и оценки прототипа.

Почему синтетика: публичные датасеты AI-ассистированного спама практически
отсутствуют (см. IMC'25 — авторы собирали корпус вручную). Синтез позволяет
контролируемо покрыть все целевые категории происхождения:

  ham            — легитимные корпоративные письма, маркетинговые рассылки,
                   автоматические уведомления (контроль ложных срабатываний);
  spam_human     — классический спам, написанный человеком (с опечатками,
                   неровным стилем);
  ai_full        — полностью LLM-сгенерированный текст + автогенерированный HTML;
  ai_partial     — LLM-база + ручная вставка/правка (склейка стилей, опечатки);
  ai_variants    — вариации одного шаблона внутри кампании (парафраз слотов);
  ai_masked      — AI-текст после «anti-detector» постобработки (добавлены
                   опечатки, варьирована длина предложений) — проверка
                   устойчивости к маскировке.

Каждая запись содержит метки y_spam и y_ai и поле origin.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

random.seed(1337)

# ------------------------- языковые пулы -------------------------
LLM_PHRASES = [
    "I hope this message finds you well.",
    "We are committed to providing a seamless experience for all valued customers.",
    "It is important to note that your account requires immediate attention.",
    "Please do not hesitate to contact our support team for further assistance.",
    "Furthermore, we kindly ask you to verify your information at your earliest convenience.",
    "Your satisfaction is our priority and we strive to exceed expectations.",
    "In today's digital age, protecting your personal information is essential.",
    "We appreciate your continued partnership and look forward to serving you.",
    "Additionally, rest assured that all transactions are secured with encryption.",
    "Should you require any clarification, our dedicated team remains available.",
]
LLM_PHRASES_RU = [
    "Надеемся, что это письмо застанет вас в добром здравии.",
    "Мы стремимся обеспечивать бесшовный сервис для всех наших клиентов.",
    "Важно отметить, что ваша учётная запись требует внимания.",
    "Не стесняйтесь обращаться в нашу службу поддержки.",
    "Кроме того, пожалуйста, подтвердите свои данные в удобное время.",
]
HUMAN_PHRASES = [
    "hey, quick question about the invoice from last week",
    "can u send me the updated file when you get a chance?",
    "sorry for the delay, been swamped with the migration stuff",
    "let me know if the attached draft works or if i should redo it",
    "meeting moved to thursday btw, cal looks crazy this week",
    "did anyone else notice the staging server acting weird again?",
    "thanks!! really appreciate you covering for me yesterday",
    "hmm not sure thats in scope, ill check with the team first",
]
SPAM_HUMAN_PHRASES = [
    "MAKE MONEY FAST!!! work from home and earn 5000 usd weekly!!!",
    "you have WON a prize, claim now before its too late dont miss out",
    "buy cheap meds online no prescription needed discreet shipping",
    "hot singles in your area want to meet tonight click here now",
    "increase ur website traffic 1000% guaranteed seo service low price",
]
CTA_LINES = [
    "Click here to secure your account: {url}",
    "Verify now to avoid suspension: {url}",
    "Claim your reward today: {url}",
    "Отсканируйте QR-код и подтвердите оплату: {url}",
    "View the limited offer: {url}",
]
HAM_SUBJECTS = [
    "Re: Q3 budget review follow-up", "Lunch tomorrow?", "PR #482 ready for review",
    "Weekly standup notes", "Contract amendment for signature",
    "Your monthly statement is ready", "Order shipped: #{oid}",
    "Reminder: team offsite Friday", "FW: vendor invoice INV-{oid}",
]
AI_CAMPAIGN_TEMPLATES = [
    ("Dear {name}, we noticed unusual sign-in activity on your {svc} account. "
     "To keep your information safe, please confirm your identity using our "
     "secure portal. Furthermore, our team has temporarily restricted access. "
     "Please do not hesitate to contact support if this was you."),
    ("Hello {name}, your {svc} subscription renewal could not be processed. "
     "We kindly ask you to update payment details at your earliest convenience. "
     "Rest assured your data remains protected. It is important to note that "
     "failure to act may result in service interruption."),
    ("Greetings {name}, an important document awaits your review in {svc}. "
     "We are committed to delivering a seamless experience, so simply open the "
     "portal and follow the guided steps. Additionally, verification takes less "
     "than two minutes."),
]
SERVICES = ["PayPal", "Microsoft", "Google", "Netflix", "Amazon", "bank", "Dropbox"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "user", "client"]
BAD_TLDS = ["xyz", "top", "click", "shop", "icu", "cyou", "sbs", "zip"]
GOOD_DOMAINS = ["newsletter-brand.com", "company-hr.io", "docs-service.net"]


def rand_local(rng: random.Random, kind: str) -> str:
    if kind == "random":
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                       for _ in range(rng.randint(10, 14)))
    return rng.choice(["no-reply", "support", "info", "billing", "hr.team",
                       "notifications", "a.kovalenko", "j.smith"])


def make_html(rng: random.Random, generated: bool, body_urls: List[str]) -> str:
    if not generated:
        rows = "\n".join(f"<p>{u}</p>" for u in body_urls)
        return f"<html><body>{rows}</body></html>"
    imgs = 'src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==="'
    hidden = '<span style="color:#ffffff;background-color:#ffffff">ref=88</span>'
    cells = "\n".join("<td></td>" for _ in range(rng.randint(2, 4)))
    return (f'<html><head><style>.t{{font-family:Arial}}</style></head><body>\n'
            f'<!-- BEGIN template module generated -->\n<table cellpadding="0" cellspacing="0">'
            f'<tr><td style="padding:10px"><table><tr>{cells}</tr>'
            f'<tr><td><a href="{body_urls[0] if body_urls else "#"}" '
            f'style="background:#0b6;color:#fff">Confirm</a></td></tr>'
            f'<tr><td><img {imgs}></td></tr><tr><td>{hidden}</td></tr></table></td></tr>'
            f'</table>\n<!-- END template -->\n</body></html>')


def gen_url(rng: random.Random, bad: bool, brand_mismatch: bool, service: str = "") -> str:
    if not bad:
        dom = rng.choice(GOOD_DOMAINS + ["company.com", "brand-news.org"])
        return f"https://{dom}/view/{rng.randint(1000, 9999)}"
    tld = rng.choice(BAD_TLDS)
    stem = (re.sub(r"[^a-z]", "", service.lower()) + "-verify") if brand_mismatch and service \
        else "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 9)))
    style = rng.random()
    if style < 0.25:
        return f"http://bit.ly/{rng.randint(10000, 99999)}"
    if style < 0.45:
        return f"http://{rng.randint(11, 222)}.{rng.randint(1, 255)}.13.7/pay"
    if style < 0.65:
        return f"https://{stem}{rng.randint(10, 99)}.{tld}/{''.join(rng.choice('abcdef') for _ in range(18))}/x/y/z"
    return f"https://{stem}.{tld}/login?session={rng.randint(10**8, 10**9 - 1)}"


def inject_typos(text: str, rng: random.Random, p: float = 0.03) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) > 5 and rng.random() < p:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def scramble_sentence_lengths(text: str, rng: random.Random) -> str:
    """Анти-детектор: разбивает/объединяет предложения, чтобы поднять CV длины."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    filler = ["By the way", "Oh", "Quick note", "BTW", "Well"]
    for s in sents:
        if rng.random() < 0.35 and len(s.split()) > 8:
            mid = len(s.split()) // 2
            ws = s.split(" ")
            out.append(" ".join(ws[:mid]) + ".")
            out.append(rng.choice(filler) + ", " + " ".join(ws[mid:]).lower())
        else:
            out.append(s)
    return " ".join(out)


def build_dataset(n_total: int = 1200) -> List[Dict]:
    rng = random.Random(42)
    rows: List[Dict] = []
    counts = {"ham": int(n_total * 0.35), "spam_human": int(n_total * 0.15),
              "ai_full": int(n_total * 0.18), "ai_partial": int(n_total * 0.12),
              "ai_variants": int(n_total * 0.12), "ai_masked": int(n_total * 0.08)}
    for origin, cnt in counts.items():
        for k in range(cnt):
            rows.append(_make_row(rng, origin, idx=len(rows)))
    # добавляем явные кампании (по 4–9 писем) из ai_variants
    for c in range(12):
        camp_id = f"CAMP{c:02d}"
        base = rng.randrange(len(AI_CAMPAIGN_TEMPLATES))
        svc = rng.choice(SERVICES)
        domain_pool = [gen_url(rng, True, True, svc).split("/")[2] for _ in range(3)]
        for j in range(rng.randint(4, 9)):
            row = _make_campaign_variant(rng, base, svc, camp_id, j, domain_pool)
            rows.append(row)
    rng.shuffle(rows)
    return rows


def _common_fields(rng: random.Random, msg_id: str):
    return {"msg_id": msg_id, "date": "2026-0%d-%02dT%02d:00:00Z" % (
        rng.randint(1, 9), rng.randint(1, 28), rng.randint(0, 23))}


def _make_row(rng: random.Random, origin: str, idx: int) -> Dict:
    if origin == "ai_variants":
        # одиночные «вариационные» письма без явной кампании тоже существуют;
        # генерируем их как короткие версии кампейн-шаблонов
        return _make_campaign_variant(rng, rng.randrange(len(AI_CAMPAIGN_TEMPLATES)),
                                      rng.choice(SERVICES), camp_id=None, idx=idx,
                                      domain_pool=None)
    row = _common_fields(rng, f"{origin}-{idx:05d}")
    row["origin"] = origin
    row["_campaign"] = None
    service = rng.choice(SERVICES)
    name = rng.choice(NAMES)

    if origin == "ham":
        row["y_spam"], row["y_ai"] = 0, 0
        subj = rng.choice(HAM_SUBJECTS).format(oid=rng.randint(10000, 99999))
        n_phr = rng.randint(2, 5)
        body = " ".join(rng.sample(HUMAN_PHRASES, min(n_phr, len(HUMAN_PHRASES))))
        if rng.random() < 0.5:  # корпоративный стиль тоже бывает ровным
            body += " " + rng.choice([
                "Please find the requested figures attached and let me know if "
                "anything needs clarification before Friday's deadline.",
                "Kind regards, Irina", "Best, DevOps team"])
        row["subject"] = subj
        row["body"] = inject_typos(body, rng, 0.02) if rng.random() < 0.4 else body
        row["from"] = f"{rand_local(rng, 'human')}@{rng.choice(GOOD_DOMAINS)}"
        row["headers"] = {"X-Mailer": rng.choice(["Microsoft Outlook 16.0",
                                                  "Apple Mail", "Thunderbird"]),
                          "List-Unsubscribe": "<https://list.brand/optout>"}
        row["has_qr"] = False
        row["urls"] = [gen_url(rng, False, False)] if rng.random() < 0.5 else []
        row["html"] = make_html(rng, False, row["urls"]) if rng.random() < 0.4 else None

    elif origin == "spam_human":
        row["y_spam"], row["y_ai"] = 1, 0
        row["subject"] = rng.choice([
            "URGNN!!! U HAVE BEEN SELECTED 4 A REWARD", "cheap pills FAST delivery!!",
            "earn $$$ from home no experiance needed"])
        body = " ".join(rng.sample(SPAM_HUMAN_PHRASES, 3))
        row["body"] = inject_typos(body, rng, 0.05)
        row["from"] = f"{rand_local(rng, 'random')}@{rng.choice(BAD_TLDS)}.mailer"
        row["headers"] = {"X-Mailer": "BulkSend 2.1"}
        row["has_qr"] = False
        row["urls"] = [gen_url(rng, True, False)]
        row["html"] = None

    elif origin == "ai_full":
        row["y_spam"], row["y_ai"] = 1, 1
        tpl = AI_CAMPAIGN_TEMPLATES[rng.randrange(len(AI_CAMPAIGN_TEMPLATES))]
        body = tpl.format(name=name, svc=service)
        body += " " + rng.choice(LLM_PHRASES + LLM_PHRASES_RU)
        url = gen_url(rng, True, True, service)
        body += " " + CTA_LINES[rng.randrange(len(CTA_LINES))].format(url=url)
        row["subject"] = rng.choice([
            f"Action required: verify your {service} account",
            f"Important notice regarding your {service} profile",
            f"{service}: unusual activity detected on your account"])
        row["body"] = body
        row["from"] = f"{rand_local(rng, 'random')}@{rng.choice(BAD_TLDS)}-svc.example"
        row["headers"] = {"X-Mailer": "Python-SMTP/3.11", "Content-Language": "en"}
        row["has_qr"] = rng.random() < 0.3
        row["urls"] = [url]
        row["html"] = make_html(rng, True, [url])

    elif origin == "ai_partial":
        row["y_spam"], row["y_ai"] = 1, 1
        tpl = AI_CAMPAIGN_TEMPLATES[rng.randrange(len(AI_CAMPAIGN_TEMPLATES))]
        llm_part = tpl.format(name=name, svc=service)
        human_part = rng.choice(HUMAN_PHRASES + SPAM_HUMAN_PHRASES)
        body = llm_part + " " + human_part
        url = gen_url(rng, True, True, service)
        body += f" link: {url}"
        row["subject"] = f"{service} alert - pls check"
        row["body"] = inject_typos(body, rng, 0.02)
        row["from"] = f"support@{rng.choice(BAD_TLDS)}-update.example"
        row["headers"] = {"User-Agent": "Mozilla/5.0 (compatible; http-client)"}
        row["has_qr"] = False
        row["urls"] = [url]
        row["html"] = make_html(rng, rng.random() < 0.6, [url])

    elif origin == "ai_masked":
        row["y_spam"], row["y_ai"] = 1, 1
        tpl = AI_CAMPAIGN_TEMPLATES[rng.randrange(len(AI_CAMPAIGN_TEMPLATES))]
        body = scramble_sentence_lengths(tpl.format(name=name, svc=service), rng)
        body = inject_typos(body, rng, 0.04)
        url = gen_url(rng, True, True, service)
        row["subject"] = f"your {service} acccount - urgent"
        row["body"] = body + f" heres the page {url}"
        row["from"] = f"helpdesk@{rng.choice(BAD_TLDS)}.example"
        row["headers"] = {}
        row["has_qr"] = False
        row["urls"] = [url]
        row["html"] = make_html(rng, True, [url])
    return row


def _make_campaign_variant(rng: random.Random, base_tpl: int, svc: str,
                           camp_id, j: int = 0, domain_pool=None,
                           idx: int = -1) -> Dict:
    mid = f"{camp_id}-{j:02d}" if camp_id else f"ai_variants-{idx:05d}"
    row = _common_fields(rng, mid)
    row["origin"] = "ai_variants"
    row["y_spam"], row["y_ai"] = 1, 1
    row["_campaign"] = camp_id
    tpl = AI_CAMPAIGN_TEMPLATES[base_tpl]
    # вариация: разные имена/сервисы + лёгкий парафраз начала + случайный слот
    name = rng.choice(NAMES)
    body = tpl.format(name=name, svc=svc)
    lead = rng.choice([
        f"Attention {name.upper()}, ", "Final reminder: ", f"{svc} Security Team — ",
        f"Notice for {name}: ", ""])
    body = lead + body
    if rng.random() < 0.5:
        body = body.replace("Furthermore", "Moreover").replace(
            "Please do not hesitate", "Feel free")   # LLM-парафраз слота
    if domain_pool is None:
        domain_pool = [gen_url(rng, True, True, svc).split("/")[2] for _ in range(2)]
    url = f"https://{rng.choice(domain_pool)}/auth?id={rng.randint(10**6, 10**7)}"
    body += f" Please use: {url}"
    subjects = [f"Security alert for your {svc} account",
                f"{svc}: action required today", f"Unusual activity — {svc} team",
                f"Your {svc} verification step", f"{svc} account hold — confirm identity"]
    suffix = f" #{j}" if camp_id else ""
    row["subject"] = rng.choice(subjects) + suffix
    row["body"] = body
    row["from"] = f"{rand_local(rng, 'random')}@{rng.choice(BAD_TLDS)}-bulk.example"
    row["headers"] = {"X-Mailer": "SendMail-API v3", "Precedence": "bulk"}
    row["has_qr"] = rng.random() < 0.2
    row["urls"] = [url]
    row["html"] = make_html(rng, True, [url])
    return row


def main(out_path: str = "data/synthetic_corpus.jsonl", n: int = 1200):
    rows = build_dataset(n)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {p}")
    stats: Dict[str, int] = {}
    for r in rows:
        stats[r["origin"]] = stats.get(r["origin"], 0) + 1
    print("Distribution:", stats)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_corpus.jsonl")
    ap.add_argument("--n", type=int, default=1200)
    a = ap.parse_args()
    main(a.out, a.n)
