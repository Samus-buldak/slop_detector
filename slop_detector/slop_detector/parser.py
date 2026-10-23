"""Парсер raw RFC822 / MIME писем в EmailRecord."""
from __future__ import annotations

import email
import email.policy
import re
from email.message import EmailMessage
from typing import Optional
from urllib.parse import urlparse

from .schema import AttachmentInfo, EmailRecord, UrlInfo

SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "link", "gdn", "cam", "rest",
    "icu", "cyou", "sbs", "shop", "lol", "bond", "beauty", "quest", "pw",
    "cc", "tk", "ml", "ga", "cf", "gq", "bit", "onion", "country", "work",
}
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rb.gy", "cutt.ly", "rebrand.ly", "shorturl.at", "id.tl", "lnkd.in",
    "4sq.com", "j.mp", "trib.al", "waplog.com", "mcaf.ee", "soo.gd", "s.id",
}
BRAND_KEYWORDS = {
    "paypal": "paypal.com", "microsoft": ("microsoft.com", "live.com", "outlook.com"),
    "google": "google.com", "apple": "apple.com", "amazon": "amazon.",
    "netflix": "netflix.", "facebook": ("facebook.com", "meta.com"),
    "bank": None, "dhl": "dhl.", "fedex": "fedex.", "ups": "ups.com",
    "github": "github.com", "dropbox": "dropbox.com", "linkedin": "linkedin.com",
}


def _domain_of(url: str) -> str:
    try:
        if "://" not in url:
            url = "http://" + url
        netloc = urlparse(url).netloc.lower()
        return netloc.split("@")[-1].split(":")[0]
    except Exception:
        return ""


def analyze_url(raw: str, claimed_brands: set[str]) -> UrlInfo:
    dom = _domain_of(raw)
    path = ""
    try:
        u = urlparse(raw if "://" in raw else "http://" + raw)
        path = (u.path or "") + ("?" + u.query if u.query else "")
    except Exception:
        pass
    info = UrlInfo(raw=raw, domain=dom, path=path)
    info.is_ip_literal = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", dom))
    tld = dom.rsplit(".", 1)[-1] if dom else ""
    info.suspicious_tld = tld in SUSPICIOUS_TLDS
    info.url_shortener = dom in URL_SHORTENERS
    info.punycode_or_homoglyph = dom.startswith("xn--") or bool(
        re.search(r"[^\x00-\x7f]", dom))
    info.digits_in_domain = bool(dom) and len(re.findall(r"\d", dom.split(".")[0])) >= 3
    seg = [s for s in path.split("/") if s]
    info.deep_path = len(seg) >= 4 or any(len(s) >= 16 for s in seg)
    # brand mismatch: текст письма упоминает бренд, а домен — не бренд-домен
    for b in claimed_brands:
        expected = BRAND_KEYWORDS.get(b.lower())
        if expected is None:
            continue
        expected_tuple = (expected,) if isinstance(expected, str) else expected
        if not any(dom.endswith(e) or e.rstrip('.') in dom for e in expected_tuple):
            if dom and not any(b.lower() in dom for _ in [0]):
                info.has_brand_mismatch = True
                break
    return info


_URL_RE = re.compile(
    r'(?:https?://|www\.)[^\s"\'<>()\[\]]+', re.IGNORECASE)
_QR_RE = re.compile(r"\b(qr[\s\-]?code|отсканир\w+\s+код|scan\s+the\s+code)\b", re.I)


def extract_html_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_raw(raw_message: str, msg_id: Optional[str] = None) -> EmailRecord:
    """Разбор raw RFC822-письма."""
    msg: EmailMessage = email.message_from_string(raw_message, policy=email.policy.default)
    rec = EmailRecord(msg_id=msg_id or msg.get("Message-ID", "unknown"))
    rec.subject = str(msg.get("Subject", "") or "")
    frm = str(msg.get("From", "") or "")
    m = re.match(r'\s*(?:"?([^"<]*)"?\s*)?<?([\w.+-]+@[\w.-]+)>?', frm)
    if m:
        rec.from_display = (m.group(1) or "").strip()
        rec.from_addr = m.group(2)
    else:
        rec.from_addr = frm.strip()
    rec.to_addrs = [a.strip() for a in str(msg.get("To", "")).split(",") if a.strip()]
    rec.date = str(msg.get("Date", "") or "")
    for key in ("received", "x-mailer", "user-agent", "content-language",
                "list-unsubscribe", "precedence", "auto-submitted", "reply-to"):
        val = msg.get(key)
        if val:
            rec.headers[key] = str(val)

    text_parts, html_parts, atts = [], [], []
    if msg.is_multipart():
        for part in msg.walk():
            disp = part.get_content_disposition()
            ct = part.get_content_type()
            if disp == "attachment" or (part.get_filename() and ct not in
                                        ("text/plain", "text/html")):
                fn = part.get_filename() or "unnamed"
                payload = part.get_payload(decode=True) or b""
                atts.append(AttachmentInfo(name=fn, ext=fn.rsplit(".", 1)[-1].lower()
                                           if "." in fn else "", size=len(payload), mime=ct))
                continue
            if ct == "text/plain":
                try:
                    text_parts.append(part.get_content())
                except Exception:
                    pass
            elif ct == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    pass
    else:
        try:
            content = msg.get_content()
        except Exception:
            content = str(msg.get_payload())
        if msg.get_content_type() == "text/html":
            html_parts.append(content)
        else:
            text_parts.append(content)

    rec.html_body = "\n".join(html_parts) if html_parts else None
    body_text = "\n".join(text_parts).strip()
    if not body_text and rec.html_body:
        body_text = extract_html_text(rec.html_body)
    rec.text_body = body_text
    rec.has_qr_reference = bool(_QR_RE.search(rec.text_body) or
                                (_QR_RE.search(rec.html_body or "")))
    if rec.html_body:
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', rec.html_body, re.I)
        imgs = re.findall(r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg)[^"\']*)["\']',
                          rec.html_body, re.I)
        all_urls = hrefs + imgs + _URL_RE.findall(rec.text_body)
    else:
        all_urls = _URL_RE.findall(rec.text_body)

    brands = {b for b in BRAND_KEYWORDS if re.search(rf"\b{b}\b", rec.full_text, re.I)}
    seen = set()
    for u in all_urls:
        d = _domain_of(u)
        if not d or d in seen:
            continue
        seen.add(d)
        rec.urls.append(analyze_url(u, brands))
    rec.attachments = atts
    return rec


def record_from_dict(row: dict, idx: int = 0) -> EmailRecord:
    """Построение записи из плоского словаря датасета (синтетика/реальные данные)."""
    rec = EmailRecord(
        msg_id=str(row.get("msg_id", f"rec-{idx}")),
        subject=str(row.get("subject", "")),
        from_addr=str(row.get("from", "")),
        from_display=str(row.get("from_display", "")),
        to_addrs=[str(x) for x in row.get("to", [])] if isinstance(row.get("to"), list)
                 else [str(row.get("to", ""))] if row.get("to") else [],
        date=str(row.get("date", "")),
        text_body=str(row.get("body", "")),
        html_body=row.get("html"),
        headers={k.lower(): str(v) for k, v in (row.get("headers") or {}).items()},
        has_qr_reference=bool(row.get("has_qr", False)),
        extra={k: v for k, v in row.items() if k.startswith("_")},
    )
    brands = {b for b in BRAND_KEYWORDS if re.search(rf"\b{b}\b", rec.full_text, re.I)}
    urls = row.get("urls") or _URL_RE.findall(rec.text_body)
    if rec.html_body:
        urls = list(urls) + re.findall(r'href=["\']([^"\']+)["\']', rec.html_body, re.I)
    seen = set()
    for u in urls:
        d = _domain_of(u)
        if not d or d in seen:
            continue
        seen.add(d)
        rec.urls.append(analyze_url(u, brands))
    for a in (row.get("attachments") or []):
        if isinstance(a, dict):
            rec.attachments.append(AttachmentInfo(
                name=a.get("name", ""), ext=a.get("ext", ""), size=int(a.get("size", 0))))
        else:
            rec.attachments.append(AttachmentInfo(name=str(a)))
    return rec
