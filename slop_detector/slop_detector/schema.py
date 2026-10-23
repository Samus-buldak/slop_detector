"""Внутреннее представление email-сообщения.

EmailRecord — нормализованная структура, из которой все сигнальные модули
извлекают признаки. Может быть построен из:
  - raw RFC822-текста (parser.py),
  - словаря датасета (синтетического или реального),
  - JSON с полями subject/body/html/url и т.д.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UrlInfo:
    """Извлечённая информация об одной ссылке/домене."""
    raw: str
    domain: str = ""
    path: str = ""
    is_ip_literal: bool = False
    suspicious_tld: bool = False
    has_brand_mismatch: bool = False   # домен не совпадает сclaimed брендом
    punycode_or_homoglyph: bool = False
    url_shortener: bool = False
    deep_path: bool = False            # длинные/случайные пути
    digits_in_domain: bool = False


@dataclass
class AttachmentInfo:
    name: str
    ext: str = ""
    size: int = 0
    mime: str = ""


@dataclass
class EmailRecord:
    msg_id: str
    subject: str = ""
    from_addr: str = ""
    from_display: str = ""
    to_addrs: List[str] = field(default_factory=list)
    date: str = ""
    text_body: str = ""
    html_body: Optional[str] = None
    urls: List[UrlInfo] = field(default_factory=list)
    attachments: List[AttachmentInfo] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    has_qr_reference: bool = False          # явное упоминание QR в тексте/HTML
    extra: Dict[str, object] = field(default_factory=dict)

    # ---------- helpers ----------
    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.text_body}"

    @property
    def domains(self) -> List[str]:
        return [u.domain for u in self.urls if u.domain]

    @property
    def sender_domain(self) -> str:
        m = re.search(r"@([\w.-]+)", self.from_addr or "")
        return m.group(1).lower() if m else ""

    def header(self, name: str) -> str:
        return self.headers.get(name.lower(), "")
