"""Тесты парсера raw-RFC822 и URL-анализа."""
from slop_detector.parser import analyze_url, parse_raw

RAW = (
    "Message-ID: <abc@ex.com>\r\n"
    "From: \"PayPal Support\" <service@paypa1-alerts.xyz>\r\n"
    "To: victim@example.com\r\n"
    "Subject: Urgent: verify your account now!\r\n"
    "Date: Mon, 01 Jun 2026 10:00:00 +0000\r\n"
    "X-Mailer: PythonEmailGen\r\n"
    "MIME-Version: 1.0\r\n"
    "Content-Type: multipart/mixed; boundary=\"B\"\r\n"
    "\r\n"
    "--B\r\n"
    "Content-Type: text/plain; charset=utf-8\r\n"
    "\r\n"
    "Dear user, please verify your PayPal account at https://paypa1-alerts.xyz/verify "
    "and scan the QR code below. http://192.168.1.1/login\r\n"
    "--B\r\n"
    "Content-Type: application/octet-stream; name=\"invoice.pdf\"\r\n"
    "Content-Disposition: attachment; filename=\"invoice.pdf\"\r\n"
    "\r\n"
    "fakebytes\r\n"
    "--B--\r\n"
)


def test_parse_headers_and_body():
    rec = parse_raw(RAW)
    assert rec.from_addr == "service@paypa1-alerts.xyz"
    assert rec.from_display == "PayPal Support"
    assert "Urgent" in rec.subject
    assert rec.header("x-mailer") == "PythonEmailGen"
    assert "verify" in rec.text_body
    assert rec.has_qr_reference is True


def test_parse_attachment_and_urls():
    rec = parse_raw(RAW)
    assert len(rec.attachments) == 1
    assert rec.attachments[0].name == "invoice.pdf"
    assert rec.attachments[0].ext == "pdf"
    domains = set(rec.domains)
    assert any("paypa1" in d for d in domains)
    ip_url = [u for u in rec.urls if u.is_ip_literal]
    assert len(ip_url) == 1


def test_brand_mismatch_flagged():
    ui = analyze_url("https://paypa1-alerts.xyz/verify", {"paypal"})
    assert ui.has_brand_mismatch is True
    assert ui.suspicious_tld is True


def test_homoglyph_punycode():
    ui = analyze_url("https://xn--80ak6aa92e.com/login", set())
    assert ui.punycode_or_homoglyph is True


def test_shortener_detected():
    ui = analyze_url("https://bit.ly/3xAbC", set())
    assert ui.url_shortener is True
