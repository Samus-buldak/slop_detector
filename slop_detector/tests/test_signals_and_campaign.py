"""Тесты URL-сигналов и кампейн-анализатора."""
from slop_detector.campaign.analyzer import (
    CampaignAnalyzer, coarse_fingerprint, jaccard_from_mh, minhash, shingles)
from slop_detector.schema import EmailRecord, UrlInfo
from slop_detector.signals.url_signals import url_signals


def _rec_with_urls(urls, body="", has_qr=False):
    return EmailRecord(msg_id="u1", subject="x", text_body=body,
                       urls=[UrlInfo(raw=u, domain=d) for u, d in urls],
                       has_qr_reference=has_qr)


def test_url_signals_ratios():
    rec = _rec_with_urls([
        ("https://bit.ly/abc", "bit.ly"),
        ("http://1.2.3.4/x", "1.2.3.4"),
    ])
    f, expl = url_signals(rec)
    assert f["url_count"] == 2
    assert f["ip_literal_ratio"] == 0.5
    assert any("IP" in x or "шортенер" in x.lower() for x in expl)


def test_qr_without_urls():
    rec = _rec_with_urls([], body="Scan the QR code to claim", has_qr=True)
    f, expl = url_signals(rec)
    assert f["qr_reference_flag"] == 1.0
    assert any("QR" in x for x in expl)


def test_cta_counting():
    rec = _rec_with_urls([], body="Click here to verify now and act now")
    f, expl = url_signals(rec)
    assert f["cta_phrase_count"] >= 3
    assert any("CTA" in x for x in expl)


# ---------------- campaign ----------------

def _camp_rec(mid, body, frm="bulk@evil-%d.com"):
    r = EmailRecord(msg_id=mid, subject="Invoice overdue", text_body=body)
    r.from_addr = frm % 1
    return r


BASE = ("Dear customer, your invoice payment is overdue. Please review the attached "
        "statement and settle the balance within 3 business days to avoid service "
        "interruption of your account subscription services today.")
VARIANT = ("Hello customer, your invoice payment is late. Please check the attached "
           "statement and pay the balance within 3 business days to avoid service "
           "interruption of your account subscription services today.")
DIFFERENT = ("Team, about the quarterly roadmap meeting: agenda covers hiring budget, "
             "infra migration timeline, and the Q3 OKR review with leadership next week.")


def test_shingles_jaccard_self_is_one():
    s = shingles(BASE)
    mh = minhash(s)
    assert abs(jaccard_from_mh(mh, mh) - 1.0) < 0.1


def test_grouping_variants_into_campaign():
    recs = [_camp_rec(f"c{i}", BASE if i % 2 else VARIANT) for i in range(6)]
    recs.append(_camp_rec("z", DIFFERENT))
    campaigns = CampaignAnalyzer(sim_threshold=0.45).group(recs)
    sizes = sorted((len(c.msg_ids) for c in campaigns), reverse=True)
    assert sizes[0] >= 6          # все варианты одной кампании вместе
    assert "z" not in campaigns[0].msg_ids or \
        all(len(c.msg_ids) == 1 for c in campaigns if "z" in c.msg_ids)


def test_ham_not_grouped_with_spam():
    recs = [_camp_rec("a", BASE), _camp_rec("b", DIFFERENT)]
    campaigns = CampaignAnalyzer().group(recs)
    assert all(len(c.msg_ids) == 1 for c in campaigns)


def test_coarse_fingerprint_stable_across_variants():
    a = _camp_rec("a", BASE)
    b = _camp_rec("b", VARIANT)
    assert coarse_fingerprint(a) == coarse_fingerprint(b)
