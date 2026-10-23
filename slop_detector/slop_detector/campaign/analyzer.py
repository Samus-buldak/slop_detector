"""Кампейн-анализ: группировка писем и выявление автоматизированной вариативности.

Метод (вдохновлён IMC'25 и SpearBot):
  1. Шинглинг-минихеш векторизация нормализованного текста (subject+body).
  2. Лигирование похожих писем в кампании (connected components по порогу
     косинусного сходства + общий домен/инфраструктура).
  3. Для каждой кампании считаем «подпись вариативности»:
     - similarity distribution (среднее/медиана/разброс);
     - template slots: позиции, где письма одной кампании различаются при
       идентичном окружении (маркер {var}/автоподстановки или LLM-парафраз);
     - paraphrase score: высокая лексическая дистанция при низкой структурной;
     - scaling pattern: одинаковые CTA/URL-шаблоны при разных формулировках.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema import EmailRecord


def normalize_text(rec: EmailRecord) -> str:
    t = f"{rec.subject} {rec.text_body}"
    # вырезаем служебные заголовки/подписи: общий boilerplate не должен
    # искусственно завышать сходство разнородных писем
    t = re.sub(r"(?im)^\s*(?:--\s*$|sent from my (?:iphone|ipad|android).*|"
               r"this (?:is )?(?:a )?message (?:was )?(?:generated|sent) automatically.*|"
               r"confidential(?:ity)? notice.*|please do not reply(?: to this email)?.*)",
               " ", t)
    t = t.lower()
    t = re.sub(r"\d+", "#", t)                       # числа → шаблонные слоты
    t = re.sub(r"[\w.+-]+@[\w.-]+", "<email>", t)
    t = re.sub(r"(?:https?://|www\.)\S+", "<url>", t)
    return re.sub(r"\s+", " ", t).strip()


BRAND_TOKENS = ["paypal", "microsoft", "netflix", "dropbox", "apple", "amazon",
                "google", "facebook", "linkedin", "github", "dhl", "fedex", "ups",
                "bank", "invoice", "delivery", "parcel"]


def brand_key(rec: EmailRecord) -> str:
    """Бренд-приманка/жанр письма по первому упомянутому ключевому слову."""
    text = f"{rec.subject} {rec.text_body}".lower()
    for b in BRAND_TOKENS:
        if re.search(rf"\b{b}\b", text):
            return b
    return ""


def coarse_fingerprint(rec: EmailRecord) -> str:
    """Грубый отпечаток инфраструктуры/жанра письма.

    Кампании объединяются только внутри одного fingerprint — это предотвращает
    «слипание» несвязанных шаблонов при транзитивном замыкании сходства.
    Fingerprint включает: домен отправителя, длину темы (грубо), наличие и
    структуру HTML (таблица/base64), количество ссылок и TLD первой ссылки.
    """
    import hashlib
    h = rec.html_body or ""
    html_kind = ("table_b64" if ("<table" in h.lower() and "base64" in h) else
                 ("table" if "<table" in h.lower() else
                  ("html" if h else "plain")))
    tld = ""
    if rec.urls:
        d = rec.urls[0].domain
        tld = d.rsplit(".", 1)[-1] if "." in d else d
    subj_bucket = min(3, len(rec.subject or "") // 25)
    n_urls = min(3, len(rec.urls))
    key = f"{rec.sender_domain}|{html_kind}|{n_urls}|{tld}|{subj_bucket}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def shingles(text: str, k: int = 5) -> set:
    toks = text.split()
    if len(toks) < k:
        return {text} if text else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(sh: set, num_perm: int = 64, seed: int = 1) -> List[int]:
    if not sh:
        return [0] * num_perm
    hashes = []
    for i in range(num_perm):
        hmin = None
        s = seed + i
        for g in sh:
            v = int(hashlib.md5(f"{s}:{g}".encode()).hexdigest(), 16) % (2**31)
            hmin = v if hmin is None or v < hmin else hmin
        hashes.append(hmin or 0)
    return hashes


def jaccard_from_mh(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    eq = sum(1 for x, y in zip(a, b) if x == y)
    return eq / len(a)


@dataclass
class Campaign:
    campaign_id: str
    msg_ids: List[str] = field(default_factory=list)
    sender_domains: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    labels: Dict[str, object] = field(default_factory=dict)


# ---------- template slot detection ----------
def _tokenize(text: str) -> List[str]:
    return text.split()


def template_slot_stats(records: List[EmailRecord]) -> Dict[str, float]:
    """Оценка «шаблонных слотов»: выравнивание через общий префикс/суффикс
    и доля токенов, различающихся между письмами при одинаковом контексте."""
    if len(records) < 2:
        return {"slot_ratio": 0.0, "prefix_shared_len": 0.0}
    seqs = [_tokenize(normalize_text(r)) for r in records]
    # общий префикс всех пар (приближение к shared template)
    def common_prefix(a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        return i
    pref = min(common_prefix(s, seqs[0]) for s in seqs[1:]) if seqs[0] else 0
    avg_len = statistics.fmean(len(s) for s in seqs) or 1
    # доля «вариабельных» токенов = 1 - покрытие общим шаблоном
    variable = max(0.0, 1.0 - (2 * pref) / avg_len) if avg_len else 0.0
    return {"slot_ratio": round(variable, 4), "prefix_shared_len": float(pref)}


class CampaignAnalyzer:
    def __init__(self, sim_threshold: float = 0.45, core_threshold: float = 0.62):
        self.sim_threshold = sim_threshold
        # минимальное «ядро» сходства для лигирования разных (не идентичных)
        # писем; защищает от transitive drift и ложного слияния шаблонов
        self.core_threshold = core_threshold

    def group(self, records: List[EmailRecord]) -> List[Campaign]:
        mhs = {r.msg_id: minhash(shingles(normalize_text(r))) for r in records}
        ids = list(mhs.keys())
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        by_rec = {r.msg_id: r for r in records}
        # 1) сначала объединяем письма с идентичным нормализованным текстом
        #    (классический bulk-спам без вариативности — это тоже кампания);
        norm_groups: Dict[str, List[str]] = defaultdict(list)
        for r in records:
            norm_groups[(coarse_fingerprint(r), normalize_text(r))].append(r.msg_id)
        for members in norm_groups.values():
            for m in members[1:]:
                union(members[0], m)
        # 2) затем — по порогу сходства шинглов. Обязательное условие — общий
        #    «жанр» письма (coarse-fingerprint ИЛИ один бренд-приманка): это
        #    блокирует chain-linking несвязанных шаблонов. Внутри жанра пары
        #    склеиваются при sim >= max(sim_threshold, core_threshold); порог
        #    core_threshold отсекает transitive drift (цепочки A~B~C~D, где
        #    крайние уже не похожи). Служебные заголовки из текста исключены
        #    (см. normalize_text), чтобы одинаковый boilerplate не склеивал
        #    разнородные сообщения.
        genre_fp = {i: coarse_fingerprint(by_rec[i]) for i in ids}
        genre_brand = {i: brand_key(by_rec[i]) for i in ids}

        def same_genre(a: str, b: str) -> bool:
            return (genre_fp[a] == genre_fp[b] or
                    (genre_brand[a] and genre_brand[a] == genre_brand[b]))

        thr = max(self.sim_threshold, self.core_threshold)
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                a, b = ids[ii], ids[jj]
                if find(a) == find(b) or not same_genre(a, b):
                    continue
                sim = jaccard_from_mh(mhs[a], mhs[b])
                same_infra = (by_rec[a].sender_domain and
                              by_rec[a].sender_domain == by_rec[b].sender_domain)
                if sim >= thr or (same_infra and sim >= thr * 0.6):
                    union(a, b)

        groups: Dict[str, List[str]] = defaultdict(list)
        for i in ids:
            groups[find(i)].append(i)

        campaigns: List[Campaign] = []
        for cid, (_, members) in enumerate(sorted(groups.items())):
            recs = [by_rec[m] for m in members]
            camp = Campaign(campaign_id=f"C{cid:03d}", msg_ids=members,
                            sender_domains=sorted({r.sender_domain for r in recs if r.sender_domain}))
            feats, expl = self._score_campaign(recs, mhs)
            camp.features = feats
            camp.explanations = expl
            campaigns.append(camp)
        return campaigns

    def _score_campaign(self, recs: List[EmailRecord],
                        mhs: Dict[str, List[int]]) -> Tuple[Dict[str, float], List[str]]:
        f: Dict[str, float] = {}
        expl: List[str] = []
        n = len(recs)
        f["campaign_size"] = float(n)
        if n < 2:
            return f, ["Группа из одного сообщения — кампания не подтверждена."]

        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(jaccard_from_mh(mhs[recs[i].msg_id], mhs[recs[j].msg_id]))
        f["pair_sim_mean"] = statistics.fmean(sims)
        f["pair_sim_std"] = statistics.pstdev(sims) if len(sims) > 1 else 0.0
        f["pair_sim_min"] = min(sims)
        f["pair_sim_max"] = max(sims)

        slots = template_slot_stats(recs)
        f.update(slots)

        # разнообразие тем при высокой схожести тел — автогенерация заголовков
        subj_norm = {re.sub(r"\d+", "#", (r.subject or "").lower()) for r in recs}
        f["subject_diversity"] = len(subj_norm) / n
        # разные домены в рамках одной кампании = ротация инфраструктуры
        doms = {d for r in recs for d in r.domains}
        f["unique_link_domains"] = float(len(doms))
        senders = {r.sender_domain for r in recs if r.sender_domain}
        f["unique_sender_domains"] = float(len(senders))

        # AI-campaign signature: высокая sim + высокая subject diversity +
        # средняя вариативность слотов → массовые парафраз-вариации
        ai_sig = (f["pair_sim_mean"] > 0.5 and f["subject_diversity"] >= 0.7
                  and 0.15 < f["slot_ratio"] < 0.85)
        f["ai_variation_flag"] = float(ai_sig)
        if ai_sig:
            expl.append("Письма кампании очень похожи структурно (sim=%.2f), но имеют "
                        "уникальные темы и вариативные слоты (%.2f) — характерно для "
                        "LLM-парафраза на масштабе." % (f["pair_sim_mean"], f["slot_ratio"]))
        elif f["pair_sim_mean"] > 0.9:
            expl.append("Практически идентичные письма — классический bulk-спам без "
                        "заметной вариативности.")
        if f["unique_sender_domains"] >= 2 and f["pair_sim_mean"] > 0.5:
            expl.append("Один шаблон рассылается с %d доменов — скоординированная "
                        "инфраструктура кампании." % f["unique_sender_domains"])
        if f["unique_link_domains"] >= 3:
            expl.append("Внутри кампании используется %d различных ссылочных доменов "
                        "— ротация URL типична для устойчивых кампаний."
                        % f["unique_link_domains"])
        return f, expl
