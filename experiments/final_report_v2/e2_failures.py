"""Experiment 2: which malicious URLs do the cascades lose by exiting early?

An additional miss is a malicious test URL that the GRU alone detects but the cascade ends
as benign at stage one. This script rebuilds every frozen route from the stored row scores,
checks the counts against v1, and breaks the misses down by URL properties fixed before
looking at the misses. It also counts benign URLs that the early exit saves from a GRU
false positive, the other side of the same decision.
"""

import hashlib
import ipaddress
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from v2common import V1, RESULTS, MAIN, FAMILIES, load_test, taus, write_json
import evaluate as ev

PREFIX = re.compile(r"^https?://", re.I)


def split_url(url):
    return urlsplit(url if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", url) else "//" + url)


def host_of(url):
    try:
        return (split_url(url).hostname or "").lower()
    except ValueError:
        return ""


def has_path_or_query(url):
    """URL components, not characters: a '/' or '?' after '#' does not count."""
    parsed = split_url(url)
    return bool(parsed.path or parsed.query)


def is_ip(host):
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def defang(url, limit=70):
    """Defang first, then cut to at most `limit` characters including '…' without splitting '[.]'."""
    text = re.sub(r"^http", "hxxp", url, flags=re.I)
    full = text.replace(".", "[.]")
    if len(full) <= limit:
        return full
    out = ""
    for c in text:
        piece = "[.]" if c == "." else c
        if len(out) + len(piece) > limit - 1:
            break
        out += piece
    return out + "…"


def tld_of(domain):
    """Last label of the registered domain; IP hosts and dotless names get their own level."""
    if is_ip(domain):
        return "IP 주소"
    if "." not in domain:
        return "점 없는 호스트"
    return domain.rsplit(".", 1)[-1].lower()


def groups(kind, frame, detected):
    train = pd.read_pickle(V1 / kind / "train.pkl")
    q = np.quantile(train.url.str.len().to_numpy(), [0.25, 0.5, 0.75])
    length = frame.url.str.len().to_numpy()
    quart = np.select([length <= q[0], length <= q[1], length <= q[2]], ["Q1", "Q2", "Q3"], "Q4")
    hosts = frame.input_url.map(host_of)
    # Fixed before looking at misses: the six TLDs with the most GRU-detected malicious test URLs
    # in this split (ties by name), the same rows for every cascade; everything else is "기타".
    raw_tld = frame.domain.map(tld_of).to_numpy()
    counts = pd.Series(raw_tld[detected]).value_counts()
    named = counts[~counts.index.isin(["IP 주소", "점 없는 호스트"])]
    top = sorted(named.index, key=lambda k: (-int(named[k]), k))[:6]
    tld = np.where(np.isin(raw_tld, top + ["IP 주소", "점 없는 호스트"]), raw_tld, "기타")
    return {
        "prefix": np.where(frame.url.str.match(PREFIX), "http(s):// 있음", "없음"),
        "www": np.where(frame.input_url.str.replace(PREFIX, "", n=1, regex=True).str.lower().str.startswith("www."), "www. 있음", "없음"),
        "ip_host": np.where(hosts.map(is_ip), "IP 주소", "도메인 이름"),
        "length_quartile": quart,
        "has_path_or_query": np.where(frame.input_url.map(has_path_or_query), "경로·쿼리 있음", "경로·쿼리 없음"),
        "source_file": frame.index.str.split(":").str[0].to_numpy(),
        "tld": tld,
    }, {"train_length_quartiles": q.tolist(), "tld_levels": top}


def main():
    breakdown, summary, cases = [], [], []
    main28 = pd.read_csv(V1 / "results/main-28.csv")
    for split, kind in MAIN.items():
        frame, s = load_test(kind)
        ids, y, gru = s["id"], s["y"], s["gru"]
        first = {f: s[f] for f in FAMILIES}
        t = taus(kind)
        masks, res = ev.configurations(y, ids, gru, first, t)
        detected = (y == 1) & (gru > 0.5)
        g, meta = groups(kind, frame, detected)
        gru_fp = (y == 0) & (gru > 0.5)
        for name, mask in masks.items():
            if name == "gru":
                continue
            ref = main28[(main28.kind == kind) & (main28.name == name)].iloc[0]
            miss = detected & ~mask
            saved_fp = gru_fp & ~mask
            assert int(miss.sum()) == int(ref.additional_misses) == res[name]["additional_misses"]
            fam = name.split("_")[0]
            p = first[fam]
            summary.append(dict(
                split=split, kind=kind, name=name, tau=t[fam],
                gru_detected_malicious=int(detected.sum()), additional_misses=int(miss.sum()),
                gru_false_positives=int(gru_fp.sum()), false_positives_removed_by_early_exit=int(saved_fp.sum()),
                miss_stage1_median=float(np.median(p[miss])) if miss.any() else None,
                miss_stage1_below_tau_over_10=int((p[miss] < t[fam] / 10).sum()),
                miss_another_file_without_prefix=int((miss & (g["source_file"] == "df_another_validation.csv") & (g["prefix"] == "없음")).sum()),
                **meta))
            for gname, values in g.items():
                levels = pd.Series(values[detected]).value_counts()
                order = sorted(levels.index, key=lambda k: (-int(levels[k]), str(k)))
                for level in order:
                    sel = values == level
                    nd = int((detected & sel).sum())
                    nm = int((miss & sel).sum())
                    breakdown.append(dict(split=split, name=name, group=gname, level=level,
                                          gru_detected=nd, additional_misses=nm,
                                          miss_rate_pct=100 * nm / nd if nd else None,
                                          share_of_misses_pct=100 * nm / miss.sum() if miss.any() else None))
            if name in ("logistic_boundary", "boost_boundary"):
                order = sorted(np.flatnonzero(miss), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
                for i in order[:3]:
                    cases.append(dict(split=split, name=name, url=defang(frame.url.iloc[i]),
                                      stage1_malicious=float(p[i]), tau=t[fam], gru_malicious=float(gru[i]),
                                      prefix=g["prefix"][i], source_file=g["source_file"][i]))
        both = (detected & ~masks["logistic_basic"]) & (detected & ~masks["boost_basic"])
        summary.append(dict(split=split, kind=kind, name="overlap_basic", additional_misses=int(both.sum())))
    b = pd.DataFrame(breakdown)
    s = pd.DataFrame(summary)
    # Every group partitions the GRU-detected malicious rows, so its levels must add up exactly.
    partition = []
    for (split, name, gname), part in b.groupby(["split", "name", "group"]):
        ref = s[(s.split == split) & (s.name == name)].iloc[0]
        ok = (int(part.gru_detected.sum()) == int(ref.gru_detected_malicious)
              and int(part.additional_misses.sum()) == int(ref.additional_misses))
        partition.append(dict(split=split, name=name, group=gname, ok=bool(ok)))
    assert all(r["ok"] for r in partition)
    b.to_csv(RESULTS / "e2-miss-breakdown.csv", index=False)
    s.to_csv(RESULTS / "e2-miss-summary.csv", index=False)
    c = pd.DataFrame(cases)
    assert (c.url.str.len() <= 70).all()
    c.to_csv(RESULTS / "e2-miss-cases.csv", index=False)
    write_json(RESULTS / "e2-checks.json", {
        "status": "PASS",
        "checks": ["additional misses equal v1 main-28 for all 12 cascades",
                   f"each of {len(partition)} (split, cascade, group) breakdowns sums to the GRU-detected and missed totals"],
        "tld_rule": "six TLDs with most GRU-detected malicious test URLs per split (ties by name), plus IP 주소, 점 없는 호스트, 기타; fixed before reading misses",
    })
    print(s.to_string())
    print(b[b.name.isin(["logistic_basic", "boost_basic"]) & (b.group != "tld")].round(2).to_string())
    print(pd.DataFrame(cases).to_string())


if __name__ == "__main__":
    main()
