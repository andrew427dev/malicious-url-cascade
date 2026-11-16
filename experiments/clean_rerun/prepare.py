"""Offline, auditable cleaning and immutable splits for the professor-approved rerun."""

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from common import CODE_DIR, ROOT, SOURCE, sha, write_json

sys.path.insert(0, str(SOURCE))
from urlcascade.features import extract_features, FEATURE_NAMES
from urlcascade.adapters import preprocess_link

EXPECTED = {
    "df_final.csv": "303effaa48986335fb6f7af07b00b2fda193fca22d990d0a78a273d542e3bd08",
    "all_top_domains_merged.csv": "da1cf85b221f618d396b135554d41df2581fb4af98bcd2c41cd7de08dfa9b501",
    "df_another_validation.csv": "676d758897212240d71da36783f1190866c18d688f516b53202802b364503b80",
}


def normalize(raw):
    if not raw:
        raise ValueError("empty_url")
    if any(unicodedata.category(c) in ("Cc", "Cf") for c in raw):
        raise ValueError("control_or_format_character")
    raw = raw.strip()
    if not raw:
        raise ValueError("empty_url")
    explicit = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw))
    parsed = urlsplit(raw if explicit or raw.startswith("//") else "//" + raw)
    if explicit and parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("unsupported_scheme")
    host = parsed.hostname
    if not host or any(c.isspace() for c in host):
        raise ValueError("invalid_host")
    # Validate port without inventing or deleting a port. Preserve user info and path case.
    parsed.port
    netloc = parsed.netloc
    at = netloc.rfind("@")
    authority = netloc[at + 1 :]
    if authority.startswith("["):
        end = authority.index("]")
        authority = authority[: end + 1].lower() + authority[end + 1 :]
    else:
        h, sep, port = authority.partition(":")
        authority = h.lower() + sep + port
    netloc = netloc[: at + 1] + authority
    canonical = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), parsed.query, parsed.fragment)
    )
    if not explicit and not raw.startswith("//"):
        canonical = canonical[2:]
    return canonical, host.lower()


def summarize(df):
    c = Counter(df.y_mal.tolist())
    return {"rows": len(df), "benign": c.get(0, 0), "malicious": c.get(1, 0)}


def check_normalizer():
    assert normalize("HTTPS://EXAMPLE.COM/AbC/")[0] == "https://example.com/AbC"
    assert normalize("Example.COM/a/?Q=X/")[0] == "example.com/a?Q=X/"
    assert normalize("http://User:Pass@EXAMPLE.COM:80/a/")[0] == "http://User:Pass@example.com:80/a"
    assert normalize("http://[2001:DB8::1]/")[0] == "http://[2001:db8::1]"
    for bad in ("a\x00.com", "a\u200b.com", "", "http://bad:xxx/"):
        try:
            normalize(bad)
        except ValueError:
            continue
        raise AssertionError("accepted invalid fixture")
    assert normalize("x.com/A")[0] != normalize("x.com/a")[0]


def make_partitions(df):
    ids = np.arange(len(df))
    train, held = train_test_split(ids, test_size=0.30, stratify=df.y_mal, random_state=42)
    val, test = train_test_split(held, test_size=0.5, stratify=df.y_mal.iloc[held], random_state=43)
    model, rest = train_test_split(
        val, test_size=2 / 3, stratify=df.y_mal.iloc[val], random_state=44
    )
    cal, policy = train_test_split(
        rest, test_size=0.5, stratify=df.y_mal.iloc[rest], random_state=45
    )
    partitions = {
        "random": {
            "train": train,
            "model": model,
            "calibration": cal,
            "policy": policy,
            "test": test,
        }
    }
    tr, he = next(
        GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(ids, groups=df.domain)
    )
    va_rel, te_rel = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43).split(
            he, groups=df.domain.iloc[he]
        )
    )
    va = he[va_rel]
    te = he[te_rel]
    mo_rel, re_rel = next(
        GroupShuffleSplit(n_splits=1, train_size=1 / 3, random_state=44).split(
            va, groups=df.domain.iloc[va]
        )
    )
    re_idx = va[re_rel]
    ca_rel, po_rel = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=45).split(
            re_idx, groups=df.domain.iloc[re_idx]
        )
    )
    partitions["domain"] = {
        "train": tr,
        "model": va[mo_rel],
        "calibration": re_idx[ca_rel],
        "policy": re_idx[po_rel],
        "test": te,
    }
    return partitions


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "protocol.json").exists():
        (ROOT / "protocol.json").write_bytes((CODE_DIR / "protocol.json").read_bytes())
    check_normalizer()
    out = ROOT / "data-before-token-audit"
    out.mkdir(exist_ok=False)
    frames, hashes = [], {}
    for name, expected in EXPECTED.items():
        p = SOURCE / "data" / "dataset" / name
        hashes[name] = sha(p)
        if hashes[name] != expected:
            raise ValueError("source_hash_changed:" + name)
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not set(df.type).issubset({"0", "1"}):
            raise ValueError("unknown_label")
        df["y_mal"] = 1 - df.type.astype(int)
        df["source_id"] = [f"{name}:{i + 2}" for i in range(len(df))]
        frames.append(df[["url", "y_mal", "source_id"]])
    df = pd.concat(frames, ignore_index=True)
    stages = [{"step": "raw", **summarize(df)}]
    raw_conflicts = set(df.groupby("url").y_mal.nunique().loc[lambda x: x > 1].index)
    reasons = Counter()
    canonical, hosts = [], []
    with (out / "excluded.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "reason", "url_sha256", "y_mal"])

        def exclude(indices, reason):
            for row in df.loc[indices].itertuples():
                writer.writerow(
                    [row.source_id, reason, hashlib.sha256(row.url.encode()).hexdigest(), row.y_mal]
                )
            reasons[reason] += len(indices)

        invalid = []
        for i, row in enumerate(df.itertuples()):
            try:
                key, host = normalize(row.url)
            except (ValueError, UnicodeError) as exc:
                key = host = ""
                invalid.append(i)
                reason = (
                    str(exc)
                    if str(exc)
                    in (
                        "empty_url",
                        "control_or_format_character",
                        "unsupported_scheme",
                        "invalid_host",
                    )
                    else "parse_error"
                )
                writer.writerow(
                    [row.source_id, reason, hashlib.sha256(row.url.encode()).hexdigest(), row.y_mal]
                )
                reasons[reason] += 1
            canonical.append(key)
            hosts.append(host)
        df["url_key"] = canonical
        df["host"] = hosts
        df = df.drop(index=invalid)
        stages.append({"step": "invalid_and_control_removed", **summarize(df)})
        conflicts = set(df.groupby("url_key").y_mal.nunique().loc[lambda x: x > 1].index)
        idx = df.index[df.url_key.isin(conflicts) | df.url.isin(raw_conflicts)]
        exclude(idx, "conflicting_label")
        df = df.drop(index=idx)
        stages.append({"step": "raw_and_normalized_conflicts_removed", **summarize(df)})
        idx = df.index[df.duplicated("url_key")]
        exclude(idx, "duplicate_normalized_url")
        df = df.drop(index=idx)
        stages.append({"step": "normalized_duplicates_removed", **summarize(df)})
        # Inputs are canonical strings; remove indistinguishable full model inputs
        # before splitting. The GRU sees the first 50 character trigrams (52 chars).
        numeric = []
        texts = []
        reps = []
        invalid = []
        for i, row in df.iterrows():
            try:
                features = extract_features(row.url_key)
                text = preprocess_link(row.url_key)
                if not text:
                    raise ValueError("empty_model_text")
            except (ValueError, TypeError, re.error):
                invalid.append(i)
                continue
            numeric.append(features)
            texts.append(text)
            reps.append(
                hashlib.sha256((text[:52] + "|" + json.dumps(features)).encode()).hexdigest()
            )
        exclude(invalid, "unusable_model_input")
        df = df.drop(index=invalid)
        df["representation_key"] = reps
        x = np.asarray(numeric, dtype=np.float32)
        df["feature_index"] = np.arange(len(df))
        rep_conflicts = set(
            df.groupby("representation_key").y_mal.nunique().loc[lambda x: x > 1].index
        )
        idx = df.index[df.representation_key.isin(rep_conflicts)]
        exclude(idx, "conflicting_model_representation")
        df = df.drop(index=idx)
        stages.append({"step": "model_representation_conflicts_removed", **summarize(df)})
        idx = df.index[df.duplicated("representation_key")]
        exclude(idx, "duplicate_model_representation")
        df = df.drop(index=idx)
        stages.append({"step": "model_representation_duplicates_removed", **summarize(df)})
    extract = tldextract.TLDExtract(
        suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
    )
    domain_cache = {}
    for host in set(df.host):
        e = extract(host)
        domain_cache[host] = (e.domain + "." + e.suffix) if e.suffix else host
    df["domain"] = df.host.map(domain_cache)
    df["id"] = df.url_key.map(lambda u: hashlib.sha256(u.encode()).hexdigest())
    df = df.sort_values("id").reset_index(drop=True)
    x = x[df.feature_index.to_numpy()]
    df["text"] = df.url_key.map(preprocess_link)
    df = df.drop(columns="feature_index")
    n = len(df)
    partitions = make_partitions(df)
    summaries = {}
    audit = {}
    for kind, parts in partitions.items():
        summaries[kind] = {}
        audit[kind] = {}
        assignment = np.full(n, "", dtype=object)
        for name, idx in parts.items():
            assignment[idx] = name
            summaries[kind][name] = {
                **summarize(df.iloc[idx]),
                "fraction": len(idx) / n,
                "domains": df.domain.iloc[idx].nunique(),
            }
            if set(df.y_mal.iloc[idx]) != {0, 1}:
                raise ValueError("class_missing")
        assert np.all(assignment != "")
        df[kind + "_split"] = assignment
        names = list(parts)
        for a_i, a in enumerate(names):
            for b in names[a_i + 1 :]:
                cols = ["id", "url_key", "representation_key"] + (
                    ["domain"] if kind == "domain" else []
                )
                pair = {c: len(set(df[c].iloc[parts[a]]) & set(df[c].iloc[parts[b]])) for c in cols}
                if any(pair.values()):
                    raise ValueError("split_overlap")
                audit[kind][a + "|" + b] = pair
        np.savez(out / (kind + "_indices.npz"), **parts)
    df.to_csv(out / "clean.csv", index=False)
    np.save(out / "features.npy", x)
    manifest = {
        "source_sha256": hashes,
        "raw_conflicting_urls": len(raw_conflicts),
        "normalized_conflicting_urls": len(conflicts),
        "representation_conflicts": len(rep_conflicts),
        "stages": stages,
        "exclusion_reasons": dict(reasons),
        "final": summarize(df),
        "splits": summaries,
        "overlap_audit": audit,
        "feature_names": FEATURE_NAMES,
        "seed": 42,
        "labels": {"malicious": 1, "benign": 0},
        "normalizer_tests": "PASS",
        "domain_method": "GroupShuffleSplit, offline bundled PSL incl private suffixes; actual row proportions reported",
        "label_limit": "Conflicts excluded without relabeling. Remaining source labels have not been independently adjudicated.",
        "files": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        "prepare_sha256": sha(Path(__file__)),
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {"stages": stages, "exclusion_reasons": dict(reasons), "splits": summaries}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
