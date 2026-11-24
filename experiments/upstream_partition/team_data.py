"""Fresh team splits, with original source IDs and per-partition first-row selection."""

import argparse
import hashlib
import json
from urllib.parse import urlsplit

from inputs import ROOT, read_frames, save_parts, counts, write_json, pd
from prepare import make_partitions
from common import validation_indices
import tldextract


def pool():
    (ROOT / "pool-started").touch(exist_ok=False)
    frames = read_frames()
    df = pd.concat(frames)
    audit = {
        "sources": {f.index[0].split(":")[0]: counts(f) for f in frames},
        "combined": counts(df),
    }
    df = df.dropna()
    audit["nonmissing"] = counts(df)
    conflicts = df.groupby("url", sort=False).type.nunique()
    keys = conflicts[conflicts > 1].index
    bad = df.url.isin(keys)
    audit["exact_conflicting_urls"] = len(keys)
    audit["conflict_rows"] = counts(df[bad])
    df = df[~bad]
    audit["after_conflicts"] = counts(df)
    df = df.drop_duplicates(["url", "type"]).copy()
    audit["after_exact_duplicates"] = counts(df)
    extract = tldextract.TLDExtract(
        suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
    )

    def hostname(url):
        import re

        explicit = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", url)
        parsed = urlsplit(url if explicit or url.startswith("//") else "//" + url)
        return (parsed.hostname or "").lower()

    from prepare import normalize

    hosts = []
    for url in df.url:
        try:
            host = normalize(url)[1]
        except (ValueError, UnicodeError):
            # Preserve upstream-compatible rows; only the group key uses parsing.
            try:
                host = hostname(url)
            except ValueError:
                host = ""
        hosts.append(host)
    audit["missing_host_group_rows"] = hosts.count("")
    domains = {}
    for host in set(hosts):
        item = extract(host)
        domains[host] = (item.domain + "." + item.suffix) if item.suffix else host
    df["domain"] = [domains[h] for h in hosts]
    df["y_mal"] = 1 - df.type
    df["raw_hash"] = [hashlib.sha256(u.encode()).hexdigest() for u in df.url]
    df = df.sort_values("raw_hash")
    splits = make_partitions(df)
    for kind, parts in splits.items():
        df[kind + "_role"] = ""
        for role, idx in (
            ("train", parts["train"]),
            ("validation", validation_indices(parts)),
            ("test", parts["test"]),
        ):
            # Keep the historical split's role-specific order, including first-row identity.
            df.loc[df.index[idx], kind + "_role"] = role
            pd.Series(df.index[idx], name="id").to_csv(
                ROOT / (kind + "-" + role + "-assigned.csv.gz"), index=False
            )
    df.to_pickle(ROOT / "team-pool.pkl")
    write_json(ROOT / "team-source.json", audit)


def split(kind):
    df = pd.read_pickle(ROOT / "team-pool.pkl")
    group, form = kind.split("-")
    parts = {}
    for name in ("train", "validation", "test"):
        ids = pd.read_csv(ROOT / (group + "-" + name + "-assigned.csv.gz")).id
        parts[name] = df.loc[ids, ["url", "type", "domain"]]
        assert (df.loc[ids, group + "_role"] == name).all()
    save_parts(ROOT / kind, parts, form == "stripped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "pool",
            "random-original",
            "random-stripped",
            "domain-original",
            "domain-stripped",
        ],
    )
    stage = parser.parse_args().stage
    assert json.loads((ROOT / "parity.json").read_text())["status"] == "PASS"
    if stage == "pool":
        pool()
    else:
        split(stage)
