"""Assign every distinct raw URL before applying any exclusion rule."""
import hashlib
from urllib.parse import urlsplit

import tldextract
from inputs import ROOT, SOURCE, REPO, pd, np, read_frames, sha, counts, write_json
from common import validation_indices
from prepare import normalize, make_partitions
import json


def host_for_group(raw):
    try:
        return normalize(raw)[1]
    except (ValueError, UnicodeError):
        try:
            return (urlsplit(raw if "://" in raw or raw.startswith("//") else "//" + raw).hostname or "").lower()
        except (ValueError, UnicodeError):
            return ""


def main():
    frames = read_frames()
    raw = pd.concat(frames)
    assert raw.url.notna().all()
    group = raw.groupby("url", sort=False).type
    first = group.first()
    ambiguous = group.nunique().gt(1)
    keys = pd.DataFrame({"y_mal": (1 - first).mask(ambiguous, 2)})
    keys["raw_hash"] = [hashlib.sha256(u.encode()).hexdigest() for u in keys.index]
    keys = keys.sort_values("raw_hash")
    assert keys.raw_hash.is_unique
    extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
    hosts = [host_for_group(u) for u in keys.index]
    domains = {}
    for h in set(hosts):
        e = extract(h)
        domains[h] = e.domain + "." + e.suffix if e.suffix else h
    keys["domain"] = [domains[h] for h in hosts]
    for split, parts in make_partitions(keys).items():
        role = np.empty(len(keys), dtype=object)
        position = np.full(len(keys), -1)
        for name, idx in (("train",parts["train"]), ("validation",validation_indices(parts)), ("test",parts["test"])):
            assert (position[idx] == -1).all()
            role[idx] = name
            position[idx] = np.arange(len(idx))
        assert (position >= 0).all()
        keys[split+"_role"] = role
        keys[split+"_order"] = position
    assert keys.groupby("domain")["domain_role"].nunique().max() == 1
    keys.to_pickle(ROOT / "assignment.pkl")
    keys.drop(columns="y_mal").to_csv(ROOT / "assignment.csv.gz")
    cfg = json.loads((REPO/"configs/upstream.json").read_text())
    write_json(ROOT/"source-manifest.json",{
        "upstream_commit":cfg["commit"],"source_code":cfg["source_sha256"],
        "files":{name: {**rec,"observed_sha256":sha(SOURCE/name),"counts":counts(f)} for (name,rec),f in zip([(k,v) for k,v in cfg["files"].items() if k.endswith(".csv")],frames)},
        "raw": counts(raw), "unique_urls":len(keys),"ambiguous_urls":int(ambiguous.sum()),
        "ambiguous_rows":int(raw.url.isin(ambiguous[ambiguous].index).sum()),
        "identical_label_repeats_over_all_rows":int(raw.duplicated(["url","type"]).sum()),
        "assignment_sha256":sha(ROOT/"assignment.csv.gz"), "missing_host_unique_urls": hosts.count("")
    })


if __name__=="__main__":
    main()
