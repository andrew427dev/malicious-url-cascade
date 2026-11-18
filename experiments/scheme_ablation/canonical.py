"""Mitigation: canonicalize the URL format before feature extraction, then rerun experiment.py.

canonical(u) = "http://" + u without its leading scheme. Every URL keeps a parsable host
when possible. Exclude all rows in encoded-input groups that cross splits or labels.
Reindex retained rows, preserve their assigned roles, and report their new denominators.

usage: canonical.py DEST  ->  URLCASCADE_RUN_DIR=DEST experiment.py random
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean_rerun"))
from common import ROOT, SOURCE, sha, write_json  # noqa: E402

sys.path.insert(0, str(SOURCE))
from urlcascade.features import extract_features  # noqa: E402
from run import strip_scheme  # noqa: E402
from prepare import summarize  # noqa: E402
from verify_inputs import KINDS, checked_parts, verified_rows  # noqa: E402


def canonical(url):
    return "http://" + strip_scheme(url)


def features(url):
    # Upstream features.py compiles the hostname as a regex; 4 of 717,793 canonical URLs
    # (backslash paths) yield an invalid pattern. They fall back to the scheme-free form,
    # which still depends only on strip_scheme(url), so format invariance holds.
    try:
        return extract_features(canonical(url))
    except re.error:
        return extract_features(strip_scheme(url))


def self_test():
    forms = ("https://a.example/x", "HTTP://a.example/x", "a.example/x")
    assert {canonical(u) for u in forms} == {"http://a.example/x"}
    assert canonical(canonical("a.example/x")) == canonical("a.example/x")
    assert extract_features(canonical("a.example/x"))[14] > 0
    odd = r"\\198.51.100.7\share\x.exe"  # canonical hostname is an invalid regex
    assert features(odd) == features("https://" + odd)


def build(src, out, df):
    parent_manifest = src / "manifest.json"
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    x = np.asarray([features(u) for u in df.url_key], dtype=np.float32)
    enc = {k: np.load(src / f"{k}-encoded.npy", mmap_mode="r") for k in KINDS}
    # Rows that differed only by scheme now share one model input. experiment.py refuses
    # inputs spanning splits or labels, so every row of such a group is excluded (same rule
    # PR #3 applied to conflicting URLs). Same-split same-label duplicates are kept.
    keep = np.ones(len(df), bool)
    for kind, e in enc.items():
        key = [hashlib.sha256(e[i].tobytes() + x[i].tobytes()).digest() for i in range(len(df))]
        g = pd.DataFrame({"key": key, "split": df[kind + "_split"], "label": df.y_mal})
        agg = g.groupby("key").agg(s=("split", "nunique"), l=("label", "nunique"))
        bad = set(agg.index[(agg.s > 1) | (agg.l > 1)])
        keep &= ~np.array([k in bad for k in key])  # pandas isin misses some bytes keys
    kept = df.loc[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError("no_rows_after_canonicalization")
    kept.to_csv(out / "clean.csv", index=False)
    np.save(out / "features.npy", x[keep])
    summaries = {}
    for kind, e in enc.items():
        np.save(out / f"{kind}-encoded.npy", e[keep])
        parts = {p: np.flatnonzero(kept[kind + "_split"] == p) for p in checked_parts(src, df, kind)}
        np.savez(out / f"{kind}_indices.npz", **parts)
        summaries[kind] = {}
        for role, indices in checked_parts(out, kept, kind).items():
            group = kept.iloc[indices]
            summaries[kind][role] = {
                **summarize(group), "fraction": len(group) / len(kept),
                "domains": int(group.domain.nunique()),
            }
    for name in manifest["files"]:
        if not (out / name).exists():
            shutil.copy(src / name, out / name)
        manifest["files"][name] = sha(out / name)
    dropped = df[~keep]
    manifest["parent_data_manifest_sha256"] = sha(parent_manifest)
    manifest["finalization_method"] = (
        "Canonicalize numeric features, exclude entire groups crossing labels or splits, "
        "retain the parent vocabularies and the retained rows' split assignments."
    )
    manifest["canonicalization_code_sha256"] = sha(__file__)
    manifest["final"] = summarize(kept)
    manifest["splits"] = summaries
    manifest["stages"].append({"step": "canonical_collision_exclusion", **manifest["final"]})
    manifest["canonical_collision_exclusion"] = {
        "rows": int((~keep).sum()),
        "malicious": int(dropped.y_mal.sum()),
        "by_random_split": dropped.random_split.value_counts().to_dict(),
        "remaining_rows": int(keep.sum()),
    }
    print(json.dumps(manifest["canonical_collision_exclusion"]))
    write_json(out / "manifest.json", manifest)


def main(dest):
    self_test()
    src, dest = ROOT / "data-final", Path(dest)
    df, _ = verified_rows(src)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "data-final"
    if target.exists():
        raise FileExistsError(target)
    with TemporaryDirectory(dir=dest) as scratch:
        out = Path(scratch) / "data-final"
        out.mkdir()
        build(src, out, df)
        if (ROOT / "protocol.json").exists():
            protocol = dest / "protocol.json"
            if protocol.exists() and sha(protocol) != sha(ROOT / "protocol.json"):
                raise ValueError("destination_protocol_mismatch")
            if not protocol.exists():
                shutil.copy(ROOT / "protocol.json", protocol)
        out.rename(target)
    print("CANONICAL", target)


if __name__ == "__main__":
    main(sys.argv[1])
