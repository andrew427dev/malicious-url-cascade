"""Mitigation: canonicalize the URL format before feature extraction, then rerun experiment.py.

canonical(u) = "http://" + u without its leading scheme. Every URL keeps a parsable host
(hostname_length > 0) and the scheme carries no signal, so writing a link with or without
http(s):// gives identical inputs. Builds a new run directory with the same rows, splits,
encodings and vocabularies as ROOT; only features.npy and its manifest digest change.

usage: canonical.py DEST  ->  URLCASCADE_RUN_DIR=DEST experiment.py random
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean_rerun"))
from common import ROOT, SOURCE, sha, write_json  # noqa: E402

sys.path.insert(0, str(SOURCE))
from urlcascade.features import extract_features  # noqa: E402
from run import strip_scheme  # noqa: E402


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


def main(dest):
    self_test()
    src, dest = ROOT / "data-final", Path(dest)
    (dest / "data-final").mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "source-snapshot.json"):
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, dest / name)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(src / "clean.csv", keep_default_na=False)
    x = np.asarray([features(u) for u in df.url_key], dtype=np.float32)
    enc = {k: np.load(src / f"{k}-encoded.npy") for k in ("random", "domain")}
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
    out = dest / "data-final"
    new = np.cumsum(keep) - 1
    df[keep].to_csv(out / "clean.csv", index=False)
    np.save(out / "features.npy", x[keep])
    for kind, e in enc.items():
        np.save(out / f"{kind}-encoded.npy", e[keep])
        parts = np.load(src / f"{kind}_indices.npz")
        np.savez(out / f"{kind}_indices.npz", **{p: new[a[keep[a]]] for p, a in parts.items()})
    for name in manifest["files"]:
        if not (out / name).exists():
            shutil.copy(src / name, out / name)
        manifest["files"][name] = sha(out / name)
    dropped = df[~keep]
    manifest["canonicalization"] = __doc__.split("\n\n")[1]
    manifest["canonical_collision_exclusion"] = {
        "rows": int((~keep).sum()),
        "malicious": int(dropped.y_mal.sum()),
        "by_random_split": dropped.random_split.value_counts().to_dict(),
        "remaining_rows": int(keep.sum()),
    }
    print(json.dumps(manifest["canonical_collision_exclusion"]))
    x = x[keep]
    write_json(dest / "data-final" / "manifest.json", manifest)
    print("CANONICAL", dest, x.shape)


if __name__ == "__main__":
    main(sys.argv[1])
