"""Mitigation: canonicalize the URL format before feature extraction, then rerun experiment.py.

canonical(u) = "http://" + u without its leading scheme. Every URL keeps a parsable host
(hostname_length > 0) and the scheme carries no signal, so writing a link with or without
http(s):// gives identical inputs. Builds a new run directory with the same rows, splits,
encodings and vocabularies as ROOT; only features.npy and its manifest digest change.

usage: canonical.py DEST  ->  URLCASCADE_RUN_DIR=DEST experiment.py random
"""

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
    for name in manifest["files"]:
        if name != "features.npy":
            shutil.copy(src / name, dest / "data-final" / name)
    urls = pd.read_csv(src / "clean.csv", keep_default_na=False).url_key
    x = np.asarray([features(u) for u in urls], dtype=np.float32)
    np.save(dest / "data-final" / "features.npy", x)
    manifest["files"]["features.npy"] = sha(dest / "data-final" / "features.npy")
    manifest["canonicalization"] = __doc__.split("\n\n")[1]
    write_json(dest / "data-final" / "manifest.json", manifest)
    print("CANONICAL", dest, x.shape)


if __name__ == "__main__":
    main(sys.argv[1])
