"""Independent readback of saved rows, classes, hashes and encoded overlaps."""

import hashlib
import json
import numpy as np
import pandas as pd
from prepare import ROOT, sha, write_json, normalize


def main():
    p = ROOT / "data-final"
    m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in m["files"].items():
        assert sha(p / name) == digest, name
    d = pd.read_csv(p / "clean.csv", keep_default_na=False)
    x = np.load(p / "features.npy")
    assert len(d) == m["final"]["rows"] == len(x)
    assert (d.y_mal == 1).sum() == m["final"]["malicious"]
    assert (d.y_mal == 0).sum() == m["final"]["benign"]
    assert d.id.is_unique and d.url_key.is_unique
    assert all(normalize(u)[0] == u for u in d.url_key)
    checks = {}
    for kind in ("random", "domain"):
        z = np.load(p / (kind + "-encoded.npy"))
        a = np.load(p / (kind + "_indices.npz"))
        all_idx = np.concatenate([a[name] for name in a.files])
        assert len(all_idx) == len(d)
        assert np.array_equal(np.sort(all_idx), np.arange(len(d)))
        hashes = [hashlib.sha256(i.tobytes() + j.tobytes()).digest() for i, j in zip(z, x)]
        assert len(set(hashes)) == len(d), "encoded_duplicate_" + kind
        for name in a.files:
            assert set(d.y_mal.iloc[a[name]]) == {0, 1}
            assert (d[kind + "_split"].iloc[a[name]] == name).all()
        if kind == "domain":
            names = list(a.files)
            for i, one in enumerate(names):
                for two in names[i + 1 :]:
                    assert not set(d.domain.iloc[a[one]]) & set(d.domain.iloc[a[two]])
        checks[kind] = {
            "partition_cover": "PASS",
            "both_classes": "PASS",
            "encoded_unique": "PASS",
            "rows": len(d),
        }
        validation = np.sort(np.concatenate([a[k] for k in ("model", "calibration", "policy")]))
        assert len(validation) == len(np.unique(validation))
        assert not np.intersect1d(validation, a["train"]).size
        assert not np.intersect1d(validation, a["test"]).size
        checks[kind]["shared_validation"] = {
            "rows": len(validation),
            "benign": int((d.y_mal.iloc[validation] == 0).sum()),
            "malicious": int((d.y_mal.iloc[validation] == 1).sum()),
            "disjoint_from_train_and_test": "PASS",
        }
    r = m["splits"]["random"]
    n = len(d)
    assert abs(r["train"]["rows"] / n - 0.7) < 2 / n
    assert abs(r["test"]["rows"] / n - 0.15) < 2 / n
    assert abs(sum(r[k]["rows"] for k in ("model", "calibration", "policy")) / n - 0.15) < 2 / n
    write_json(
        ROOT / "data-audit.json",
        {
            "status": "PASS",
            "final": m["final"],
            "checks": checks,
            "manifest_sha256": sha(p / "manifest.json"),
            "audit_sha256": sha(__file__),
            "label_semantics": "source conflicts removed; independent relabeling NOT_RUN",
        },
    )
    print("DATA_AUDIT PASS", m["final"], flush=True)


if __name__ == "__main__":
    main()
