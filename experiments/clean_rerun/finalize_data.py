"""Quarantine unstable URL normalization, refit vocabularies on retained train rows."""

import os

for name in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import hashlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from prepare import ROOT, normalize, sha, summarize, write_json
from urlcascade.adapters import vectorizer


def main():
    before = ROOT / "data"
    out = ROOT / "data-final"
    out.mkdir(exist_ok=False)
    m = json.loads((before / "manifest.json").read_text(encoding="utf-8"))
    d = pd.read_csv(before / "clean.csv", keep_default_na=False)
    x = np.load(before / "features.npy")
    stable = np.array([normalize(u)[0] == u for u in d.url_key])
    rejected = d.loc[~stable, ["id", "source_id", "y_mal"]].copy()
    rejected["reason"] = "non_idempotent_normalization"
    rejected.to_csv(out / "normalization-quarantine.csv", index=False)
    d = d.loc[stable].reset_index(drop=True)
    x = x[stable]
    summaries = {}
    for kind in ("random", "domain"):
        parts = {
            name: np.flatnonzero(d[kind + "_split"].to_numpy() == name)
            for name in ("train", "model", "calibration", "policy", "test")
        }
        tf.keras.utils.set_random_seed(42)
        v = vectorizer()
        v.adapt(
            tf.data.Dataset.from_tensor_slices(d.text.iloc[parts["train"]].tolist()).batch(2048)
        )
        z = np.concatenate(
            [
                v(tf.constant(d.text.iloc[i : i + 4096].tolist())).numpy().astype("int32")
                for i in range(0, len(d), 4096)
            ]
        )
        keys = [hashlib.sha256(a.tobytes() + b.tobytes()).digest() for a, b in zip(z, x)]
        if len(set(keys)) != len(d):
            raise RuntimeError("final_vocabulary_collision")
        np.save(out / (kind + "-encoded.npy"), z)
        np.savez(out / (kind + "_indices.npz"), **parts)
        write_json(out / (kind + "-vocabulary.json"), v.get_vocabulary())
        summaries[kind] = {
            name: {
                **summarize(d.iloc[idx]),
                "fraction": len(idx) / len(d),
                "domains": d.domain.iloc[idx].nunique(),
            }
            for name, idx in parts.items()
        }
        print("FINAL_INPUT_AUDIT", kind, "PASS", flush=True)
    d.to_csv(out / "clean.csv", index=False)
    np.save(out / "features.npy", x)
    m.update(
        final=summarize(d),
        splits=summaries,
        normalization_quarantine_rows=len(rejected),
        finalization_method="Exclude non-idempotent normalization; retain existing assignments; refit each vocabulary using remaining train rows only; require all exact encoded inputs unique.",
        previous_token_manifest_sha256=sha(before / "manifest.json"),
        finalize_sha256=sha(__file__),
    )
    m["stages"].append({"step": "unstable_normalization_quarantine", **summarize(d)})
    m["files"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "manifest.json", m)
    print("FINAL", m["final"], flush=True)


if __name__ == "__main__":
    main()
