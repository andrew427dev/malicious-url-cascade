"""Resolve vocabulary-induced indistinguishable inputs before final training."""

import os

for name in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from prepare import ROOT, make_partitions, summarize, sha, write_json
from urlcascade.adapters import vectorizer


def main():
    before = ROOT / "data-before-token-audit"
    out = ROOT / "data"
    out.mkdir(exist_ok=False)
    df = pd.read_csv(before / "clean.csv", keep_default_na=False)
    x = np.load(before / "features.npy")
    original = json.loads((before / "manifest.json").read_text(encoding="utf-8"))
    rounds = []
    with (out / "encoded-excluded.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "id", "reason", "y_mal"])
        for iteration in range(1, 21):
            parts = make_partitions(df)
            parent = np.arange(len(df))
            touched = set()
            audits = {}
            vocabularies = {}
            encoded_by_kind = {}

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(a, b):
                a, b = find(a), find(b)
                if a != b:
                    parent[max(a, b)] = min(a, b)

            for kind, p in parts.items():
                tf.keras.utils.set_random_seed(42)
                layer = vectorizer()
                layer.adapt(
                    tf.data.Dataset.from_tensor_slices(df.text.iloc[p["train"]].tolist()).batch(
                        2048
                    )
                )
                vocabularies[kind] = layer.get_vocabulary()
                z = np.concatenate(
                    [
                        layer(tf.constant(df.text.iloc[i : i + 4096].tolist()))
                        .numpy()
                        .astype("int32")
                        for i in range(0, len(df), 4096)
                    ]
                )
                encoded_by_kind[kind] = z
                seen = {}
                duplicates = 0
                for i, (a, b) in enumerate(zip(z, x)):
                    key = hashlib.sha256(a.tobytes() + b.tobytes()).digest()
                    if key in seen:
                        j = seen[key]
                        union(i, j)
                        touched.update((i, j))
                        duplicates += 1
                    else:
                        seen[key] = i
                audits[kind] = {
                    "duplicate_rows": duplicates,
                    "vocabulary_size": len(vocabularies[kind]),
                }
                print("INPUT_AUDIT", iteration, kind, audits[kind], flush=True)
                del seen
            if not touched:
                rounds.append(
                    {
                        "round": iteration,
                        "before": summarize(df),
                        "audits": audits,
                        "removed_conflict_rows": 0,
                        "removed_duplicate_rows": 0,
                    }
                )
                for kind in parts:
                    write_json(out / (kind + "-vocabulary.json"), vocabularies[kind])
                    np.save(out / (kind + "-encoded.npy"), encoded_by_kind[kind])
                break
            groups = defaultdict(list)
            for i in touched:
                groups[find(i)].append(i)
            remove = set()
            conflict_count = duplicate_count = 0
            for members in groups.values():
                members = sorted(members)
                if len(set(df.y_mal.iloc[members])) > 1:
                    excluded = members
                    reason = "encoded_label_conflict"
                    conflict_count += len(excluded)
                else:
                    excluded = members[1:]
                    reason = "encoded_duplicate"
                    duplicate_count += len(excluded)
                for i in excluded:
                    writer.writerow([iteration, df.id.iloc[i], reason, int(df.y_mal.iloc[i])])
                remove.update(excluded)
            rounds.append(
                {
                    "round": iteration,
                    "before": summarize(df),
                    "audits": audits,
                    "removed_conflict_rows": conflict_count,
                    "removed_duplicate_rows": duplicate_count,
                }
            )
            keep = np.array([i for i in range(len(df)) if i not in remove])
            df = df.iloc[keep].reset_index(drop=True)
            x = x[keep]
            write_json(out / "refinement-progress.json", rounds)
        else:
            raise RuntimeError("input_dedup_did_not_converge")
    summaries = {}
    overlaps = {}
    for kind, p in parts.items():
        assignment = np.full(len(df), "", dtype=object)
        summaries[kind] = {}
        overlaps[kind] = {}
        for name, idx in p.items():
            assignment[idx] = name
            if set(df.y_mal.iloc[idx]) != {0, 1}:
                raise ValueError("class_missing")
            summaries[kind][name] = {
                **summarize(df.iloc[idx]),
                "fraction": len(idx) / len(df),
                "domains": df.domain.iloc[idx].nunique(),
            }
        assert np.all(assignment != "")
        df[kind + "_split"] = assignment
        names = list(p)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                cols = ["id", "url_key", "representation_key"] + (
                    ["domain"] if kind == "domain" else []
                )
                pair = {c: len(set(df[c].iloc[p[a]]) & set(df[c].iloc[p[b]])) for c in cols}
                assert not any(pair.values())
                overlaps[kind][a + "|" + b] = pair
        np.savez(out / (kind + "_indices.npz"), **p)
    df.to_csv(out / "clean.csv", index=False)
    np.save(out / "features.npy", x)
    original.update(
        final=summarize(df),
        splits=summaries,
        overlap_audit=overlaps,
        encoded_refinement=rounds,
        encoded_refinement_method="Union exact encoded+numeric collisions across both training-only vocabularies; remove conflicting components entirely, retain lowest ID for equal-label components; re-split/refit until zero collisions, before classifier training.",
        previous_manifest_sha256=sha(before / "manifest.json"),
        refine_sha256=sha(__file__),
    )
    original["stages"].append({"step": "encoded_input_deduplication", **summarize(df)})
    original["files"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "manifest.json", original)
    print("FINAL_DATA", json.dumps(original["final"]), flush=True)


if __name__ == "__main__":
    main()
