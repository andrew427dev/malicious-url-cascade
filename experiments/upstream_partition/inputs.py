"""Input contract for the pinned notebook and the team's partition experiments."""

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

for key in (
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["URLCASCADE_RUN_DIR"]).resolve()
SOURCE = ROOT / "source"
sys.path[:0] = [str(REPO), str(SOURCE), str(REPO / "experiments/clean_rerun")]
from common import sha, write_json
from urlcascade.adapters import preprocess_link, vectorizer
from urlcascade.features import extract_features, FEATURE_NAMES
from Utils import DataProcessing, FeatureCreation

FILES = ("df_final.csv", "all_top_domains_merged.csv", "df_another_validation.csv")
PREFIX = re.compile(r"^https?://", re.I)


def read_frames():
    config = json.loads((REPO / "configs/upstream.json").read_text())
    for rel, digest in config["source_sha256"].items():
        if sha(SOURCE / rel) != digest:
            raise ValueError("Upstream code digest differs: " + rel)
    frames = []
    for name in FILES:
        path = SOURCE / "dataset" / name
        if sha(path) != config["files"]["dataset/" + name]["sha256"]:
            raise ValueError("CSV digest differs: " + name)
        frame = pd.read_csv(path)
        if list(frame.columns) != ["url", "type"] or not set(frame.type.unique()) <= {
            0,
            1,
        }:
            raise ValueError("Unexpected source schema or label")
        frame.index = pd.Index(
            [f"{name}:{i}" for i in range(2, len(frame) + 2)], name="id"
        )
        frames.append(frame)
    return frames


def counts(frame):
    return {
        "rows": len(frame),
        "malicious": int((frame.type == 0).sum()),
        "benign": int((frame.type == 1).sum()),
    }


def inputs(frame, normalized=False):
    urls = (
        frame.url.str.replace(PREFIX, "", n=1, regex=True) if normalized else frame.url
    )
    # Compute all features before text normalization and first-row selection.
    x = np.empty((len(urls), 18), np.float32)
    for i, url in enumerate(urls):
        try:
            x[i] = extract_features(url)
        except (ValueError, re.error) as exc:
            raise ValueError(f"Unprocessable input at {frame.index[i]}") from exc
    return x, urls.map(preprocess_link)


def prepared(frame, normalized=False):
    x, texts = inputs(frame, normalized)
    keep = ~texts.duplicated()
    return frame.loc[keep].assign(text=texts[keep]), x[keep]


def parity(frames):
    sample = pd.concat([f.sample(256, random_state=42) for f in frames])
    fixture = pd.DataFrame(
        {
            "url": [
                "example.org/a-b",
                "example.org/ab",
                "https://www.Example.org/ a-b",
                "a" * 55 + "x",
                "a" * 55 + "y",
            ],
            "type": [1, 0, 0, 0, 1],
        },
        index=["first", "hyphenless", "multiple", "long_x", "long_y"],
    )
    sample = pd.concat([fixture, sample, sample.iloc[:10]])
    # Use unique row identifiers for independently duplicated sample rows.
    sample.index = np.arange(len(sample))
    audit = []
    for batch in (sample, sample.iloc[::-1]):
        kept, x = prepared(batch)
        ref = FeatureCreation.create_features(batch.copy())
        assert tuple(ref.columns[2:]) == FEATURE_NAMES
        ref.url = ref.url.map(DataProcessing.preprocess_link)
        ref = ref.drop_duplicates("url").dropna()
        assert kept.index.equals(ref.index)
        ds = DataProcessing.process_train_data(batch.copy(), 128)
        ref_text, ref_x, ref_y = zip(
            *[(t.numpy(), n.numpy(), y.numpy()) for (t, n), y in ds]
        )
        layer = vectorizer()
        layer.adapt(ds.map(lambda z, y: z[0]))
        tok = layer(tf.constant(kept.text.tolist())).numpy()
        np.testing.assert_array_equal(
            tok, layer(tf.constant(np.concatenate(ref_text))).numpy()
        )
        upstream_tokens = np.concatenate(
            [
                DataProcessing.vectorize_text(text, numeric, label, layer)[0][0].numpy()
                for (text, numeric), label in ds
            ]
        )
        np.testing.assert_array_equal(tok, upstream_tokens)
        np.testing.assert_array_equal(x, np.concatenate(ref_x))
        np.testing.assert_array_equal(kept.type, np.concatenate(ref_y))
        assert {3, 4} <= set(kept.index)
        np.testing.assert_array_equal(
            tok[kept.index.get_loc(3)], tok[kept.index.get_loc(4)]
        )
        audit.append(
            {
                "input": len(batch),
                "kept": len(kept),
                "hyphen_group_survivor": int(
                    kept.index[kept.text == "exampleorgab"][0]
                ),
                "token_columns": tok.shape[1],
                "numeric_columns": x.shape[1],
            }
        )
    write_json(
        ROOT / "parity.json",
        {"status": "PASS", "orders": audit, "real_sample_rows": 768},
    )


def encode(texts, layer):
    return np.concatenate(
        [
            layer(tf.constant(texts.iloc[i : i + 2048].tolist()))
            .numpy()
            .astype("int32")
            for i in range(0, len(texts), 2048)
        ]
    )


def save_parts(out, parts, normalized=False):
    out.mkdir(exist_ok=False)
    audit = {"stages": {}, "overlaps": {}}
    final = {}
    layer = vectorizer()
    for name, frame in parts.items():
        kept, x = prepared(frame, normalized)
        audit["stages"][name] = {
            "before": counts(frame),
            "after": counts(kept),
            "removed": len(frame) - len(kept),
        }
        if name == "train":
            layer.adapt(
                tf.data.Dataset.from_tensor_slices(kept.text.tolist()).batch(128)
            )
            write_json(out / "vocab.json", layer.get_vocabulary())
        tokens = encode(kept.text, layer)
        np.save(out / (name + "-tokens.npy"), tokens)
        np.save(out / (name + "-features.npy"), x)
        kept.to_pickle(out / (name + ".pkl"))
        kept[["type"]].assign(y_mal=1 - kept.type).to_csv(out / (name + "-rows.csv.gz"))
        final[name] = kept
    for a, b in itertools.combinations(parts, 2):
        cols = ["url", "text"] + (["domain"] if "domain" in final[a] else [])
        audit["overlaps"][a + "/" + b] = {
            c: len(set(final[a][c]) & set(final[b][c])) for c in cols
        }
        if out.name.startswith("domain") and audit["overlaps"][a + "/" + b]["domain"]:
            raise AssertionError("Registration domain overlap")
    write_json(out / "data.json", audit)


def author(frames, final):
    df = pd.concat(frames if final else frames[:2]).sample(frac=1, random_state=42)
    unique = df.drop_duplicates()
    pool = unique.dropna()
    audit = {
        "shuffled": counts(df),
        "exact_row_unique": counts(unique),
        "nonmissing": counts(pool),
        "conflicting_raw_urls": int((df.groupby("url").type.nunique() > 1).sum()),
    }
    if final:
        parts = {"train": pool}
    else:
        from sklearn.model_selection import train_test_split

        tr, va = train_test_split(
            pool, test_size=0.2, stratify=pool.type, random_state=42
        )
        parts = {"train": tr, "validation": va, "external": frames[2]}
    out = ROOT / ("author-final" if final else "author-validation")
    save_parts(out, parts)
    write_json(out / "source-stages.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["parity", "author-validation", "author-final"]
    )
    stage = parser.parse_args().stage
    frames = read_frames()
    if stage == "parity":
        (ROOT / "parity-started").touch(exist_ok=False)
        parity(frames)
    else:
        assert json.loads((ROOT / "parity.json").read_text())["status"] == "PASS"
        author(frames, stage == "author-final")
