"""Train, freeze, and evaluate new random/domain splits without reusing old weights."""

import os

for key, value in {
    "CUDA_VISIBLE_DEVICES": "-1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "PYTHONDONTWRITEBYTECODE": "1",
}.items():
    os.environ[key] = value
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import (
    ROOT,
    SOURCE,
    sha,
    write_json,
    snapshot_sources,
    verify_execution_source,
    validation_indices,
    seal_training_artifacts,
    verify_training_artifacts,
)

sys.path[:0] = [str(SOURCE), str(SOURCE / "upstream")]
from urlcascade.adapters import preprocess_link, vectorizer
from urlcascade.features import extract_features
from Utils.Model import create_model

TAUS = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
RATES = [0, 0.01, 0.05, 0.1, 0.2]


def malicious_scores(benign_scores):
    """Preserve the float32 GRU boundary when converting its benign probability."""
    return 1.0 - np.asarray(benign_scores, dtype=np.float64)


def metrics(y, score, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "pr_auc_ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "fpr": float(fp / (tn + fp)),
    }


def route(p, ids, tau, rate, policy, seed=0):
    mask = p >= tau
    candidates = np.flatnonzero(~mask)
    count = int(np.floor(len(candidates) * rate))
    if count:
        if policy == "boundary":
            ranked = sorted(candidates, key=lambda i: (-float(p[i]), ids[i]))
        elif policy == "random":
            ranked = sorted(
                candidates,
                key=lambda i: (hashlib.sha256(f"{seed}|{ids[i]}".encode()).digest(), ids[i]),
            )
        else:
            raise ValueError("invalid_recheck_policy")
        mask[np.asarray(ranked[:count])] = True
    return mask, count


def cascade_metrics(y, p2, mask, base_mask=None):
    pred = (p2 > 0.5) & mask
    # Operational score: all early benign exits tie at -1, below any routed score.
    score = np.where(mask, p2, -1.0)
    m = metrics(y, score, pred)
    m.update(
        t2_calls=int(mask.sum()),
        t2_call_rate=float(mask.mean()),
        additional_misses=int(np.sum((y == 1) & (p2 > 0.5) & ~mask)),
        recall_loss=float(np.sum((y == 1) & (p2 > 0.5) & ~mask) / np.sum(y == 1)),
    )
    if base_mask is not None:
        added = mask & ~base_mask
        m.update(
            recovered_malicious=int(np.sum(added & (y == 1) & (p2 > 0.5))),
            new_false_positives=int(np.sum(added & (y == 0) & (p2 > 0.5))),
        )
    return m


def self_test():
    p = np.array([0.01, 0.04, 0.049, 0.05, 0.2])
    ids = np.array(["a", "b", "c", "d", "e"])
    mask, k = route(p, ids, 0.05, 1 / 3, "boundary")
    assert k == 1 and mask.tolist() == [False, False, True, True, True]
    r, k = route(p, ids, 0.05, 1 / 3, "random")
    assert r.sum() == mask.sum() and np.all(r[p >= 0.05])
    y = np.array([1, 0, 1, 0, 1])
    p2 = np.array([0.7, 0.8, 0.6, 0.5, 0.9])
    basic, _ = route(p, ids, 0.05, 0, "basic")
    m = cascade_metrics(y, p2, mask, basic)
    assert m["recovered_malicious"] == 1 and m["new_false_positives"] == 0
    assert m["tp"] == 2 and m["fp"] == 0 and m["additional_misses"] == 1


def train(kind):
    self_test()
    tf.keras.utils.set_random_seed(42)
    out = ROOT / kind
    out.mkdir(exist_ok=False)
    snapshot_sources()
    data = ROOT / "data-final"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha(data / name) != digest:
            raise ValueError("data_hash_changed:" + name)
    df = pd.read_csv(data / "clean.csv", keep_default_na=False)
    x = np.load(data / "features.npy")
    y = df.y_mal.to_numpy()
    indices = np.load(data / (kind + "_indices.npz"))
    parts = {name: indices[name] for name in indices.files}
    tr = parts["train"]
    va = ca = po = validation_indices(parts)
    settings = {
        "kind": kind,
        "seed": 42,
        "epochs": 20,
        "batch_size": 128,
        "threads": 1,
        "taus": TAUS,
        "rates": RATES,
        "main_recheck_rate": 0.05,
        "random_seeds": list(range(30)),
        "timing_sample": 1000,
        "timing_repeats": 5,
        "selection": "maximum early exits subject to policy-validation recall loss <= 0.01; ties lower tau",
        "validation_roles": {
            "model": "shared full validation: hyperparameters and GRU checkpoint",
            "calibration": "same full validation: sigmoid calibration",
            "policy": "same full validation: tau selection",
        },
        "validation_rows": len(va),
        "validation_indices_sha256": hashlib.sha256(va.tobytes()).hexdigest(),
        "protocol": json.loads((ROOT / "protocol.json").read_text(encoding="utf-8")),
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "sklearn": sklearn.__version__,
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "platform": platform.platform(),
        "data_manifest_sha256": sha(data / "manifest.json"),
        "experiment_sha256": sha(Path(__file__)),
        "source_files_sha256": {
            name: sha(SOURCE / name)
            for name in (
                "urlcascade/adapters.py",
                "urlcascade/features.py",
                "upstream/Utils/Model.py",
            )
        },
        "self_tests": "PASS",
        "test_used_for_selection": False,
    }
    write_json(out / "settings.json", settings)
    candidates = {}
    models = {}
    calibration = {}
    for family in ("logistic", "boost"):
        entries = []
        fitted = []
        for setting in [0.1, 1.0, 10.0] if family == "logistic" else [7, 15, 31]:
            if family == "logistic":
                estimator = make_pipeline(
                    StandardScaler(), LogisticRegression(C=setting, max_iter=5000, random_state=42)
                )
            else:
                estimator = HistGradientBoostingClassifier(
                    max_iter=60,
                    max_leaf_nodes=setting,
                    learning_rate=0.1,
                    early_stopping=False,
                    random_state=42,
                )
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                estimator.fit(x[tr], y[tr])
            if any(issubclass(w.category, ConvergenceWarning) for w in ws):
                raise RuntimeError("convergence_failure")
            ap = float(average_precision_score(y[va], estimator.predict_proba(x[va])[:, 1]))
            entries.append(
                {
                    "setting": setting,
                    "model_validation_ap": ap,
                    "warnings": [str(w.message) for w in ws],
                }
            )
            fitted.append(estimator)
            print(kind, family, setting, "AP", ap, flush=True)
        winner = int(np.argmax([e["model_validation_ap"] for e in entries]))
        base = fitted[winner]
        calibrated = CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(x[ca], y[ca])
        joblib.dump(calibrated, out / (family + ".joblib"))
        models[family] = calibrated
        candidates[family] = {"entries": entries, "winner": entries[winner]["setting"]}
        calibration[family] = {}
        for name, est in [("uncalibrated", base), ("sigmoid", calibrated)]:
            prob = est.predict_proba(x[po])[:, 1]
            calibration[family][name] = {
                "brier_on_policy_validation": float(brier_score_loss(y[po], prob)),
                "logloss_on_policy_validation": float(log_loss(y[po], prob)),
            }
    length = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    ).fit(x[tr, 13:14], y[tr])
    joblib.dump(length, out / "length.joblib")
    write_json(out / "t1-selection.json", {"candidates": candidates, "calibration": calibration})
    vocabulary = json.loads((data / (kind + "-vocabulary.json")).read_text(encoding="utf-8"))
    vectorizer(vocabulary)  # Validate the supplied vocabulary before fitting.
    write_json(out / "vocabulary.json", vocabulary)
    encoded = np.load(data / (kind + "-encoded.npy"))
    # Exact model-input fingerprint catches vocabulary/OOV/truncation collisions.
    keys = [hashlib.sha256(encoded[i].tobytes() + x[i].tobytes()).digest() for i in range(len(df))]
    reps = pd.DataFrame({"key": keys, "split": df[kind + "_split"], "label": y})
    collisions = reps.groupby("key").agg(
        rows=("key", "size"), splits=("split", "nunique"), labels=("label", "nunique")
    )
    audit = {
        "duplicate_input_groups": int((collisions.rows > 1).sum()),
        "cross_split_input_groups": int((collisions.splits > 1).sum()),
        "conflicting_input_groups": int((collisions.labels > 1).sum()),
    }
    write_json(out / "encoded-input-audit.json", audit)
    if audit["cross_split_input_groups"] or audit["conflicting_input_groups"]:
        raise RuntimeError("encoded_input_collision_requires_explicit_resolution")
    del reps, keys, collisions
    model = create_model(50, 10000, 18)
    assert model.count_params() == 162481
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            payload = {
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - self.start,
                **{k: float(v) for k, v in logs.items()},
            }
            with (out / "progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            print("EPOCH", kind, json.dumps(payload), flush=True)

    history = model.fit(
        [encoded[tr], x[tr]],
        1 - y[tr],
        validation_data=([encoded[va], x[va]], 1 - y[va]),
        epochs=20,
        batch_size=128,
        shuffle=True,
        verbose=0,
        callbacks=[
            Progress(),
            tf.keras.callbacks.CSVLogger(str(out / "epochs.csv")),
            tf.keras.callbacks.ModelCheckpoint(
                str(out / "t2.h5"), monitor="val_loss", save_best_only=True
            ),
        ],
    )
    if len(history.history["loss"]) != 20:
        raise RuntimeError("incomplete_training")
    write_json(
        out / "gru-selection.json",
        {
            "epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "params": model.count_params(),
            "history": history.history,
        },
    )
    seal_training_artifacts(out)
    return


def evaluate(kind):
    """Evaluate completed training only after unconditional artifact verification."""
    import h5py

    out = ROOT / kind
    data = ROOT / "data-final"
    if (out / "complete.json").exists():
        raise RuntimeError("already_complete")
    settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    source = Path(__file__)
    verify_training_artifacts(out)
    verify_execution_source("experiment.py", settings["experiment_sha256"])
    snapshot_sources("evaluation-source-snapshot.json")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if sha(data / "manifest.json") != settings["data_manifest_sha256"]:
        raise RuntimeError("changed_data_manifest")
    for name, digest in manifest["files"].items():
        if sha(data / name) != digest:
            raise RuntimeError("changed_data:" + name)
    selection = json.loads((out / "gru-selection.json").read_text(encoding="utf-8"))
    losses = np.asarray(selection["history"]["val_loss"])
    if len(losses) != 20 or not np.isfinite(losses).all():
        raise RuntimeError("incomplete_training")
    if selection["epoch"] != int(np.argmin(losses)) + 1:
        raise RuntimeError("checkpoint_selection_changed")
    df = pd.read_csv(data / "clean.csv", keep_default_na=False)
    x = np.load(data / "features.npy")
    encoded = np.load(data / (kind + "-encoded.npy"))
    y = df.y_mal.to_numpy()
    parts = np.load(data / (kind + "_indices.npz"))
    po, te = validation_indices(parts), parts["test"]
    vocabulary = json.loads((out / "vocabulary.json").read_text(encoding="utf-8"))
    layer = vectorizer(vocabulary)
    models = {name: joblib.load(out / (name + ".joblib")) for name in ("logistic", "boost")}
    length = joblib.load(out / "length.joblib")
    # h5py handles Unicode paths on Windows; passing its handle avoids TF's
    # legacy filesystem path probe. Weights are read without rewriting them.
    with h5py.File(out / "t2.h5", "r") as handle:
        model = tf.keras.models.load_model(handle, compile=False)
    write_json(
        out / "evaluation-code.json",
        {
            "code_sha256": sha(source),
            "training_code_sha256": settings["experiment_sha256"],
            "model_loader": "h5py file handle; compile=False; weights unchanged",
            "training_manifest_sha256": sha(out / "training-manifest.json"),
            "probability_conversion": "float64 subtraction preserves the float32 GRU decision boundary",
        },
    )

    def score(idx):
        benign = model.predict([encoded[idx], x[idx]], batch_size=2048, verbose=0).ravel()
        return malicious_scores(benign)

    p2policy = score(po)
    frozen = {}
    for family, est in models.items():
        p = est.predict_proba(x[po])[:, 1]
        records = []
        for tau in TAUS:
            mask, _ = route(p, df.id.iloc[po].to_numpy(), tau, 0, "basic")
            records.append({"tau": tau, **cascade_metrics(y[po], p2policy, mask)})
        eligible = [r for r in records if r["recall_loss"] <= 0.01]
        best = min(eligible, key=lambda r: (r["t2_call_rate"], r["tau"]))
        frozen[family] = {"tau": best["tau"], "selection": best, "candidates": records}
    write_json(
        out / "policy-frozen.json",
        {
            "families": frozen,
            "frozen_before_test": True,
            "model_hashes": {
                n: sha(out / n)
                for n in ("t2.h5", "vocabulary.json", "logistic.joblib", "boost.joblib")
            },
        },
    )
    # First test scoring only after all selection and calibration are frozen.
    p2 = score(te)
    ids = df.id.iloc[te].to_numpy()
    yt = y[te]
    scores = {"y": yt, "p2": p2, "id": ids}
    results = {
        "gru": metrics(yt, p2, p2 > 0.5),
        "length": metrics(
            yt, length.predict_proba(x[te, 13:14])[:, 1], length.predict(x[te, 13:14])
        ),
        "first_stage": {},
        "policies": {},
        "sensitivity": {},
        "random_seed_summary": {},
    }
    results["gru"].update(t2_calls=len(te), t2_call_rate=1.0, recall_loss=0.0, additional_misses=0)
    for family, est in models.items():
        p = est.predict_proba(x[te])[:, 1]
        scores[family] = p
        tau = frozen[family]["tau"]
        results["first_stage"][family] = metrics(yt, p, p >= 0.5)
        base, _ = route(p, ids, tau, 0, "basic")
        for policy, rate in [("basic", 0), ("random", 0.05), ("boundary", 0.05)]:
            mask, count = route(p, ids, tau, rate, policy)
            results["policies"][family + "_" + policy] = {
                "tau": tau,
                "rate": rate,
                "rechecked": count,
                **cascade_metrics(yt, p2, mask, base),
            }
        seeds = []
        for seed in range(30):
            mask, count = route(p, ids, tau, 0.05, "random", seed)
            seeds.append({"seed": seed, **cascade_metrics(yt, p2, mask, base)})
        results["random_seed_summary"][family] = {
            "records": seeds,
            "mean": {
                k: float(np.mean([v[k] for v in seeds]))
                for k in ("recall", "f1", "fpr", "recovered_malicious", "new_false_positives")
            },
            "std": {
                k: float(np.std([v[k] for v in seeds], ddof=1))
                for k in ("recall", "f1", "fpr", "recovered_malicious", "new_false_positives")
            },
        }
        sensitivity = []
        for candidate_tau in TAUS:
            bm, _ = route(p, ids, candidate_tau, 0, "basic")
            for rate in RATES:
                for pol in ["basic"] if rate == 0 else ["random", "boundary"]:
                    mask, count = route(p, ids, candidate_tau, rate, pol)
                    m = cascade_metrics(yt, p2, mask, bm)
                    sensitivity.append(
                        {"tau": candidate_tau, "rate": rate, "policy": pol, "rechecked": count, **m}
                    )
        results["sensitivity"][family] = sensitivity
    np.savez_compressed(out / "test-scores.npz", **scores)
    write_json(out / "results.json", results)
    # Equal-count rechecks are cohort/batch policies; timing uses real selective
    # execution in batches on the same fixed 1,000 held-out URLs, not all-GRU replay.
    rng = np.random.default_rng(46)
    sample = rng.choice(te, size=min(1000, len(te)), replace=False)
    sample_urls = df.url_key.iloc[sample].tolist()
    sample_ids = df.id.iloc[sample].to_numpy()
    expected_malicious = (
        model.predict([encoded[sample], x[sample]], batch_size=2048, verbose=0).ravel() < 0.5
    )

    @tf.function(
        input_signature=[tf.TensorSpec([None, 50], tf.int32), tf.TensorSpec([None, 18], tf.float32)]
    )
    def forward(tok, num):
        return model([tok, num], training=False)

    forward(encoded[sample[:2]], x[sample[:2]]).numpy()

    def execute(name):
        start = time.perf_counter()
        numeric = np.asarray([extract_features(u) for u in sample_urls], dtype=np.float32)
        tfeatures = time.perf_counter()
        if name == "gru":
            mask = np.ones(len(sample), dtype=bool)
            t1end = routeend = tfeatures
        else:
            family, policy = name.split("_")
            prob = models[family].predict_proba(numeric)[:, 1]
            t1end = time.perf_counter()
            mask, count = route(
                prob, sample_ids, frozen[family]["tau"], 0 if policy == "basic" else 0.05, policy
            )
            routeend = time.perf_counter()
        chosen = np.flatnonzero(mask)
        prediction = np.zeros(len(sample), dtype=bool)
        if len(chosen):
            text = tf.constant([preprocess_link(sample_urls[i]) for i in chosen])
            tok = tf.cast(layer(text), tf.int32)
            safe = forward(tok, numeric[chosen]).numpy().ravel()
            prediction[chosen] = safe < 0.5
        end = time.perf_counter()
        # Verify actual calls against cached full-GRU decisions on these IDs.
        expected = expected_malicious & mask
        if not np.array_equal(expected, prediction):
            raise RuntimeError("timing_prediction_mismatch")
        return {
            "feature_s": tfeatures - start,
            "t1_s": t1end - tfeatures,
            "routing_s": routeend - t1end,
            "t2_s": end - routeend,
            "total_s": end - start,
            "t2_calls": int(mask.sum()),
            "n": len(sample),
            "predictions_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        }

    names = ["gru"] + list(results["policies"])
    for name in names:
        execute(name)
    timing = {name: [] for name in names}
    for repeat in range(5):
        for name in rng.permutation(names):
            timing[name].append({"repeat": repeat, **execute(name)})
    timing_summary = {
        name: {
            "median_total_s": float(np.median([r["total_s"] for r in rows])),
            "mean_total_s": float(np.mean([r["total_s"] for r in rows])),
            "amortized_ms_per_url": float(np.median([r["total_s"] / r["n"] * 1000 for r in rows])),
            "throughput_urls_s": float(np.median([r["n"] / r["total_s"] for r in rows])),
            "mean_feature_s": float(np.mean([r["feature_s"] for r in rows])),
            "mean_t1_s": float(np.mean([r["t1_s"] for r in rows])),
            "mean_routing_s": float(np.mean([r["routing_s"] for r in rows])),
            "mean_t2_s": float(np.mean([r["t2_s"] for r in rows])),
            "t2_calls": rows[0]["t2_calls"],
        }
        for name, rows in timing.items()
    }
    write_json(
        out / "timing.json",
        {
            "scope": "offline equal-budget cohort, amortized batch latency; not single-request p95",
            "sample_ids": sample_ids.tolist(),
            "warmups_per_policy": 1,
            "repeats": 5,
            "rounds": timing,
            "summary": timing_summary,
        },
    )
    write_json(
        out / "complete.json",
        {
            "status": "PASS",
            "kind": kind,
            "files": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        },
    )
    print("COMPLETE", kind, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["random", "domain", "self-test"])
    parser.add_argument("--resume-evaluation", action="store_true")
    args = parser.parse_args()
    if args.kind == "self-test":
        self_test()
        print("SELF_TEST PASS")
    else:
        if not args.resume_evaluation:
            train(args.kind)
        evaluate(args.kind)
