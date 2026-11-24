"""Freeze validation routing, then score held-out rows and paired representations."""

import argparse
import json
from inputs import ROOT, PREFIX, sha, write_json, inputs, encode, vectorizer, np, pd
from Utils.Model import create_model
from experiment import route, malicious_scores, metrics, cascade_metrics, RATES, TAUS
from sklearn.metrics import precision_recall_fscore_support
import joblib

FAMILIES = ("logistic", "boost")
POLICIES = (("basic", 0), ("random", 0.05), ("boundary", 0.05))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def load_gru(out):
    from provenance import verify_seal

    verify_seal(out)
    rec = read(out / "trained.json")
    if (
        sha(out / "gru.h5") != rec["weights_sha256"]
        or sha(out / "vocab.json") != rec["vocabulary_sha256"]
    ):
        raise ValueError("GRU weights or vocabulary changed")
    model = create_model(50, 10000, 18)
    model.load_weights(str(out / "gru.h5"))
    return model


def score(model, tok, x):
    return malicious_scores(
        model.predict([tok, x], batch_size=2048, verbose=0).reshape(-1)
    )


def load_rows(out, role):
    frame = pd.read_pickle(out / (role + ".pkl"))
    rows = pd.read_csv(out / (role + "-rows.csv.gz"), index_col="id")
    assert rows.index.equals(frame.index) and np.array_equal(rows.type, frame.type)
    return (
        frame,
        rows.y_mal.to_numpy(),
        np.load(out / (role + "-tokens.npy")),
        np.load(out / (role + "-features.npy")),
    )


def load_stages(out):
    out = out / "stage1"
    hashes = read(out / "t1-selected.json")["hashes"]
    if any(sha(out / (f + ".joblib")) != digest for f, digest in hashes.items()):
        raise ValueError("First stage changed")
    return {f: joblib.load(out / (f + ".joblib")) for f in FAMILIES}


def author(out):
    model = load_gru(out)
    summary = {}
    for role in ("validation", "external"):
        frame, y, tok, x = load_rows(out, role)
        from inputs import tf

        dataset = tf.data.Dataset.from_tensor_slices((np.arange(len(y)), tok, x, 1 - y))
        dataset = dataset.shuffle(10000, seed=42).batch(128).cache().prefetch(1)
        order = np.concatenate([idx.numpy() for idx, *_ in dataset])
        np.testing.assert_array_equal(np.sort(order), np.arange(len(y)))
        shuffled = dataset.map(lambda idx, t, numeric, label: ((t, numeric), label))
        actual_labels = np.concatenate([label.numpy() for _, label in shuffled])
        np.testing.assert_array_equal(actual_labels, 1 - y[order])
        prob = np.empty(len(y), dtype="float64")
        prob[order] = malicious_scores(model.predict(shuffled, verbose=0).reshape(-1))
        pred = prob > 0.5
        summary[role] = metrics(y, prob, pred)
        for k, v in zip(
            ("weighted_precision", "weighted_recall", "weighted_f1"),
            precision_recall_fscore_support(y, pred, average="weighted")[:3],
        ):
            summary[role][k] = float(v)
        np.savez_compressed(
            out / (role + "-scores.npz"),
            id=np.asarray(frame.index, dtype=str),
            evaluation_order=order,
            y=y,
            gru=prob,
        )
        if role == "external":
            other = pd.concat(
                [pd.read_pickle(out / (p + ".pkl")) for p in ("train", "validation")]
            )
            unseen = ~frame.url.isin(other.url)
            yy, pp = y[unseen], pred[unseen]
            summary["external_without_exact_training_or_validation_url"] = {
                "n": len(yy),
                "malicious": int(yy.sum()),
                "benign": int((yy == 0).sum()),
                "malicious_recall": float(np.mean(pp[yy == 1])),
                "fpr": float(np.mean(pp[yy == 0])) if np.any(yy == 0) else None,
            }
    write_json(out / "results.json", summary)


def author_examples():
    import ast
    from Utils.Evaluation import make_prediction

    cells = read(ROOT / "source/Model_Training.ipynb")["cells"]
    records = []
    for condition, source_cell, output_cell, batch in (
        ("author-validation", 12, 13, 128),
        ("author-final", 16, 16, 1),
        ("author-final", 24, 24, 128),
    ):
        out = ROOT / condition
        assignments = ast.parse("".join(cells[source_cell]["source"])).body
        urls = next(
            ast.literal_eval(node.value)
            for node in assignments
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "links"
        )
        original = next(
            ast.literal_eval("".join(o["data"]["text/plain"]))
            for o in cells[output_cell]["outputs"]
            if o["output_type"] == "execute_result"
        )
        actual = make_prediction(
            load_gru(out), batch, vectorizer(read(out / "vocab.json")), urls, 0.5
        )
        assert len(actual) == len(original) == len(urls)
        records.append(
            {
                "model": condition,
                "notebook_cell": output_cell,
                "published_predictions": original,
                "new_predictions": actual,
            }
        )
    write_json(ROOT / "author-examples.json", records)


def freeze(out, model, stages):
    frame, y, tok, x = load_rows(out, "validation")
    gru = score(model, tok, x)
    frozen = {}
    for f, est in stages.items():
        p = est.predict_proba(x)[:, 1]
        grid = []
        for tau in TAUS:
            mask, _ = route(p, frame.index, tau, 0, "basic")
            grid.append({"tau": tau, **cascade_metrics(y, gru, mask)})
        winner = min(
            (r for r in grid if r["recall_loss"] <= 0.01),
            key=lambda r: (r["t2_calls"], r["tau"]),
        )
        frozen[f] = {"tau": winner["tau"], "grid": grid}
    write_json(
        out / "routing.json",
        {
            "families": frozen,
            "gru_sha256": sha(out / "gru.h5"),
            "first_stage_sha256": read(out / "stage1/t1-selected.json")["hashes"],
            "validation_rows": len(y),
        },
    )
    return {f: d["tau"] for f, d in frozen.items()}


def configurations(y, ids, gru, first, taus):
    masks = {"gru": np.ones(len(y), bool)}
    rows = {"gru": cascade_metrics(y, gru, masks["gru"])}
    for f, p in first.items():
        base = p >= taus[f]
        for policy, rate in POLICIES:
            mask, k = route(p, ids, taus[f], rate, policy)
            name = f + "_" + policy
            masks[name] = mask
            rows[name] = {
                "tau": taus[f],
                "rechecks": k,
                **cascade_metrics(y, gru, mask, base),
            }
    return masks, rows


def supplemental(y, ids, gru, p, tau):
    base = p >= tau
    seeds = []
    for seed in range(30):
        mask, k = route(p, ids, tau, 0.05, "random", seed)
        seeds.append(
            {"seed": seed, "rechecks": k, **cascade_metrics(y, gru, mask, base)}
        )
    grid = []
    for t in TAUS:
        for rate in RATES:
            for policy in ("random", "boundary") if rate else ("basic",):
                mask, k = route(p, ids, t, rate, policy)
                grid.append(
                    {
                        "tau": t,
                        "rate": rate,
                        "policy": policy,
                        "rechecks": k,
                        **cascade_metrics(y, gru, mask, p >= t),
                    }
                )
    return {"random_seeds": seeds, "sensitivity": grid}


def controls(out, frame, y, x, first):
    from fit_stages import OMIT, COLUMNS

    dest = out / "stage1"
    supplemental = read(dest / "t1-selected.json")["supplemental"]
    result = dict(supplemental)
    for family in FAMILIES:
        if out.name.endswith("original"):
            reduced = joblib.load(dest / (family + "-omit.joblib"))
            p = reduced.predict_proba(x[:, COLUMNS])[:, 1]
            result[family + "_omit_columns"] = {
                "omitted": list(OMIT),
                **metrics(y, p, p >= 0.5),
            }
    if out.name.endswith("stripped"):
        train_raw = pd.read_pickle(out / "train.pkl").url.str.replace(
            PREFIX, "", n=1, regex=True
        )
        test_raw = frame.url.str.replace(PREFIX, "", n=1, regex=True)
        include = ~test_raw.isin(train_raw).to_numpy()
        result["stripped_collision_diagnostic"] = {
            "overlap_rows": int((~include).sum()),
            "kept": int(include.sum()),
            "malicious": int(y[include].sum()),
            "benign": int((y[include] == 0).sum()),
            "first_stage": {
                f: metrics(y[include], p[include], p[include] >= 0.5)
                for f, p in first.items()
            },
        }
    length = joblib.load(dest / "length.joblib")
    p = length.predict_proba(x[:, 13:14])[:, 1]
    result["length"] = metrics(y, p, p >= 0.5)
    represented = (
        frame.url.str.replace(PREFIX, "", n=1, regex=True)
        if out.name.endswith("stripped")
        else frame.url
    )
    has = represented.str.match(PREFIX).to_numpy()
    result["scheme_only"] = metrics(y, has.astype(float), has)
    result["scheme_label"] = {
        str(v): {"n": int((has == v).sum()), "malicious": int(y[has == v].sum())}
        for v in (False, True)
    }
    write_json(out / "controls-final.json", result)


def require_frozen():
    for group in ("random", "domain"):
        for form in ("original", "stripped"):
            out = ROOT / (group + "-" + form)
            frozen = read(out / "routing.json")
            if sha(out / "gru.h5") != frozen["gru_sha256"]:
                raise ValueError("Frozen GRU changed")
            if any(
                sha(out / "stage1" / (key + ".joblib")) != digest
                for key, digest in frozen["first_stage_sha256"].items()
            ):
                raise ValueError("Frozen first stage changed")


def team(out):
    model = load_gru(out)
    stages = load_stages(out)
    require_frozen()
    taus = {f: v["tau"] for f, v in read(out / "routing.json")["families"].items()}
    frame, y, tok, x = load_rows(out, "test")
    ids = np.asarray(frame.index, dtype=str)
    gru = score(model, tok, x)
    first = {f: m.predict_proba(x)[:, 1] for f, m in stages.items()}
    masks, results = configurations(y, ids, gru, first, taus)
    controls(out, frame, y, x, first)
    np.savez_compressed(out / "scores.npz", id=ids, y=y, gru=gru, **first)
    write_json(
        out / "results.json",
        {
            "policies": results,
            "first_stage": {f: metrics(y, p, p >= 0.5) for f, p in first.items()},
        },
    )
    write_json(
        out / "sensitivity.json",
        {f: supplemental(y, ids, gru, p, taus[f]) for f, p in first.items()},
    )
    if out.name.endswith("original"):
        # No preparation/dedup call in the paired evaluation.
        raw, rawtext = inputs(frame)
        layer = vectorizer(read(out / "vocab.json"))
        np.testing.assert_array_equal(raw, x)
        np.testing.assert_array_equal(encode(rawtext, layer), tok)
        alt, text = inputs(frame, True)
        assert text.index.equals(frame.index)
        alt_tok = encode(text, layer)
        has = frame.url.str.match(PREFIX).to_numpy()
        char_change = np.any(tok != alt_tok, axis=1)
        numeric_change = np.any(x != alt, axis=1)
        ag = score(model, alt_tok, alt)
        ap = {f: m.predict_proba(alt)[:, 1] for f, m in stages.items()}
        am, ar = configurations(y, ids, ag, ap, taus)
        paired = {}
        for where, selected in (("with_prefix", has), ("without_prefix", ~has)):
            flips = {}
            for name, mask in masks.items():
                a, b = (gru > 0.5) & mask, (ag > 0.5) & am[name]
                flips[name] = {
                    "changed": int(((a != b) & selected).sum()),
                    "malicious_to_benign": int((a & ~b & selected).sum()),
                    "benign_to_malicious": int((~a & b & selected).sum()),
                }
            paired[where] = {
                "rows": int(selected.sum()),
                "malicious": int(y[selected].sum()),
                "token_changes": int((selected & char_change).sum()),
                "numeric_changes": int((selected & numeric_change).sum()),
                "flips": flips,
            }
        np.savez_compressed(out / "paired-scores.npz", id=ids, y=y, gru=ag, **ap)
        write_json(out / "fixed-prefix.json", {"stripped": ar, "pairs": paired})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.kind
    if args.kind == "author-examples":
        require_frozen()
        if (ROOT / "author-examples.json").exists():
            raise FileExistsError("Existing notebook examples")
        author_examples()
    elif args.freeze:
        if (out / "routing.json").exists():
            raise FileExistsError("Routing already frozen")
        freeze(out, load_gru(out), load_stages(out))
    else:
        if (out / "results.json").exists():
            raise FileExistsError("Existing evaluation")
        require_frozen()
        (author if args.kind == "author-validation" else team)(out)
