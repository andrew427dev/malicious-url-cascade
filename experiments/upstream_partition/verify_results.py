"""Audit saved test predictions, budgets, split membership and all timing rows."""

from itertools import product
import hashlib
import platform

from inputs import ROOT, pd, np, sha, write_json, PREFIX
from evaluate import FAMILIES, POLICIES, read, configurations, metrics
from sklearn.metrics import average_precision_score


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def check_policy(y, gru, mask, row):
    pred = (gru > 0.5) & mask
    pos = y == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int(pos.sum() - tp)
    assert [row[k] for k in ("tp", "fp", "fn", "tn")] == [
        tp,
        fp,
        fn,
        int(len(y) - pos.sum() - fp),
    ]
    assert row["t2_calls"] == mask.sum()
    assert row["additional_misses"] == int((pos & (gru > 0.5) & ~mask).sum())
    close(
        [row[k] for k in ("accuracy", "recall", "fpr", "t2_call_rate")],
        [(pred == y).mean(), tp / pos.sum(), fp / (~pos).sum(), mask.mean()],
    )


def mask_for(ids, p, tau, rate, policy, seed=0):
    mask = p >= tau
    early = np.flatnonzero(~mask)
    k = int(len(early) * rate)
    if policy == "boundary":
        order = sorted(early, key=lambda i: (-p[i], ids[i]))
    elif k:
        order = sorted(
            early,
            key=lambda i: (
                hashlib.sha256(f"{seed}|{ids[i]}".encode()).digest(),
                ids[i],
            ),
        )
    else:
        order = []
    mask[order[:k]] = True
    return mask


def verify_cohort_inputs(kind):
    out = ROOT / kind
    inputs = read(out / "full-input-audit.json")
    roles = ("train", "validation", "test")
    required = {"data.json", "vocab.json"} | {
        role + suffix
        for role in roles
        for suffix in (".pkl", "-rows.csv.gz", "-features.npy", "-tokens.npy")
    }
    if inputs["status"] != "PASS" or set(inputs["files"]) != required:
        raise ValueError("Incomplete or failed input audit for " + kind)
    records_by_role = read(out / "data.json")["stages"]
    if inputs["rows_rebuilt_from_source"] != {
        r: records_by_role[r]["after"]["rows"] for r in roles
    }:
        raise ValueError("Full row coverage differs for " + kind)
    for name, digest in inputs["files"].items():
        assert sha(out / name) == digest, name


def audit(kind):
    out = ROOT / kind
    verify_cohort_inputs(kind)
    records = read(out / "results.json")
    with np.load(out / "scores.npz") as z:
        ids, y, gru = z["id"], z["y"], z["gru"]
        p = {f: z[f] for f in FAMILIES}
    frame = pd.read_pickle(out / "test.pkl")
    assert np.array_equal(frame.index, ids) and np.array_equal(1 - frame.type, y)
    flags = frame.url.str.match(PREFIX).to_numpy()
    frozen = read(out / "routing.json")
    selection = read(out / "trained.json")
    assert len(selection["history"]["val_loss"]) == 20
    assert selection["epoch"] == int(np.argmin(selection["history"]["val_loss"])) + 1
    assert frozen["gru_sha256"] == selection["weights_sha256"] == sha(out / "gru.h5")
    check_policy(y, gru, np.ones(len(y), bool), records["policies"]["gru"])
    sweeps = read(out / "sensitivity.json")
    representative = {}
    categories = []
    for f in FAMILIES:
        assert frozen["first_stage_sha256"][f] == sha(out / "stage1" / (f + ".joblib"))
        grid = frozen["families"][f]["grid"]
        tau = min(
            (r for r in grid if r["recall_loss"] <= 0.01),
            key=lambda r: (r["t2_calls"], r["tau"]),
        )["tau"]
        assert tau == frozen["families"][f]["tau"]
        close(average_precision_score(y, p[f]), records["first_stage"][f]["pr_auc_ap"])
        for policy, rate in POLICIES:
            mask = mask_for(ids, p[f], tau, rate, policy)
            check_policy(y, gru, mask, records["policies"][f + "_" + policy])
        base = p[f] >= tau
        rescued = mask & ~base & (gru > 0.5)
        boundary = records["policies"][f + "_boundary"]
        groups = {
            "early_exit_miss": (
                (y == 1) & (gru > 0.5) & ~mask,
                boundary["additional_misses"],
            ),
            "recovered_malicious": (
                (y == 1) & rescued,
                boundary["recovered_malicious"],
            ),
            "new_false_positive": ((y == 0) & rescued, boundary["new_false_positives"]),
            "gru_false_negative": (
                (y == 1) & (gru <= 0.5),
                records["policies"]["gru"]["fn"],
            ),
        }
        for tag, (selected, expected_count) in groups.items():
            assert int(selected.sum()) == expected_count
            stable = sorted(ids[selected].tolist())
            categories.append(
                {
                    "family": f,
                    "category": tag,
                    "population": len(stable),
                    "first_source_id": stable[0] if stable else None,
                }
            )
            if tag == "early_exit_miss":
                representative[f] = stable[:3]
        for entry in sweeps[f]["random_seeds"]:
            check_policy(
                y, gru, mask_for(ids, p[f], tau, 0.05, "random", entry["seed"]), entry
            )
        assert (
            len(sweeps[f]["random_seeds"]) == 30
            and len(sweeps[f]["sensitivity"]) == 108
        )
        for entry in sweeps[f]["sensitivity"]:
            mask = mask_for(ids, p[f], entry["tau"], entry["rate"], entry["policy"])
            check_policy(y, gru, mask, entry)
    if kind.endswith("original"):
        fixed = read(out / "fixed-prefix.json")
        with np.load(out / "paired-scores.npz") as z:
            assert np.array_equal(ids, z["id"]) and np.array_equal(y, z["y"])
            for name in records["policies"]:
                alt_mask = np.ones(len(ids), dtype=bool)
                a = gru > 0.5
                if name != "gru":
                    family, policy = name.split("_")
                    tau = frozen["families"][family]["tau"]
                    rate = 0 if policy == "basic" else 0.05
                    a &= mask_for(ids, p[family], tau, rate, policy)
                    alt_mask = mask_for(ids, z[family], tau, rate, policy)
                check_policy(y, z["gru"], alt_mask, fixed["stripped"][name])
                b = (z["gru"] > 0.5) & alt_mask
                for tag, where in (("with_prefix", flags), ("without_prefix", ~flags)):
                    assert fixed["pairs"][tag]["flips"][name] == {
                        "changed": int(((a != b) & where).sum()),
                        "malicious_to_benign": int((a & ~b & where).sum()),
                        "benign_to_malicious": int((~a & b & where).sum()),
                    }
    timing = read(out / "timing.json")
    assert len(set(timing["sample_ids"])) == 1000 and set(timing["sample_ids"]) <= set(
        ids
    )
    for repeat in range(1, 6):
        rows = [r for r in timing["rounds"] if r["repeat"] == repeat]
        assert len(rows) == 7 and {r["name"] for r in rows} == set(records["policies"])
        assert sorted(r["order"] for r in rows) == list(range(1, 8))
    ix = frame.index.get_indexer(timing["sample_ids"])
    for name, t in timing["summary"].items():
        rows = [r for r in timing["rounds"] if r["name"] == name]
        times = [r["total_ms"] for r in rows]
        close(
            [t["ms"], t["min_ms"], t["max_ms"]],
            [np.median(times), min(times), max(times)],
        )
        expected = 1000
        if name != "gru":
            family, policy = name.split("_")
            tau = frozen["families"][family]["tau"]
            expected = mask_for(
                ids[ix], p[family][ix], tau, 0 if policy == "basic" else 0.05, policy
            ).sum()
        assert all(r["calls"] == expected for r in rows)
        for r in rows:
            close(r["total_ms"], r["features_ms"] + r["route_ms"] + r["gru_ms"])
    write_json(out / "representative-cases.json", categories)
    return representative


def paired_retraining(group):
    a, b = [ROOT / (group + "-" + form) for form in ("original", "stripped")]
    left = pd.read_csv(a / "test-rows.csv.gz", index_col="id")
    right = pd.read_csv(b / "test-rows.csv.gz", index_col="id")
    ids = left.index.intersection(right.index, sort=False)
    assert np.array_equal(left.y_mal.loc[ids], right.y_mal.loc[ids])
    result = {
        "common_n": len(ids),
        "original_n": len(left),
        "stripped_n": len(right),
        "ordered_ids_equal": left.index.equals(right.index),
    }
    ta, tb = read(a / "timing.json"), read(b / "timing.json")
    assert ta["sample_ids"] == tb["sample_ids"]
    for repeat in range(1, 6):
        joined = [
            r["pair_order"]
            for z in (ta, tb)
            for r in z["rounds"]
            if r["repeat"] == repeat
        ]
        assert sorted(joined) == list(range(1, 15))
    for form, out, frame in (("original", a, left), ("stripped", b, right)):
        with np.load(out / "scores.npz") as z:
            ix = frame.index.get_indexer(ids)
            taus = {
                f: d["tau"] for f, d in read(out / "routing.json")["families"].items()
            }
            _, r = configurations(
                z["y"][ix], ids, z["gru"][ix], {f: z[f][ix] for f in FAMILIES}, taus
            )
            result[form] = r
    return result


def hardware():
    import sys

    cpu, system = platform.processor() or platform.machine(), platform.system()
    release = platform.release()
    if system == "Windows":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.getwindowsversion().build >= 22000 and release in ("10", "11"):
            release = "11"
    return {"cpu": cpu, "operating_system": f"{system} {release}"}


def main():
    import importlib.metadata as meta
    from provenance import verify_seal

    for kind in (
        "author-validation",
        "author-final",
        "random-original",
        "random-stripped",
        "domain-original",
        "domain-stripped",
    ):
        verify_seal(kind)
        trained = read(ROOT / kind / "trained.json")
        epochs = pd.read_csv(ROOT / kind / "epochs.csv")
        np.testing.assert_array_equal(epochs.epoch, np.arange(20))
        for key, values in trained["history"].items():
            close(epochs[key].to_numpy(), values)

    out = ROOT / "summary"
    out.mkdir(exist_ok=True)
    cases = {}
    performance = []
    times = []
    for group, form in product(("random", "domain"), ("original", "stripped")):
        kind = group + "-" + form
        cases[kind] = audit(kind)
        for name, row in read(ROOT / kind / "results.json")["policies"].items():
            performance.append({"condition": kind, "policy": name, **row})
        for name, row in read(ROOT / kind / "timing.json")["summary"].items():
            times.append({"condition": kind, "policy": name, **row})
    pd.DataFrame(performance).to_csv(out / "performance.csv", index=False)
    pd.DataFrame(times).to_csv(out / "timing.csv", index=False)
    common = {g: paired_retraining(g) for g in ("random", "domain")}
    write_json(out / "common-test.json", common)
    for role in ("validation", "external"):
        with np.load(ROOT / "author-validation" / (role + "-scores.npz")) as z:
            from sklearn.metrics import precision_recall_fscore_support

            outdir = ROOT / "author-validation"
            manifest = pd.read_csv(outdir / (role + "-rows.csv.gz"))
            np.testing.assert_array_equal(manifest.id, z["id"])
            np.testing.assert_array_equal(manifest.y_mal, z["y"])
            np.testing.assert_array_equal(
                np.sort(z["evaluation_order"]), np.arange(len(manifest))
            )
            prediction = z["gru"] > 0.5
            r = metrics(z["y"], z["gru"], prediction)
            record = read(outdir / "results.json")
            saved = record[role]
            for k, v in r.items():
                close(saved[k], v)
            weighted = precision_recall_fscore_support(
                z["y"], prediction, average="weighted", zero_division=0
            )[:3]
            close(
                weighted,
                [
                    saved[k]
                    for k in ("weighted_precision", "weighted_recall", "weighted_f1")
                ],
            )
            if role == "external":
                urls = pd.read_pickle(outdir / "external.pkl").url
                known = set(pd.read_pickle(outdir / "train.pkl").url) | set(
                    pd.read_pickle(outdir / "validation.pkl").url
                )
                unseen = ~urls.isin(known).to_numpy()
                stored = record["external_without_exact_training_or_validation_url"]
                positive = unseen & (z["y"] == 1)
                negative = unseen & (z["y"] == 0)
                assert [int(mask.sum()) for mask in (unseen, positive, negative)] == [
                    stored[k] for k in ("n", "malicious", "benign")
                ]
                close(prediction[positive].mean(), stored["malicious_recall"])
                if negative.any():
                    close(prediction[negative].mean(), stored["fpr"])
                else:
                    assert stored["fpr"] is None
    for k in ("author-final", "author-validation"):
        rec = read(ROOT / k / "trained.json")
        assert (
            rec["epoch"] == 20
            and len(rec["history"]["loss"]) == 20
            and sha(ROOT / k / "gru.h5") == rec["weights_sha256"]
        )
    write_json(
        out / "checks.json",
        {
            "status": "PASS",
            "examples_by_source_id": cases,
            "primary_rows": len(performance),
            "timed_batches": len(times) * 5,
            "platform": platform.platform(),
            "hardware": hardware(),
            "python": platform.python_version(),
            "packages": {
                p: meta.version(p)
                for p in (
                    "tensorflow",
                    "keras",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "tldextract",
                    "h5py",
                    "scipy",
                    "joblib",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
