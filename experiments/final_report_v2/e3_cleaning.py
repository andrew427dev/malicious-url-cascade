"""Experiment 3: how much of the basic-cleaning advantage comes from repeated model inputs?

Basic cleaning keeps test rows whose model input (50 tokens + 18 numbers) or whose first 52
model characters + 18 numbers already occur in its own training or validation rows; team
cleaning removes such groups across all roles. This script finds those rows with the same
definitions v1 used (review_evidence_v1.signatures and cohorts.build), then compares the
GRU of both cleanings on the shared test URLs with those rows removed or kept alone.

It also extends the raw-versus-normalized input check to every role of the team data: the
team models read the normalized URL, so it counts how many rows would get other tokens or
other numeric features if the raw URL were used instead.
"""

import hashlib

import numpy as np
import pandas as pd

from v2common import V1, RESULTS, MAIN, REFERENCE, load_test, read_json, write_json
from inputs import extract_features, preprocess_link, vectorizer, encode, FEATURE_NAMES

ROLES = ("train", "validation", "test")


def exact_keys(out, role):
    """v1 review_evidence_v1.signatures: SHA-256 of the 50 token ids and 18 float32 features."""
    tok = np.load(out / (role + "-tokens.npy"), mmap_mode="r")
    num = np.load(out / (role + "-features.npy"), mmap_mode="r")
    return np.array([hashlib.sha256(t.tobytes() + n.tobytes()).digest() for t, n in zip(tok, num)], dtype=object)


def prefix_keys(out, role):
    """cohorts.build key: first 52 model characters plus the 18 numeric features.

    v1 hashed json.dumps of the Python feature list; the features are integer counts, so the
    float32 bytes stored for the model identify the same values (checked below).
    """
    frame = pd.read_pickle(out / (role + ".pkl"))
    num = np.load(out / (role + "-features.npy"), mmap_mode="r")
    assert np.array_equal(num, np.round(num)), "non-integer feature breaks the key equivalence"
    return np.array([hashlib.sha256(t[:52].encode() + b"|" + n.tobytes()).digest()
                     for t, n in zip(frame.text, num)], dtype=object)


def seen_before(out, keyfn):
    prior = set(keyfn(out, "train")) | set(keyfn(out, "validation"))
    return np.array([k in prior for k in keyfn(out, "test")])


def gru_counts(y, gru):
    pred = gru > 0.5
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    n = len(y)
    return dict(n=n, malicious=tp + fn, benign=fp + tn, tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn) if tp + fn else None,
                accuracy=(tp + tn) / n if n else None,
                fpr=fp / (fp + tn) if fp + tn else None)


def overlap_comparison(split):
    ref_kind, team_kind = REFERENCE[split], MAIN[split]
    ref_frame, ref_s = load_test(ref_kind)
    team_frame, team_s = load_test(team_kind)
    common = pd.read_csv(V1 / "results" / (split + "-common.csv.gz"))["id"].to_numpy()
    flags, totals = {}, []
    for kind, frame in ((ref_kind, ref_frame), (team_kind, team_frame)):
        out = V1 / kind
        exact = seen_before(out, exact_keys)
        prefix = seen_before(out, prefix_keys)
        in_common = frame.index.isin(common)
        totals.append(dict(split=split, kind=kind, test_rows=len(frame),
                           # Basic cleaning keeps the first row per model string within a role; team cleaning
                           # keys on string+numbers instead, so equal strings with other numbers can remain.
                           test_repeated_model_text=int(frame.text.duplicated().sum()),
                           exact_full=int(exact.sum()), exact_common=int(exact[in_common].sum()),
                           prefix52_full=int(prefix.sum()), prefix52_common=int(prefix[in_common].sum()),
                           either_full=int((exact | prefix).sum()), both_full=int((exact & prefix).sum()),
                           exact_full_malicious=int(ref_s["y"][exact].sum()) if kind == ref_kind else None))
        flags[kind] = (pd.Series(exact, frame.index), pd.Series(prefix, frame.index))
    # v1 table 6/8 recorded these counts; reproduce them before using the rows.
    v1 = {v["kind"]: v for v in read_json(V1 / "results/review-evidence.json")["input_overlap"]}
    for t in totals:
        assert (t["exact_full"], t["exact_common"]) == (v1[t["kind"]]["full"], v1[t["kind"]]["common"]), t
    # Subsets of the shared test URLs, defined by the basic-cleaning rows only (team has none).
    rows = []
    ref_pos = ref_frame.index.get_indexer(common)
    team_pos = team_frame.index.get_indexer(common)
    assert (ref_pos >= 0).all() and (team_pos >= 0).all()
    y_ref, y_team = ref_s["y"][ref_pos], team_s["y"][team_pos]
    assert np.array_equal(y_ref, y_team)
    exact_c = flags[ref_kind][0].to_numpy()[ref_pos]
    prefix_c = flags[ref_kind][1].to_numpy()[ref_pos]
    subsets = {
        "all_common": np.ones(len(common), bool),
        "without_exact_input_overlap": ~exact_c,
        "only_exact_input_overlap": exact_c,
        "without_prefix52_overlap": ~prefix_c,
        "only_prefix52_overlap": prefix_c,
        "without_either_overlap": ~(exact_c | prefix_c),
    }
    for name, sel in subsets.items():
        a = gru_counts(y_ref[sel], ref_s["gru"][ref_pos][sel])
        b = gru_counts(y_team[sel], team_s["gru"][team_pos][sel])
        row = dict(split=split, subset=name)
        row.update({"basic_" + k: v for k, v in a.items()})
        row.update({"team_" + k: v for k, v in b.items() if k not in ("n", "malicious", "benign")})
        if a["recall"] is not None:
            row["recall_diff_pp"] = 100 * (b["recall"] - a["recall"])
        row["accuracy_diff_pp"] = 100 * (b["accuracy"] - a["accuracy"]) if a["n"] else None
        rows.append(row)
    return totals, rows


def representation_check(split):
    """Team data only: raw URL versus the normalized URL the team models read, every role."""
    kind = MAIN[split]
    out = V1 / kind
    layer = vectorizer(read_json(out / "vocab.json"))
    rows, columns = [], []
    for role in ROLES:
        frame = pd.read_pickle(out / (role + ".pkl"))
        stored_tok = np.load(out / (role + "-tokens.npy"))
        stored_num = np.load(out / (role + "-features.npy"))
        # The stored arrays were built from input_url; confirm before comparing with the raw URL.
        assert np.array_equal(encode(frame.input_url.map(preprocess_link), layer), stored_tok)
        raw_num = np.asarray([extract_features(u) for u in frame.url], np.float32)
        raw_tok = encode(frame.url.map(preprocess_link), layer)
        tok_diff = np.any(raw_tok != stored_tok, axis=1)
        num_diff = np.any(raw_num != stored_num, axis=1)
        text_diff = (frame.url.map(preprocess_link) != frame.text).to_numpy()
        trailing = (frame.url.str.rstrip("/") != frame.url) & (frame.input_url.str.rstrip("/") == frame.input_url)
        rows.append(dict(split=split, kind=kind, role=role, rows=len(frame),
                         url_changed_by_normalization=int((frame.url != frame.input_url).sum()),
                         trailing_slash_removed=int(trailing.sum()),
                         model_text_differs=int(text_diff.sum()), token_rows_differ=int(tok_diff.sum()),
                         numeric_rows_differ=int(num_diff.sum()),
                         numeric_differ_with_trailing_slash=int((num_diff & trailing.to_numpy()).sum())))
        for j, name in enumerate(FEATURE_NAMES):
            columns.append(dict(split=split, role=role, feature=name,
                                rows_differ=int((raw_num[:, j] != stored_num[:, j]).sum())))
    return rows, columns


def main():
    totals, subsets, rep, cols = [], [], [], []
    for split in ("random", "domain"):
        t, s = overlap_comparison(split)
        totals += t; subsets += s
        r, c = representation_check(split)
        rep += r; cols += c
    pd.DataFrame(totals).to_csv(RESULTS / "e3-overlap-counts.csv", index=False)
    pd.DataFrame(subsets).to_csv(RESULTS / "e3-common-subsets.csv", index=False)
    pd.DataFrame(rep).to_csv(RESULTS / "e3-representation.csv", index=False)
    c = pd.DataFrame(cols)
    c[c.rows_differ > 0].to_csv(RESULTS / "e3-representation-columns.csv", index=False)
    # The handoff measured 19,393 differing numeric rows on the URL-random team test; reproduce it.
    test_random = next(r for r in rep if r["split"] == "random" and r["role"] == "test")
    checks = {
        "v1_exact_input_overlap_reproduced": True,
        "team_overlaps_zero": all(t["exact_full"] == 0 and t["prefix52_full"] == 0 for t in totals if t["kind"].startswith("team-")),
        "random_test_numeric_rows_differ": test_random["numeric_rows_differ"],
        "random_test_token_rows_differ": test_random["token_rows_differ"],
        "handoff_value_19393": test_random["numeric_rows_differ"] == 19393,
    }
    checks["status"] = "PASS" if checks["team_overlaps_zero"] and checks["handoff_value_19393"] else "FAIL"
    write_json(RESULTS / "e3-checks.json", checks)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(totals).to_string())
    print(pd.DataFrame(subsets).round(4).to_string())
    print(pd.DataFrame(rep).to_string())
    print(c[c.rows_differ > 0].to_string())
    print(checks)
    assert checks["status"] == "PASS"


if __name__ == "__main__":
    main()
