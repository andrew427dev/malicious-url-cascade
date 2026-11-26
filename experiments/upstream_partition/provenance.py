"""Link training to the exact row, tensor, vocabulary and source artifacts."""

import argparse
import json
from pathlib import Path
from inputs import ROOT, SOURCE, REPO, sha, write_json, FEATURE_NAMES


def seal(kind, timing="before_fit", snapshot=None):
    out = ROOT / kind
    target = out / "training-inputs.json"
    if target.exists():
        raise FileExistsError(target)
    here = Path(__file__).resolve().parent
    code = snapshot or here
    paths = {str(p.relative_to(ROOT)): p for p in (SOURCE / "dataset").glob("*.csv")}
    for name in ("DataProcessing.py", "FeatureCreation.py", "Model.py"):
        p = SOURCE / "Utils" / name
        paths[str(p.relative_to(ROOT))] = p
    for name in ("inputs.py", "train_gru.py", "team_data.py"):
        paths["runner/" + name] = code / name
    for name in (
        "urlcascade/adapters.py",
        "urlcascade/features.py",
        "experiments/clean_rerun/prepare.py",
    ):
        paths["team/" + name] = REPO / name
    for p in [
        out / "vocab.json",
        *out.glob("*-features.npy"),
        *out.glob("*-tokens.npy"),
        *out.glob("*-rows.csv.gz"),
    ]:
        paths[p.name] = p
    write_json(
        target, {"capture": timing, "files": {k: sha(p) for k, p in paths.items()}}
    )


def verify_seal(kind):
    out = ROOT / kind
    rec = json.loads((out / "training-inputs.json").read_text())
    code = (
        Path(__file__).resolve().parent
        if rec["capture"] == "before_fit"
        else ROOT / "started-training-source"
    )
    required = {
        "vocab.json",
        "train-tokens.npy",
        "train-features.npy",
        "train-rows.csv.gz",
        "runner/train_gru.py",
        "source/Utils/Model.py",
    }
    files = {key.replace("\\", "/"): digest for key, digest in rec["files"].items()}
    if not required <= files.keys():
        raise ValueError("Incomplete training input manifest")
    for key, digest in files.items():
        prefix, _, tail = key.partition("/")
        if prefix == "runner":
            path = code / tail
        elif prefix == "team":
            path = REPO / tail
        elif prefix == "source":
            path = SOURCE / tail
        else:
            path = out / key
        if sha(path) != digest:
            raise ValueError("Training input changed: " + key)


def example():
    from inputs import pd, tf, np, vectorizer, FeatureCreation, DataProcessing

    df = pd.DataFrame({"url": ["https://example.org", "example.org"], "type": [1, 1]})
    ref = FeatureCreation.create_features(df.copy())
    texts = df.url.map(DataProcessing.preprocess_link)
    vocab = ROOT / "random-original/vocab.json"
    tokens = vectorizer(json.loads(vocab.read_text()))(
        tf.constant(texts.tolist())
    ).numpy()
    assert tuple(ref.columns[2:]) == FEATURE_NAMES and np.array_equal(
        tokens[0], tokens[1]
    )
    write_json(
        ROOT / "feature-example.json",
        {
            "urls": df.url.tolist(),
            "text": texts.tolist(),
            "vocab_sha256": sha(vocab),
            "names": list(ref.columns[2:]),
            "values": ref.iloc[:, 2:].values.tolist(),
            "tokens": tokens.tolist(),
            "equal": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    args = parser.parse_args()
    if args.kind == "example":
        example()
    else:
        seal(args.kind)
