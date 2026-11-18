"""Paths, immutable execution-source evidence, and JSON/hash utilities."""

import hashlib
import json
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPOSITORY = CODE_DIR.parents[1]
SOURCE = Path(
    os.environ.get(
        "URLCASCADE_SOURCE",
        REPOSITORY / "reference_code" if (REPOSITORY / "reference_code").exists() else REPOSITORY,
    )
).resolve()
ROOT = Path(
    os.environ.get("URLCASCADE_RUN_DIR", REPOSITORY / "runs" / "clean-rerun-shared-validation")
).resolve()
REFERENCE_FILES = ("urlcascade/adapters.py", "urlcascade/features.py", "upstream/Utils/Model.py")
TRAINING_ARTIFACTS = (
    "settings.json",
    "t1-selection.json",
    "gru-selection.json",
    "vocabulary.json",
    "logistic.joblib",
    "boost.joblib",
    "length.joblib",
    "t2.h5",
    "encoded-input-audit.json",
    "progress.jsonl",
    "epochs.csv",
)


def validation_indices(parts):
    """The user-approved shared validation set; test rows are excluded."""
    import numpy as np

    rows = np.sort(np.concatenate([parts[name] for name in ("model", "calibration", "policy")]))
    if (
        len(np.unique(rows)) != len(rows)
        or np.intersect1d(rows, parts["train"]).size
        or np.intersect1d(rows, parts["test"]).size
    ):
        raise RuntimeError("invalid_shared_validation_partition")
    return rows


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def seal_training_artifacts(out):
    """Seal completed training before any validation-policy or test evaluation."""
    destination = out / "training-manifest.json"
    if destination.exists():
        raise RuntimeError("training_manifest_already_exists")
    write_json(
        destination,
        {
            "status": "PASS",
            "seal_origin": "training_completed_before_evaluation",
            "files": {name: sha(out / name) for name in TRAINING_ARTIFACTS},
        },
    )


def verify_training_artifacts(out):
    """Always check every trained model and selection input before loading it."""
    path = out / "training-manifest.json"
    if not path.exists():
        raise RuntimeError("missing_training_manifest")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or set(record.get("files", {})) != set(TRAINING_ARTIFACTS):
        raise RuntimeError("invalid_training_manifest")
    for name, digest in record["files"].items():
        if sha(out / name) != digest:
            raise RuntimeError("training_artifact_changed:" + name)


def snapshot_sources(filename="source-snapshot.json"):
    """Keep original bytes inside JSON so Git line endings cannot change evidence."""
    sources = {}
    execution_files = (
        "common.py",
        "prepare.py",
        "refine_inputs.py",
        "finalize_data.py",
        "experiment.py",
        "learning_curves.py",
    )
    for name, path in [(n, CODE_DIR / n) for n in execution_files] + [
        ("reference/" + n, SOURCE / n) for n in REFERENCE_FILES
    ]:
        raw = path.read_bytes()
        sources[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "utf8": raw.decode("utf-8")}
    destination = ROOT / filename
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != sources:
            raise RuntimeError("execution_source_changed_between_cohorts")
    else:
        write_json(destination, sources)


def verify_execution_source(name, digest, *, reference=False):
    """Verify the code actually used by the recorded run, not a newer source file."""
    current = (SOURCE if reference else CODE_DIR) / name
    if current.exists() and sha(current) == digest:
        return
    for filename in ("source-snapshot.json", "evaluation-source-snapshot.json"):
        snapshot = ROOT / filename
        if not snapshot.exists():
            continue
        records = json.loads(snapshot.read_text(encoding="utf-8"))
        keys = (
            ["reference/" + name]
            if reference
            else [name, "experiment-before-path-fix.py"]
            if name == "experiment.py"
            else [name]
        )
        for key in keys:
            record = records.get(key)
            if (
                record
                and record["sha256"] == digest
                and hashlib.sha256(record["utf8"].encode("utf-8")).hexdigest() == digest
            ):
                return
    raise RuntimeError("unverified_execution_source:" + name)
