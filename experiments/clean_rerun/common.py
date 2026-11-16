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


def snapshot_sources():
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
    destination = ROOT / "source-snapshot.json"
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
    snapshot = ROOT / "source-snapshot.json"
    if snapshot.exists():
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
