"""Read-only input preflight for the complete v2 e0--e6 workflow.

Uses only the standard library: no pickle/joblib loading, model imports, inference, or writes.
The default checks files, JSON and the byte hashes used by the existing loaders. Add
--check-packages to also locate the analysis dependencies without importing them. Passing this
preflight is not a numerical reproduction or a check of binary/package compatibility.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

# Keep even the helper import read-only when this script is invoked without python -B.
sys.dont_write_bytecode = True
from v2paths import folders, refuse_protected

SPLITS = ("random", "domain")
KINDS = tuple(prefix + split + "-original" for prefix in ("", "team-") for split in SPLITS)
TEAM = tuple("team-" + split + "-original" for split in SPLITS)
ROLES = ("train", "validation", "test")
DATASETS = ("df_final.csv", "all_top_domains_merged.csv", "df_another_validation.csv")
PACKAGES = ("numpy", "pandas", "sklearn", "scipy", "tensorflow", "joblib", "h5py",
            "tldextract", "threadpoolctl")
# This is the existing provenance.verify_seal contract, including the pre-assignment flow.
SEAL_REQUIRED = {"vocab.json", "train-tokens.npy", "train-features.npy", "train-rows.csv.gz",
                 "runner/train_gru.py", "source/Utils/Model.py"}


def required_files(v1, code):
    """Inputs read by e0--e6; generated v2 timing/results and report-only inputs are excluded."""
    paths = [v1 / name for name in (
        "source-manifest.json", "results/verification.json", "results/main-28.csv",
        "results/random-common.csv.gz", "results/domain-common.csv.gz",
        "results/review-evidence.json", "results/timing-summary.csv", "results/sensitivity.csv",
        "batch-random.json", "batch-domain.json", "team-random-original/cleaning.json")]
    paths += [code / name for name in (
        "configs/upstream.json", "urlcascade/adapters.py", "urlcascade/features.py",
        "experiments/clean_rerun/common.py", "experiments/clean_rerun/experiment.py",
        "experiments/clean_rerun/prepare.py", "experiments/upstream_partition/inputs.py",
        "experiments/upstream_partition/evaluate.py", "experiments/upstream_partition/cohorts.py",
        "experiments/upstream_partition/provenance.py", "experiments/upstream_partition/aggregate.py",
        "experiments/upstream_partition/paired_time.py")]
    paths += [v1 / "source/dataset" / name for name in DATASETS]
    paths += [v1 / "source/Utils" / name for name in ("DataProcessing.py", "FeatureCreation.py", "Model.py")]
    for kind in KINDS:
        out = v1 / kind
        paths += [out / name for name in (
            "test-rows.csv.gz", "scores.npz", "controls-final.json", "results.json",
            "stage1/t1-selected.json", "stage1/length.joblib")]
        paths += [out / (role + suffix) for role in ROLES
                  for suffix in (".pkl", "-tokens.npy", "-features.npy")]
    for kind in TEAM:
        paths += [v1 / kind / name for name in (
            "validation-rows.csv.gz", "vocab.json", "gru.h5", "trained.json",
            "training-inputs.json", "routing.json", "paired-scores.npz", "fixed-prefix.json",
            "stage1/logistic.joblib", "stage1/boost.joblib")]
    return paths


class Audit:
    def __init__(self):
        self.issues = []
        self.files = {}
        self.json = {}
        self.digests = {}
        self.hash_checks = 0

    def issue(self, kind, path, detail):
        self.issues.append(dict(kind=kind, path=str(path), detail=detail))

    def need(self, path):
        path = Path(path)
        if path not in self.files:
            self.files[path] = path.is_file()
            if not self.files[path]:
                self.issue("missing_file", path, "Required input file is absent.")
        return self.files[path]

    def read_json(self, path):
        path = Path(path)
        if path not in self.json:
            self.json[path] = None
            if self.need(path):
                try:
                    self.json[path] = json.loads(path.read_text(encoding="utf8"))
                except (OSError, UnicodeError, ValueError) as exc:
                    self.json[path] = None
                    self.issue("invalid_json", path, str(exc))
                else:
                    expected = list if path.name == "vocab.json" else dict
                    if not isinstance(self.json[path], expected):
                        self.issue("invalid_record", path, "Expected a JSON " + ("array." if expected is list else "object."))
                        self.json[path] = None
        return self.json[path]

    def mapping(self, path, field):
        rec = self.read_json(path)
        if rec is None:
            return None
        value = rec.get(field) if isinstance(rec, dict) else None
        if not isinstance(value, dict) or not value:
            self.issue("invalid_record", path, "Expected a nonempty mapping: " + field)
            return None
        return value

    def digest(self, path, expected, record):
        path = Path(path)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            self.issue("invalid_record", record, "Invalid SHA-256 for " + str(path))
            return
        if not self.need(path):
            return
        if path not in self.digests:
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                self.digests[path] = digest.hexdigest()
            except OSError as exc:
                self.issue("unreadable_file", path, str(exc))
                self.digests[path] = None
        actual = self.digests[path]
        if actual is not None:
            self.hash_checks += 1
            if actual != expected:
                self.issue("sha256_mismatch", path, "Differs from " + str(record)
                           + "; preserve the recorded bytes, including CRLF/LF. Do not reseal to bypass it.")


def check_seal(audit, v1, code, kind):
    out = v1 / kind
    record = out / "training-inputs.json"
    files = audit.mapping(record, "files")
    if files is None:
        return
    files = {key.replace("\\", "/"): value for key, value in files.items()}
    missing = sorted(SEAL_REQUIRED - files.keys())
    if missing:
        audit.issue("invalid_record", record, "Missing existing seal keys: " + ", ".join(missing))
    rec = audit.read_json(record)
    if not isinstance(rec.get("capture"), str):
        audit.issue("invalid_record", record, "Missing capture field.")
        return
    runner = code / "experiments/upstream_partition" if rec["capture"] == "before_fit" else v1 / "started-training-source"
    for key, digest in files.items():
        prefix, _, tail = key.partition("/")
        base = {"runner": runner, "team": code, "source": v1 / "source"}.get(prefix)
        path = base / tail if base is not None else out / key
        audit.digest(path, digest, record)


def check(v1, v2, code, check_packages=False):
    """Return a report without creating directories or importing scientific packages."""
    v1, v2, code = map(Path, (v1, v2, code))
    audit = Audit()
    report = dict(scope="v1 inputs for e0-e6 (no report generation or numerical execution)",
                  v1=str(v1), v2=str(v2), code=str(code), packages_checked=False,
                  packages=[], issues=audit.issues)
    try:
        refuse_protected(v2 / "results")
        refuse_protected(v2 / "report")
    except SystemExit as exc:
        audit.issue("protected_output", v2, str(exc))
        report["status"] = "FAIL"
        return report
    for path in required_files(v1, code):
        if path.suffix == ".json":
            audit.read_json(path)
        else:
            audit.need(path)
    verification = audit.read_json(v1 / "results/verification.json")
    if verification is not None and (not isinstance(verification, dict) or verification.get("status") != "PASS"):
        audit.issue("invalid_record", v1 / "results/verification.json", "Expected the v1 verification status PASS.")

    config_path = code / "configs/upstream.json"
    sources = audit.mapping(config_path, "source_sha256")
    if sources:
        for name, digest in sources.items():
            audit.digest(v1 / "source" / name, digest, config_path)
    datasets = audit.mapping(config_path, "files")
    if datasets:
        for name in DATASETS:
            record = datasets.get("dataset/" + name, {})
            digest = record.get("sha256") if isinstance(record, dict) else None
            audit.digest(v1 / "source/dataset" / name, digest, config_path)

    for kind in KINDS:
        record = v1 / kind / "stage1/t1-selected.json"
        hashes = audit.mapping(record, "hashes")
        if hashes is not None:
            required = {"length", "logistic", "boost"} if kind in TEAM else {"length"}
            for name in sorted(required - hashes.keys()):
                audit.issue("invalid_record", record, "Missing stage-one hash: " + name)
            # load_stages checks every recorded joblib, including the omit controls.
            for name, digest in hashes.items():
                if kind in TEAM or name == "length":
                    audit.digest(v1 / kind / "stage1" / (name + ".joblib"), digest, record)
    for kind in TEAM:
        check_seal(audit, v1, code, kind)
        record = v1 / kind / "trained.json"
        trained = audit.read_json(record)
        if trained is not None:
            for name, field in (("gru.h5", "weights_sha256"), ("vocab.json", "vocabulary_sha256")):
                digest = trained.get(field) if isinstance(trained, dict) else None
                audit.digest(v1 / kind / name, digest, record)

    if check_packages:
        report["packages_checked"] = True
        for name in PACKAGES:
            available = importlib.util.find_spec(name) is not None
            report["packages"].append(dict(module=name, available=available))
            if not available:
                audit.issue("missing_package", name, "Not found by Python module discovery; install the documented environment.")
    report.update(status="FAIL" if audit.issues else "PASS", files_checked=len(audit.files),
                  json_checked=len(audit.json), recorded_hashes_checked=audit.hash_checks,
                  limitations=["No array/pickle/joblib deserialization, row alignment, or numerical checks.",
                               "Package discovery does not test imports, versions, or binary compatibility.",
                               "Generated v2 e6 timing files must exist before e6 summarize; they are not initial inputs."])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-packages", action="store_true", help="also locate the packages needed to execute e0-e6")
    args = parser.parse_args(argv)
    try:
        v1, v2, code = folders()
    except SystemExit as exc:
        report = dict(status="FAIL", issues=[dict(kind="folder_configuration", detail=str(exc))])
    else:
        report = check(v1, v2, code, args.check_packages)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
