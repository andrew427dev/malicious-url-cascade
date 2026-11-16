"""Package code, reports, weights and audit evidence; omit the large URL corpus."""

import hashlib
import json
import zipfile
from common import CODE_DIR, REPOSITORY, ROOT, SOURCE, sha, write_json


def main():
    audit = json.loads((ROOT / "result-audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["learning_curves"]["status"] != "PASS":
        raise RuntimeError("unverified_results")
    diagnostics = json.loads((ROOT / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics["status"] != "PASS" or diagnostics["result_audit_sha256"] != sha(
        ROOT / "result-audit.json"
    ):
        raise RuntimeError("unverified_or_stale_diagnostics")
    target = ROOT / "재실험 코드와 검증 결과.zip"
    names = {}
    run_prefix = "runs/clean-rerun-shared-validation/"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def add(path, name):
            if name in names:
                raise ValueError("duplicate_archive_member:" + name)
            archive.write(path, name)
            names[name] = sha(path)

        for path in sorted(CODE_DIR.glob("*.py")):
            add(path, "experiments/clean_rerun/" + path.name)
        for name in ("README.md", "requirements-experiment.txt", "pyproject.toml", "protocol.json"):
            add(CODE_DIR / name, "experiments/clean_rerun/" + name)
        add(CODE_DIR / "README.md", "README.md")
        for name in ("test_clean_rerun_contract.py", "test_clean_rerun_evaluation.py"):
            add(REPOSITORY / "tests" / name, "tests/" + name)
        for name in (
            "scripts/fetch_assets.py",
            "scripts/verify_assets.py",
            "scripts/fetch_reference.py",
            "configs/upstream.json",
        ):
            add(REPOSITORY / name, name)
        for name in (
            "urlcascade/__init__.py",
            "urlcascade/features.py",
            "urlcascade/adapters.py",
            "upstream/Utils/Model.py",
            "configs/upstream.json",
        ):
            add(SOURCE / name, "reference_code/" + name)
        add(
            SOURCE / "evidence/experiment-a-internal.json",
            "reference_code/evidence/experiment-a-internal.json",
        )
        for name in (
            "environment.json",
            "data-audit.json",
            "result-audit.json",
            "diagnostics.json",
            "policy-synthetic-checks.json",
            "upstream-reproduction-source.json",
            "source-snapshot.json",
            "evaluation-source-snapshot.json",
            "data-source-snapshot.json",
            "protocol.json",
            "learning-curves.json",
            "learning-curve-indices.npz",
        ):
            add(ROOT / name, run_prefix + name)
        for kind in ("random", "domain"):
            for path in sorted((ROOT / kind).rglob("*")):
                if path.is_file():
                    add(path, run_prefix + kind + "/" + path.relative_to(ROOT / kind).as_posix())
            for path in sorted((ROOT / "previous-results" / kind).glob("*.json")):
                add(path, run_prefix + "previous-results/" + kind + "/" + path.name)
        for stage in ("data-before-token-audit", "data", "data-final"):
            add(ROOT / stage / "manifest.json", run_prefix + stage + "/manifest.json")
        revision = ROOT / "evaluation-revision.json"
        if revision.exists():
            add(revision, run_prefix + revision.name)
            add(
                ROOT / "verification/prepare_evaluation_revision.py",
                run_prefix + "verification/prepare_evaluation_revision.py",
            )
        for path in sorted(ROOT.glob("figure-*.svg")):
            add(path, "docs/clean-rerun/" + path.name)
        for suffix in (".pdf", ".md"):
            add(
                ROOT / ("악성 URL 캐스케이드 정제 데이터 재실험 보고서" + suffix),
                "docs/clean-rerun/report-ko" + suffix,
            )
        for path in sorted((ROOT / "template").rglob("*")):
            if path.is_file():
                add(path, run_prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(
                {
                    "files": names,
                    "omitted_large_data": [
                        "raw CSVs",
                        "clean.csv",
                        "numeric and encoded arrays",
                        "excluded-row logs",
                    ],
                    "note": "Reports and frozen weights/scores are included. Regenerate data in a new run directory to train or rerun the complete data audit.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, digest in names.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    write_json(
        ROOT / "package-verification.json",
        {
            "status": "PASS",
            "members": len(names) + 1,
            "sha256": sha(target),
            "bytes": target.stat().st_size,
        },
    )
    print(target)


if __name__ == "__main__":
    main()
