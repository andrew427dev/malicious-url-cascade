"""Notebook-kernel launcher for gate selection and policy-frozen evaluation."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from scripts.cloud_run import child_environment
from urlcascade.data import sha256
from urlcascade.runtime import compute_context

ROOT = Path(__file__).resolve().parents[1]
A_HASH = "bfcab02a621e757965929cf0c6fd925b853d53c5edbc0a10b8847d26771ce4ca"


def private_output(notebook, version, name, expected):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) > 1:
        raise ValueError(f"Multiple attached inputs named {name}")
    if matches:
        path = matches[0]
    else:
        import kagglehub

        path = Path(
            kagglehub.notebook_output_download(
                f"{notebook}/versions/{version}", path=name
            )
        )
        if path.is_dir():
            files = list(path.rglob(name))
            if len(files) != 1:
                raise ValueError(f"Expected exactly one downloaded {name}")
            path = files[0]
    if sha256(path) != expected:
        raise ValueError(f"Unexpected {name} from {notebook} version {version}")
    return path


def extract_member(archive, suffix, dest):
    names = [n for n in archive.namelist() if n.endswith("/" + suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one archive member with suffix {suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(names[0]) as src, dest.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def main(
    stage_name,
    model_notebook,
    model_version,
    archive_hash,
    reference_notebook,
    gate_notebook=None,
    gate_version=None,
    gate_archive_hash=None,
):
    from datetime import datetime, timezone

    work = Path("/kaggle/working/hufs-compsec-url-cascade")
    compute_context("kaggle", work)
    out = Path("/kaggle/working/evaluation-output")
    out.mkdir(exist_ok=False)
    if stage_name not in ("gate", "final"):
        raise ValueError("Expected a gate or final job")
    env = child_environment(work)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    py = work / "env/bin/python"

    def command(args):
        print("RUN", list(map(str, args)), flush=True)
        subprocess.run(args, env=env, check=True, cwd=ROOT)

    def stage(name, *extra):
        run_id = (
            name + "-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        command(
            [
                sys.executable,
                ROOT / "scripts/cloud_run.py",
                "--runtime",
                "kaggle",
                "--work",
                work,
                "--stage",
                name,
                "--cpu-only",
                "--run-id",
                run_id,
                *extra,
            ]
        )
        run_dir = work / "runs" / run_id
        shutil.copy2(run_dir / "status.json", out / (name + "-status.json"))
        return run_dir

    setup = stage("setup")
    for name in ("environment.json", "requirements-resolved.txt"):
        shutil.copy2(setup / name, out / name)
    stage("fetch")
    split = stage("prepare-b") / "split"
    models = work / "selected-b"
    b_zip = private_output(
        model_notebook,
        model_version,
        "hufs-b-frozen-models.zip",
        archive_hash,
    )
    with zipfile.ZipFile(b_zip) as archive:
        for name in (
            "models-frozen.json",
            "t1.joblib",
            "t2.h5",
            "b-vocab.json",
            "t1-candidates.json",
            "b-epochs.csv",
        ):
            extract_member(archive, "experiment-b/" + name, models / name)
    gate = out / "gate"
    if stage_name == "gate":
        command(
            [
                py,
                ROOT / "evaluation/gate.py",
                "--models",
                models,
                "--models-sha256",
                sha256(models / "models-frozen.json"),
                "--split",
                split,
                "--out",
                gate,
            ]
        )
    else:
        with zipfile.ZipFile(
            private_output(
                gate_notebook, gate_version, "hufs-b-gate-policy.zip", gate_archive_hash
            )
        ) as archive:
            for name in ("policy.json", "gate-candidates.json", "gate-latencies.json"):
                extract_member(archive, "gate/" + name, gate / name)
        from evaluation.gate import verify_file, load_policy

        policy_path = gate / "policy.json"
        policy_meta = load_policy(policy_path, sha256(policy_path))
        verify_file(models / "models-frozen.json", policy_meta["models_sha256"])
        for name in ("gate-candidates.json", "gate-latencies.json"):
            verify_file(gate / name, policy_meta["evidence_sha256"][name])
    policy = gate / "policy.json"
    policy_hash = sha256(policy)
    (out / "policy.sha256").write_text(policy_hash + "\n")
    print("Frozen B policy", policy_hash, flush=True)
    common = ["--policy", policy, "--policy-sha256", policy_hash]
    command(
        [
            py,
            ROOT / "evaluation/predict.py",
            "--models",
            models,
            *common,
            "--url",
            "https://example.org/demo",
        ]
    )
    if stage_name == "gate":
        destination = out.parent / "hufs-b-gate-policy.zip"
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, "evaluation/" + str(path.relative_to(out)))
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip():
                raise ValueError("Gate archive CRC failed")
        print("Verified gate archive", sha256(destination), flush=True)
        cleanup(work)
        return
    # Open holdouts for evaluation only after the policy has been written.
    audit = stage("audit-b", "--split-dir", split) / "audit"
    command(
        [
            py,
            ROOT / "evaluation/final.py",
            "--models",
            models,
            "--split",
            split,
            "--external",
            audit / "external.jsonl",
            *common,
            "--out",
            out / "final",
        ]
    )
    command(
        [
            py,
            ROOT / "evaluation/representations.py",
            "--models",
            models,
            "--split",
            split,
            *common,
            "--out",
            out / "representation-audit.json",
        ]
    )
    a_dir = work / "historical-a"
    with zipfile.ZipFile(
        private_output(reference_notebook, 1, "hufs-experiment-a.zip", A_HASH)
    ) as archive:
        for name in ("latest.h5", "vocab.json"):
            extract_member(archive, "experiment-a/" + name, a_dir / name)
    command(
        [
            py,
            ROOT / "evaluation/historical.py",
            "--model",
            a_dir / "latest.h5",
            "--vocab",
            a_dir / "vocab.json",
            "--csv",
            work / "data/upstream/dataset/df_another_validation.csv",
            *common,
            "--out",
            out / "a-external.json",
        ]
    )
    for directory in (out / "gate", out / "final/test", out / "final/external"):
        for path in sorted(directory.glob("*")):
            print("OUTPUT", str(path.relative_to(out)), sha256(path), flush=True)
    cleanup(work)


def cleanup(work):
    # Kaggle retains the evaluation reports. Versioned model/split inputs already exist upstream.
    staging = Path(tempfile.mkdtemp(prefix="hufs-eval-"))
    shutil.move(str(work), str(staging / "workspace"))
