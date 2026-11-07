"""Build a private Kaggle evaluation notebook for one verified B model archive."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(
    model_notebook,
    version,
    digest,
    reference_notebook=None,
    *,
    stage="gate",
    gate_notebook=None,
    gate_version=None,
    gate_digest=None,
):
    if stage not in ("gate", "final"):
        raise ValueError("Expected a gate or final notebook")
    handles = [model_notebook]
    if stage == "final":
        if not (
            gate_version
            and gate_version > 0
            and gate_digest
            and re.fullmatch("[a-f0-9]{64}", gate_digest)
            and reference_notebook
            and gate_notebook
        ):
            raise ValueError(
                "Final evaluation requires the verified gate version and archive, and the A notebook"
            )
        handles.extend([gate_notebook, reference_notebook])
    for handle in handles:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", handle):
            raise ValueError("Expected a Kaggle notebook handle in owner/slug format")
    if version < 1 or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError(
            "Supply the successful B version number and its verified ZIP SHA-256"
        )
    # Keep the already-submitted training notebooks immutable.
    source = json.loads(
        (ROOT / "notebooks/HUFS_CompSec_Experiment_B.ipynb").read_text()
    )
    cells = source["cells"][:4]
    cells[0]["source"] = [
        {"gate": "# 실험 B 게이트 선택", "final": "# 동결 정책의 최종 평가"}[stage]
    ]
    cells[1]["source"] = [f"STAGE = {stage!r}"]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((ROOT / "evaluation").glob("*.py")):
            if p.name != "build_notebook.py":
                info = zipfile.ZipInfo(
                    "evaluation/" + p.name, date_time=(2026, 11, 7, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, p.read_bytes())
    raw = payload.getvalue()
    code = f"""import sys, io, zipfile, base64, hashlib
raw = base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest() == {hashlib.sha256(raw).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for entry in archive.infolist():
        name = PurePosixPath(entry.filename)
        assert len(name.parts) == 2 and name.parts[0] == 'evaluation' and name.suffix == '.py'
        location = CODE / name
        location.parent.mkdir(exist_ok=True)
        location.write_bytes(archive.read(entry))
sys.path.insert(0, str(CODE))
from evaluation.kaggle_run import main
main({stage!r}, {model_notebook!r}, {version}, {digest!r}, {reference_notebook!r}, {gate_notebook!r}, {gate_version!r}, {gate_digest!r})
"""
    cells.append(
        dict(
            cell_type="code",
            metadata={},
            execution_count=None,
            outputs=[],
            source=code.splitlines(True),
        )
    )
    return dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(name="python3", display_name="Python 3", language="python")
        ),
        nbformat=4,
        nbformat_minor=4,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-notebook", required=True)
    p.add_argument("--stage", choices=["gate", "final"], required=True)
    p.add_argument("--gate-notebook")
    p.add_argument("--gate-version", type=int)
    p.add_argument("--gate-sha256")
    p.add_argument("--reference-notebook")
    p.add_argument("--model-version", type=int, required=True)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    notebook = build(
        a.model_notebook,
        a.model_version,
        a.archive_sha256,
        a.reference_notebook,
        stage=a.stage,
        gate_notebook=a.gate_notebook,
        gate_version=a.gate_version,
        gate_digest=a.gate_sha256,
    )
    with a.output.open("x") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
