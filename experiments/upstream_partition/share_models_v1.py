"""Archive this run's fitted models and row-level scores with verifiable relative paths."""
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(os.environ["URLCASCADE_RUN_DIR"]).resolve()


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        value.update(block)
    return value.hexdigest()


def archive(title, files):
    output = ROOT / "share" / (title + " v1.zip")
    hashes = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=3) as writer:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            with path.open("rb") as source:
                hashes[name] = digest(source)
            writer.write(path, name)
        writer.writestr("SHA256.json", json.dumps(hashes, indent=2, ensure_ascii=False))
    with ZipFile(output) as reader:
        for name, expected in hashes.items():
            with reader.open(name) as source:
                assert digest(source) == expected, name
    print(title, len(hashes), output.stat().st_size, flush=True)


def main():
    audit = json.loads((ROOT / "results/verification.json").read_text(encoding="utf8"))
    assert audit["status"] == "PASS" and len(audit["conditions"]) == 8
    models, scores = [], []
    for name in [*audit["conditions"], "author-validation", "author-final"]:
        cohort = ROOT / name
        models.extend(cohort / name for name in ("gru.h5", "vocab.json", "epochs.csv", "trained.json", "training-inputs.json"))
        if (cohort / "routing.json").exists():
            models.append(cohort / "routing.json")
            models.extend((cohort / "stage1").glob("*"))
        scores.extend(cohort.glob("*scores.npz"))
    assert len(scores) == 14, "8 primary + 4 fixed-prefix + 2 author cohorts"
    archive("새 학습 모델·어휘·보정기·분기 기록", models)
    archive("시험 URL별 정답·예측 점수", scores)


if __name__ == "__main__":
    main()
