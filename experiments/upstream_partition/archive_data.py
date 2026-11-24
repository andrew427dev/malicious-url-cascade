"""Export the actual training cohorts, then compare every archived row and tensor."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

PREFIX = re.compile(r"^https?://", re.IGNORECASE)
ROLES = {
    "author-validation": ("train", "validation", "external"),
    "author-final": ("train",),
    **{
        f"{group}-{form}": ("train", "validation", "test")
        for group in ("random", "domain")
        for form in ("original", "stripped")
    },
}


def digest(stream):
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(block)
    return checksum.hexdigest()


def rows_for(src, role, stripped):
    df = pd.read_pickle(src / (role + ".pkl"))
    ids = pd.read_csv(src / (role + "-rows.csv.gz"), index_col="id")
    assert df.index.equals(ids.index) and np.array_equal(df.type, ids.type)
    np.testing.assert_array_equal(ids.y_mal, 1 - df.type)
    rows = pd.DataFrame(
        {
            "row_order": np.arange(1, len(df) + 1, dtype=np.int64),
            "id": df.index,
            "type": df.type.to_numpy(),
            "y_mal": ids.y_mal.to_numpy(),
            "raw_url": df.url.to_numpy(),
            "model_url": (
                df.url.str.replace(PREFIX, "", n=1, regex=True) if stripped else df.url
            ).to_numpy(),
            "processed_text": df.text.to_numpy(),
        }
    )
    if "domain" in df:
        rows["registered_domain_group"] = df.domain.to_numpy()
    return rows


def package(root, dest, kind):
    source = root / kind
    target = dest / (kind + ".zip")
    cohorts = {}
    with ZipFile(
        target, "x", ZIP_DEFLATED, compresslevel=3, allowZip64=True
    ) as archive:
        for role in ROLES[kind]:
            rows = rows_for(source, role, kind.endswith("-stripped"))
            with archive.open(role + "/rows.csv", "w", force_zip64=True) as stream:
                with io.TextIOWrapper(stream, encoding="utf8", newline="") as writer:
                    rows.to_csv(
                        writer, index=False, lineterminator="\n", chunksize=10000
                    )
            cohorts[role] = {"rows": len(rows), "malicious": int(rows.y_mal.sum())}
            for item in ("features.npy", "tokens.npy"):
                archive.write(source / (role + "-" + item), role + "/" + item)
        for name in ("vocab.json", "data.json", "training-inputs.json"):
            archive.write(source / name, name)
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive member failed CRC check")
        for role in cohorts:
            expected = rows_for(source, role, kind.endswith("-stripped"))
            offset = 0
            with archive.open(role + "/rows.csv") as stream:
                for chunk in pd.read_csv(
                    stream,
                    chunksize=20000,
                    keep_default_na=False,
                    dtype={
                        c: "object"
                        for c in expected.select_dtypes(include="object").columns
                    },
                ):
                    part = expected.iloc[offset : offset + len(chunk)]
                    pd.testing.assert_frame_equal(
                        chunk.reset_index(drop=True), part.reset_index(drop=True)
                    )
                    offset += len(chunk)
            assert offset == len(expected)
            for item in ("features.npy", "tokens.npy"):
                with (
                    archive.open(role + "/" + item) as copy,
                    (source / (role + "-" + item)).open("rb") as original,
                ):
                    assert digest(copy) == digest(original)
        for name in ("vocab.json", "data.json", "training-inputs.json"):
            assert archive.read(name) == (source / name).read_bytes()
    with target.open("rb") as stream:
        return {
            "file": target.name,
            "size": target.stat().st_size,
            "sha256": digest(stream),
            "roles": cohorts,
        }


def main(root, dest):
    dest.mkdir(exist_ok=False, parents=True)
    records = []
    for kind in ROLES:
        print("Exporting", kind, flush=True)
        records.append(package(root, dest, kind))
    (dest / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("new_archive_dir", type=Path)
    args = parser.parse_args()
    main(args.run, args.new_archive_dir)
