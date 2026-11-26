"""Resolve the v1 and v2 folders and refuse to write inside a v1 run folder.

Importing this module has no side effects. `folders()` and `refuse_protected()` raise SystemExit
with a reason code in square brackets: missing_env, v1_not_run_folder, v1_protected.
"""

import os
from pathlib import Path

MARKER = "source-manifest.json"
HERE = Path(__file__).resolve().parent
# In the repository the v1 modules live in experiments/upstream_partition. In the original output
# layout (final-v1/code next to final-v2/code) they are the code snapshot inside the v1 run folder.
IN_REPO = (HERE.parents[1] / "urlcascade").is_dir()


def _env(variable):
    value = os.environ.get(variable)
    return Path(value).resolve() if value else None


def _missing(variable):
    raise SystemExit(f"refused[missing_env]: set {variable} to the folder (see README.md).")


def folders():
    """Return (V1, V2, CODE) resolved from the layout and the two environment variables."""
    v1, v2 = _env("URLCASCADE_V1_DIR"), _env("URLCASCADE_V2_DIR")
    if IN_REPO:
        if v2 is None:
            _missing("URLCASCADE_V2_DIR")
        if v1 is None:
            _missing("URLCASCADE_V1_DIR")
        code = HERE.parents[1]
    else:
        if v1 is None and v2 is not None:
            _missing("URLCASCADE_V1_DIR")
        if v2 is None:
            v2 = HERE.parent
        if v1 is None:
            v1 = (v2.parent / "final-v1").resolve()
        code = v1 / "code"
    if not ((v1 / MARKER).is_file() and (v1 / "results/verification.json").is_file()):
        raise SystemExit(
            f"refused[v1_not_run_folder]: {v1} needs {MARKER} and results/verification.json."
        )
    return v1, v2, code


def refuse_protected(target):
    """Return the resolved target, refusing it and every folder above it that holds the v1 marker."""
    t = Path(target).resolve()
    for d in (t, *t.parents):
        if (d / MARKER).is_file():
            raise SystemExit(f"refused[v1_protected]: {target} is inside v1 run folder {d}")
    return t
