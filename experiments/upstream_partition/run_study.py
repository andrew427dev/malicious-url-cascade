"""Reproduce the full offline experiment under a new directory."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
KINDS = ("random-original", "domain-original", "random-stripped", "domain-stripped")


def run(script, *args):
    subprocess.run(
        [sys.executable, "-u", str(HERE / (script + ".py")), *map(str, args)],
        check=True,
    )


def finish():
    for kind in KINDS:
        run("evaluate", kind, "--freeze")
    run("evaluate", "author-validation")
    for kind in KINDS:
        run("evaluate", kind)
    run("evaluate", "author-examples")
    for group in ("random", "domain"):
        run("timing", group)
    run("verify_results")


def main(destination):
    os.environ["URLCASCADE_RUN_DIR"] = str(destination.resolve())
    run("fetch", destination)
    run("inputs", "parity")
    for kind in ("author-validation", "author-final"):
        run("inputs", kind)
        run("train_gru", kind)
    run("team_data", "pool")
    for kind in KINDS:
        run("team_data", kind)
        run("audit_data", kind)
        run("train_gru", kind)
        run("fit_stages", kind)
    run("provenance", "example")
    finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("new_directory", type=Path)
    main(parser.parse_args().new_directory)
