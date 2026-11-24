"""Fetch the pinned public notebook, preprocessing source and matching CSV blobs."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((REPO / "configs/upstream.json").read_text())
    root = f"https://raw.githubusercontent.com/munna0912/URL_CLASSIFIER/{config['commit']}/"
    files = dict(config["source_sha256"])
    files.update(
        {
            "Model_Training.ipynb": "3cb31cba0b55419326ce8f49817959621a8feb5c419e98f0d2d2b91acafcb945",
            "Utils/Evaluation.py": "be8120d8aee7bd18893eeb57c83e419ec19d8a566f2e49fcc0f1cc7589a77ccd",
        }
    )
    assets = {path: (root + path, h) for path, h in files.items()}
    for path, rec in config["files"].items():
        if path.endswith(".csv"):
            assets[path] = rec["recovery_url"], rec["sha256"]
    for path, (url, digest) in assets.items():
        target = destination / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing source differs: " + path)
        else:
            with urlopen(url, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Source mismatch: " + path)
            target.write_bytes(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    fetch(parser.parse_args().destination)
