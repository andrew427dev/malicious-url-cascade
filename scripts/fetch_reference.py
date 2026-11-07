"""Materialize the three verified Python modules needed for model reproduction."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch_reference(root=ROOT):
    spec = json.loads((root / 'configs/upstream.json').read_text())
    base = f"https://raw.githubusercontent.com/munna0912/URL_CLASSIFIER/{spec['commit']}/"
    for name, digest in spec['source_sha256'].items():
        target = root / 'upstream' / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Existing reference file differs from the pinned version: {name}')
            continue
        with urlopen(base + name, timeout=30) as response:
            data = response.read(1024 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Reference download failed integrity check: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
            temp.write(data)
        staging = Path(temp.name)
        try:
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)


if __name__ == '__main__':
    fetch_reference()
