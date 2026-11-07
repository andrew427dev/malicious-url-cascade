"""Verify actual bytes against the fixed upstream LFS OIDs; never trust size alone."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    if not path.is_file():
        return dict(status='missing', path=str(path))
    with path.open('rb') as f:
        if f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            return dict(status='lfs_pointer_only', path=str(path))
        f.seek(0)
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    size, digest = path.stat().st_size, h.hexdigest()
    return dict(path=str(path), size=size, sha256=digest,
                status='verified' if size == expected['size'] and digest == expected['sha256'] else 'mismatch')


def verify_optional(path, expected):
    try:
        return dict(verify(path, expected), required=False)
    except OSError as exc:
        return dict(status='failed', path=str(path), required=False,
                    error=str(exc), error_type=type(exc).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument('--internal-only', action='store_true')
    scope.add_argument('--required-data-only', action='store_true',
                       help='Require all three CSVs; report public weights separately.')
    args = p.parse_args()
    manifest = json.loads((ROOT/'configs/upstream.json').read_text())
    selected = manifest['files']
    if args.internal_only:
        selected = {k: v for k, v in selected.items() if k in (
            'dataset/df_final.csv', 'dataset/all_top_domains_merged.csv')}
    elif args.required_data_only:
        selected = {k: v for k, v in selected.items() if not k.endswith('.keras')}
    results = {name: verify(args.root/name, info) for name, info in selected.items()}
    report = dict(commit=manifest['commit'], results=results)
    if args.required_data_only:
        report['optional_results'] = {
            name: verify_optional(args.root/name, info)
            for name, info in manifest['files'].items() if name.endswith('.keras')}
    report['required_ok'] = all(r['status'] == 'verified' for r in results.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['required_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
