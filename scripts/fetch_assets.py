"""Fixed upstream assets only. Never fetch any URL occurring in a dataset."""
import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from verify_assets import ROOT, verify


def fetch_one(dest, name, info, commit):
    if dest.exists():
        check = verify(dest, info)
        if check['status'] != 'verified':
            raise RuntimeError('Existing asset is not valid; inspect it explicitly: ' + str(dest))
        return check
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix+'.partial')
    url = 'https://media.githubusercontent.com/media/munna0912/URL_CLASSIFIER/' + commit + '/' + name
    # Atomic promotion only after SHA-256 and size both match. Existing partials
    # remain evidence and are never overwritten silently on retry.
    try:
        response = urlopen(url, timeout=30)
    except HTTPError as exc:
        if exc.code not in (403, 404) or 'recovery_url' not in info:
            raise
        # Fixed, independently verified copy. The target hash remains the LFS OID.
        url = info['recovery_url']
        response = urlopen(url, timeout=60)
    with response, temp.open('xb') as out:
        count = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > info['size']:
                raise RuntimeError('asset_exceeds_expected_size:' + name)
            out.write(block)
    check = verify(temp, info)
    if check['status'] != 'verified':
        raise RuntimeError('asset_integrity_failed:' + name)
    os.replace(temp, dest)
    return dict(check, path=str(dest), source=url)


def fetch_manifest(manifest, dest, include_reference_model=False):
    report = dict(commit=manifest['commit'], results={}, optional_results={})
    # Finish and validate required CSVs before attempting optional public weights.
    ordered = sorted(manifest['files'].items(), key=lambda item: item[0].endswith('.keras'))
    for name, info in ordered:
        optional = name.endswith('.keras')
        results = report['optional_results'] if optional else report['results']
        if optional and not include_reference_model:
            results[name] = dict(status='not_requested', required=False)
            continue
        try:
            results[name] = fetch_one(dest/name, name, info, manifest['commit'])
        except Exception as exc:
            results[name] = dict(status='failed', path=str(dest/name),
                                 error=str(exc), error_type=type(exc).__name__)
        results[name]['required'] = not optional
        if not optional and results[name]['status'] != 'verified':
            break
    required = [name for name in manifest['files'] if not name.endswith('.keras')]
    report['required_ok'] = all(report['results'].get(name, {}).get('status') == 'verified'
                                for name in required)
    report['status'] = 'required_assets_verified' if report['required_ok'] else 'failed'
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--include-reference-model', action='store_true')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    manifest = json.loads((ROOT/'configs/upstream.json').read_text())
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    report = fetch_manifest(manifest, args.dest, args.include_reference_model)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))
    return 0 if report['required_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
