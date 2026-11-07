"""Run on verified cloud compute (or legacy submitted job), after W1."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import pkgutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urlcascade.data import audit, read_rows, DomainGrouper, freeze_folds, validate_partitions, sha256, write_jsonl
from verify_assets import verify
from urlcascade.runtime import compute_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime', choices=['auto','kaggle','colab','school'], default='auto')
    args = p.parse_args()
    runtime = compute_context(args.runtime, args.out)
    versions = {name: importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','tldextract')}
    if versions['scikit-learn'] != '1.2.2' or versions['tldextract'] != '5.4.0':
        raise RuntimeError('Pinned splitter/parser versions required')
    manifest = json.loads((ROOT/'configs/upstream.json').read_text())
    internals = ['dataset/df_final.csv', 'dataset/all_top_domains_merged.csv']
    verified = {name: verify(args.assets/name, manifest['files'][name]) for name in internals}
    if any(r['status'] != 'verified' for r in verified.values()):
        raise RuntimeError('Internal data integrity failed: ' + json.dumps(verified))
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = pkgutil.get_data('tldextract', '.tld_set_snapshot')
    if not snapshot or b'BEGIN PRIVATE DOMAINS' not in snapshot:
        raise RuntimeError('PSL snapshot with private rules required')
    psl = args.out/'public_suffix_list.dat'
    psl.write_bytes(snapshot)
    grouper = DomainGrouper(psl, sha256(psl))
    clean, excluded, keys, groups = audit(read_rows([args.assets/n for n in internals]), grouper)
    partitions = freeze_folds(clean)
    write_jsonl(args.out/'internal.jsonl', partitions)
    write_jsonl(args.out/'internal-excluded.jsonl', excluded)
    # Preserve all internal overlap keys, including quarantined/unknown-label rows.
    (args.out/'internal-overlap.json').write_text(json.dumps(dict(url_keys=sorted(keys), group_ids=sorted(groups))))
    for split in ('train','model_selection','gate_selection','test'):
        write_jsonl(args.out/(split+'.jsonl'), [r for r in partitions if r['split'] == split])
    report = dict(status='frozen', experiment='B', runtime=runtime, source_commit=manifest['commit'],
                  versions=versions, assets=verified, psl_sha256=sha256(psl),
                  splitter=dict(n_splits=10, shuffle=True, random_state=42),
                  sorting=['url_key','input_id'], fold_map={'train':[0,1,2,3,4,5], 'model_selection':[6], 'gate_selection':[7], 'test':[8,9]},
                  summary=validate_partitions(partitions), excluded=len(excluded),
                  files={f.name:sha256(f) for f in sorted(args.out.iterdir()) if f.is_file()},
                  remaining_audit=['Representation collisions', 'External overlap audit before policy-frozen evaluation'])
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
