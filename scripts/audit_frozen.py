"""Audit a frozen internal split and freeze the independent external cohort."""
from collections import Counter
import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urlcascade.adapters import preprocess_link
from urlcascade.data import DomainGrouper, audit, exclude_external, read_rows, sha256, write_jsonl
from urlcascade.features import extract_features
from urlcascade.runtime import compute_context
from verify_assets import verify


def json_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def summary(rows):
    return dict(rows=len(rows), classes=dict(Counter(str(r['y_original']) for r in rows)),
                groups=len({r['group_id'] for r in rows}))


def collision_stats(counts):
    return dict(collision_keys=sum(n > 1 for n in counts.values()),
                rows_in_collisions=sum(n for n in counts.values() if n > 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime', choices=['kaggle','colab'], required=True)
    args = p.parse_args()
    compute_context(args.runtime, args.out)
    split = args.split_dir
    manifest = json.loads((split/'manifest.json').read_text())
    if manifest.get('status') != 'frozen' or manifest.get('experiment') != 'B':
        raise ValueError('expected frozen B split')
    for name, expected in manifest['files'].items():
        if sha256(split/name) != expected:
            raise ValueError('frozen file changed: ' + name)
    upstream = json.loads((ROOT/'configs/upstream.json').read_text())
    external = 'dataset/df_another_validation.csv'
    source = verify(args.assets/external, upstream['files'][external])
    if source['status'] != 'verified':
        raise ValueError('external source failed hash check')
    args.out.mkdir(parents=True, exist_ok=False)
    text_counts, pre_vocab_inputs = Counter(), Counter()
    split_groups = {name:set() for name in ('train','model_selection','gate_selection','test')}
    retained_classes = Counter()
    for row in json_rows(split/'internal.jsonl'):
        raw = row['raw_url']
        text = preprocess_link(raw)
        text_counts[text] += 1
        # 50 character trigrams use at most the first 52 characters.
        prefix = text[:52] if len(text) >= 3 else ''
        pre_vocab_inputs[(prefix, tuple(extract_features(raw)))] += 1
        retained_classes[str(row['y_original'])] += 1
        split_groups[row['split']].add(row['group_id'])
    internal = dict(retained_by_original_label=dict(retained_classes),
                    cleaned_text=collision_stats(text_counts),
                    trigram_prefix_and_numeric_before_vocabulary=collision_stats(pre_vocab_inputs))
    del text_counts, pre_vocab_inputs
    intersections = {a + '/' + b: len(split_groups[a] & split_groups[b])
                     for a,b in combinations(split_groups, 2)}
    if any(intersections.values()):
        raise ValueError('group overlap')
    internal['pairwise_intersections'] = intersections
    reasons = Counter((r['exclusion_reason'], str(r['y_original'])) for r in json_rows(split/'internal-excluded.jsonl'))
    internal['exclusions'] = [dict(reason=k[0],original_label=k[1],rows=v) for k,v in sorted(reasons.items())]
    raw_counts = {}
    for asset in upstream['files']:
        if asset.endswith('.csv'):
            raw_counts[asset] = dict(Counter(str(r['y_original']) for r in read_rows([args.assets/asset])))
    overlap = json.loads((split/'internal-overlap.json').read_text())
    grouper = DomainGrouper(split/'public_suffix_list.dat', manifest['psl_sha256'])
    clean, rejected, _, _ = audit(read_rows([args.assets/external]), grouper)
    kept, removed = exclude_external(clean, set(overlap['url_keys']), set(overlap['group_ids']))
    if set(r['url_key'] for r in kept).intersection(overlap['url_keys']) or set(r['group_id'] for r in kept).intersection(overlap['group_ids']):
        raise ValueError('external overlap after exclusion')
    write_jsonl(args.out/'external.jsonl', kept)
    write_jsonl(args.out/'external-excluded.jsonl', [*rejected,*removed])
    report = dict(source_commit=upstream['commit'], split_manifest_sha256=sha256(split/'manifest.json'),
                  raw_by_file_and_original_label=raw_counts, internal=internal, external_source=source,
                  external_before_internal_overlap=summary(clean), external_kept=summary(kept),
                  external_excluded=dict(Counter(r['exclusion_reason'] for r in [*rejected,*removed])),
                  final_vectorized_input_check='After selection of the B training vocabulary; OOV may introduce additional collisions.',
                  files={f.name:sha256(f) for f in sorted(args.out.iterdir())})
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
