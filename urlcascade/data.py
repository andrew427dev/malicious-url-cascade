"""B-only audit and frozen grouping. Original A pipeline stays separate."""
from __future__ import annotations
import csv
import hashlib
import ipaddress
import json
import re
from pathlib import Path
from .core import hostname_for_grouping
from .features import extract_features


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class DomainGrouper:
    def __init__(self, snapshot, expected_sha):
        import tldextract
        snapshot = Path(snapshot).resolve()
        if sha256(snapshot) != expected_sha:
            raise ValueError('psl_hash_mismatch')
        # Only this local file is read. No cached/global/online fallback.
        self.extract = tldextract.TLDExtract(
            suffix_list_urls=(snapshot.as_uri(),), cache_dir=None,
            fallback_to_snapshot=False, include_psl_private_domains=True)

    def __call__(self, raw):
        host = hostname_for_grouping(raw)
        try:
            return 'ip:' + ipaddress.ip_address(host).compressed
        except ValueError:
            pass
        result = self.extract(host)
        if not result.domain or not result.suffix:
            raise ValueError('no_registered_domain')
        return 'domain:' + result.domain + '.' + result.suffix


def read_rows(paths):
    for path in sorted(map(Path, paths)):
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'url', 'type'} <= set(reader.fieldnames or []):
                raise ValueError('unexpected_columns:' + path.name)
            for line, row in enumerate(reader, 2):
                yield dict(input_id=f'{path.name}:{line}', raw_url=row['url'],
                           y_original=row['type'], source=path.name,
                           original_source='unconfirmed')


def audit(rows, grouper):
    by_key, excluded = {}, []
    all_keys, all_groups = set(), set()
    seen_ids = set()
    for original in rows:
        row = dict(original)
        if row['input_id'] in seen_ids:
            raise ValueError('duplicate_input_id')
        seen_ids.add(row['input_id'])
        raw = row.get('raw_url')
        if isinstance(raw, str) and raw.strip():
            all_keys.add(raw.strip())
            # Includes conflicts/unknown labels for external-overlap exclusion.
            try:
                all_groups.add(grouper(raw.strip()))
            except (ValueError, UnicodeError):
                pass
        try:
            if row['y_original'] not in (0, 1, '0', '1'):
                raise ValueError('unknown_label')
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError('empty_url')
            row['url_key'] = raw.strip()
            row['group_id'] = grouper(row['raw_url'])
            row['y_original'] = int(row['y_original'])
            row['y_mal'] = 1 - row['y_original']
            # Reject feature extraction failure before freezing common rows.
            extract_features(row['raw_url'])
        except (ValueError, TypeError, UnicodeError, re.error) as exc:
            row['exclusion_reason'] = str(exc)
            excluded.append(row)
            continue
        by_key.setdefault(row['url_key'], []).append(row)
    clean = []
    for key in sorted(by_key):
        group = sorted(by_key[key], key=lambda r: r['input_id'])
        if len({r['y_original'] for r in group}) > 1:
            excluded.extend(dict(r, exclusion_reason='conflicting_label') for r in group)
            continue
        first = dict(group[0])
        first['source'] = sorted({r['source'] for r in group})
        first['source_rows'] = [r['input_id'] for r in group]
        first['exclusion_reason'] = ''
        clean.append(first)
        excluded.extend(dict(r, exclusion_reason='duplicate_same_label',
                             retained_input_id=first['input_id']) for r in group[1:])
    return clean, excluded, all_keys, all_groups


def freeze_folds(rows):
    import numpy as np
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    if sklearn.__version__ != '1.2.2':
        raise RuntimeError('split_requires_sklearn_1.2.2')
    rows = sorted(rows, key=lambda r: (r['url_key'], r['input_id']))
    if len({r['group_id'] for r in rows}) < 10:
        raise ValueError('fewer_than_10_groups')
    y = np.array([r['y_mal'] for r in rows])
    groups = np.array([r['group_id'] for r in rows])
    if set(y) != {0, 1}:
        raise ValueError('both_classes_required')
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    result = [dict(r) for r in rows]
    for fold, (_, idx) in enumerate(cv.split(np.zeros((len(rows), 1)), y, groups)):
        split = 'train' if fold < 6 else ('model_selection' if fold == 6 else
                  'gate_selection' if fold == 7 else 'test')
        for i in idx:
            result[int(i)].update(fold=fold, split=split)
    validate_partitions(result)
    return result


def validate_partitions(rows):
    sets, summary = {}, {}
    if len({r['input_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_partition_id')
    for split in ('train', 'model_selection', 'gate_selection', 'test'):
        subset = [r for r in rows if r['split'] == split]
        if {r['y_mal'] for r in subset} != {0, 1}:
            raise ValueError('missing_class:' + split)
        sets[split] = {r['group_id'] for r in subset}
        counts = {}
        for row in subset:
            counts[row['group_id']] = counts.get(row['group_id'], 0) + 1
        summary[split] = dict(rows=len(subset), groups=len(sets[split]),
                             malicious=sum(r['y_mal'] for r in subset),
                             normal=sum(1-r['y_mal'] for r in subset),
                             largest_group_fraction=max(counts.values()) / len(subset))
    from itertools import combinations
    for a, b in combinations(sets, 2):
        if sets[a] & sets[b]:
            raise ValueError('domain_overlap:' + a + ':' + b)
    if any(r['split'] not in sets for r in rows):
        raise ValueError('unknown_split')
    return summary


def exclude_external(clean, all_internal_keys, all_internal_groups):
    kept, excluded = [], []
    for row in clean:
        reason = ('internal_url_overlap' if row['url_key'] in all_internal_keys else
                  'internal_domain_overlap' if row['group_id'] in all_internal_groups else '')
        if reason:
            excluded.append(dict(row, exclusion_reason=reason))
        else:
            kept.append(row)
    return kept, excluded


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
