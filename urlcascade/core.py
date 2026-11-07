"""Ordered, selective inference. URLs are strings; no network calls here."""
from __future__ import annotations

import math
import re
from urllib.parse import urlsplit
from .features import extract_features

GATE_CANDIDATES = (0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def hostname_for_grouping(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('empty_or_nonstring')
    text = raw.strip()
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in text):
        raise ValueError('whitespace_or_control')
    parsed = urlsplit(text if '://' in text or text.startswith('//') else '//' + text)
    if parsed.scheme and parsed.scheme.lower() not in ('http', 'https'):
        raise ValueError('unsupported_scheme')
    _ = parsed.port  # reject malformed or out-of-range ports
    host = parsed.hostname
    if not host:
        raise ValueError('missing_host')
    if '%' in host:
        raise ValueError('zone_or_escaped_host')
    host = host.rstrip('.').lower()
    import ipaddress
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        pass
    # Numeric-looking invalid IPs must not become domain groups.
    if re.fullmatch(r'[0-9.]+', host):
        raise ValueError('invalid_ip')
    host = host.encode('idna').decode('ascii')
    if len(host) > 253 or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', x)
                             for x in host.split('.')):
        raise ValueError('invalid_host')
    return host


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid_probability')
    return value


def infer(records, *, mode, t1=None, t2=None, tau=0):
    """t1(raw_features)->p_mal; t2(raw_urls, raw_features)->s_safe.

    Adapters are loaded once by the caller. Feature extraction is inside every
    timed call; T1 scaling stays inside its adapter. No URL dedup or shuffle.
    """
    if mode not in ('always_normal', 't1', 't2', 'cascade'):
        raise ValueError('unknown_mode')
    probability(tau)
    ids = [r['input_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate_input_id')
    out, valid, features = [], [], []
    for i, record in enumerate(records):
        row = dict(input_id=record['input_id'], status='ok', label_mal=None,
                   t1_score_mal=None, t2_called=False, t2_score_mal=None, route=None)
        out.append(row)
        try:
            hostname_for_grouping(record.get('raw_url'))
            values = extract_features(record['raw_url']) if mode != 'always_normal' else None
        except (ValueError, TypeError, UnicodeError, re.error) as exc:
            row.update(status='invalid', route='invalid', error=str(exc))
            continue
        if mode == 'always_normal':
            row.update(label_mal=0, route='always_normal')
        else:
            valid.append(i)
            features.append(values)
    if not valid:
        return out
    forwarded = []
    if mode in ('t1', 'cascade'):
        if t1 is None:
            raise ValueError('t1_required')
        scores = list(t1(features))
        if len(scores) != len(valid):
            raise ValueError('t1_output_count')
        for j, (i, score) in enumerate(zip(valid, scores)):
            try:
                p = probability(score)
            except (TypeError, ValueError):
                out[i].update(status='error', route='t1_error', error='invalid_probability')
                continue
            out[i]['t1_score_mal'] = p
            if mode == 't1':
                out[i].update(label_mal=int(p >= 0.5), route='t1')
            elif p < tau:
                out[i].update(label_mal=0, route='early_normal')
            else:
                forwarded.append((i, j))
    else:
        forwarded = [(i, j) for j, i in enumerate(valid)]
    if forwarded:
        if t2 is None:
            raise ValueError('t2_required')
        urls = [records[i]['raw_url'] for i, _ in forwarded]
        raw_numeric = [features[j] for _, j in forwarded]
        scores = list(t2(urls, raw_numeric))
        if len(scores) != len(forwarded):
            raise ValueError('t2_output_count')
        for (i, _), score in zip(forwarded, scores):
            out[i].update(t2_called=True, route='t2')
            try:
                s = probability(score)
                # Compare in original score space to preserve exact 0.5 boundary.
                out[i].update(label_mal=int(s < 0.5), t2_score_mal=1 - s)
            except (TypeError, ValueError):
                out[i].update(status='error', route='t2_error', error='invalid_probability')
    return out


def binary_metrics(y, prediction):
    if len(y) != len(prediction) or any(v not in (0, 1) for v in (*y, *prediction)):
        raise ValueError('binary_labels_required')
    tp = sum(a == b == 1 for a, b in zip(y, prediction))
    tn = sum(a == b == 0 for a, b in zip(y, prediction))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, prediction))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, prediction))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, fpr=fpr,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                accuracy=(tp + tn) / len(y) if y else None,
                balanced_accuracy=(recall + 1 - fpr) / 2 if recall is not None and fpr is not None else None)


def cascade_invariants(y, baseline, cascade, called):
    if not len(y) == len(baseline) == len(cascade) == len(called):
        raise ValueError('row_count_mismatch')
    b, c = binary_metrics(y, baseline), binary_metrics(y, cascade)
    if any(cp and not bp for cp, bp in zip(cascade, baseline)):
        raise AssertionError('cascade_not_subset')
    if any(cp and not route for cp, route in zip(cascade, called)):
        raise AssertionError('early_exit_cannot_be_malicious')
    k = sum(a == 1 and bp == 1 and not route for a, bp, route in zip(y, baseline, called))
    lost = sum(a == 1 and bp == 1 and cp == 0 for a, bp, cp in zip(y, baseline, cascade))
    if k != lost:
        raise AssertionError('t2_decision_mismatch_on_forwarded_rows')
    m = sum(y)
    delta = k / m if m else None
    if m and not math.isclose(b['recall'] - c['recall'], delta, abs_tol=1e-12):
        raise AssertionError('recall_identity')
    if b['fpr'] is not None and c['fpr'] > b['fpr']:
        raise AssertionError('fpr_increased')
    return dict(additional_fn=k, malicious=m, recall_drop=delta,
                allowed=(k <= m // 100) if m else False,
                early_malicious=sum(a == 1 and not route for a, route in zip(y, called)),
                call_rate=sum(called) / len(called) if called else None)


def select_gate(candidates):
    """Candidates contain allowed, tau, and five measured mean latencies."""
    import statistics
    valid = []
    for row in candidates:
        if row['tau'] not in GATE_CANDIDATES:
            raise ValueError('unexpected_gate')
        if not row['allowed']:
            continue
        times = row['mean_latencies']
        if len(times) != 5 or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('five_real_timing_repetitions_required')
        valid.append((row['tau'], statistics.median(times)))
    if not valid:
        raise ValueError('no_allowed_gate')
    fastest = min(t for _, t in valid)
    return min(tau for tau, t in valid if t <= fastest * 1.05)
