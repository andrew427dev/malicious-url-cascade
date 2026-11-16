"""Synthetic checks of boundary decisions, equal budgets and error accounting."""

import numpy as np
from experiment import route, cascade_metrics, self_test
from audit_results import mask_for, check
from common import ROOT, write_json

self_test()
rng = np.random.default_rng(42)
n = 101
y = np.array([0, 1] * 50 + [1])
p = rng.random(n)
p[:3] = [0.1, 0.0, 1.0]
ids = np.array([str(i) for i in range(n)])
p2 = rng.random(n)
total = 0
for tau in [0, 0.1, 0.5, 1]:
    for rate in [0, 0.01, 0.05, 0.1, 0.2, 1]:
        for mode in ["random", "boundary"]:
            a, k = route(p, ids, tau, rate, mode)
            b, j = mask_for(p, ids, tau, rate, mode)
            assert np.array_equal(a, b) and k == j
            base = p >= tau
            record = cascade_metrics(y, p2, a, base)
            check(y, p2, b, record, base)
            total += 1
write_json(
    ROOT / "policy-synthetic-checks.json",
    {
        "status": "PASS",
        "cases": total,
        "includes": [
            "threshold equality",
            "zero candidates",
            "zero recheck",
            "all candidates rechecked",
            "independent masks",
            "additional misses and recovered errors",
        ],
    },
)
print("POLICY_SYNTHETIC PASS", total)
