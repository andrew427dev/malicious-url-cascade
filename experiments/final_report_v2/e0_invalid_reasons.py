"""Professor's 11/13 request: report how control characters and invalid URLs were handled.

Team cleaning drops a raw URL when prepare.normalize raises (cohorts.build, rule invalid_url).
v1 recorded only the total (642 rows). This script re-applies the same normalizer to every
raw row of the three public CSVs and counts the rows by the reason the normalizer gives.
"""

import pandas as pd

from v2common import V1, RESULTS, read_json, write_json
from inputs import read_frames
from cohorts import model_url


def reason(url):
    try:
        model_url(url, "original", True)
        return None
    except (ValueError, UnicodeError) as exc:
        text = str(exc)
        return text if text in ("empty_url", "control_or_format_character", "unsupported_scheme",
                                "invalid_host") else "invalid_port_or_other:" + type(exc).__name__ + ":" + text[:40]


def main():
    raw = pd.concat(read_frames())          # verifies the CSV digests
    unique = pd.Series(raw.url.unique())
    reasons = dict(zip(unique, unique.map(reason)))
    raw["reason"] = raw.url.map(reasons)
    bad = raw[raw.reason.notna()]
    table = (bad.assign(malicious=bad.type.eq(0), benign=bad.type.eq(1))
                .groupby("reason").agg(rows=("url", "size"), urls=("url", "nunique"),
                                       malicious=("malicious", "sum"), benign=("benign", "sum"))
                .reset_index())
    table.to_csv(RESULTS / "e0-invalid-reasons.csv", index=False)
    stage = next(s for s in read_json(V1 / "team-random-original/cleaning.json")["stages"] if s["rule"] == "invalid_url")
    ok = int(table.rows.sum()) == stage["removed"]["rows"] and int(table.malicious.sum()) == stage["removed"]["malicious"]
    write_json(RESULTS / "e0-checks.json", {"status": "PASS" if ok else "FAIL", "raw_rows": len(raw),
                                            "invalid_rows": int(table.rows.sum()), "v1_invalid_url": stage["removed"]})
    print(table.to_string())
    assert ok


if __name__ == "__main__":
    main()
