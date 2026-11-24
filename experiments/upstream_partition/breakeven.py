"""How many GRU calls must a first stage remove before the cascade is faster?

Reads the stage-level timings already recorded in each condition's timing.json.
No model is loaded and nothing is re-measured.

    python experiments/upstream_partition/breakeven.py
"""

import json
from pathlib import Path

import numpy as np

EVIDENCE = Path(__file__).parents[2] / "evidence/upstream-partition-20261124"
FIGURE = Path(__file__).parents[2] / "docs/upstream-partition-20261124/breakeven.png"
CONDITIONS = ["random-original", "random-stripped", "domain-original", "domain-stripped"]
STAGE = {"logistic": "LR", "boost": "HGB"}
BATCH = 1000
# label offsets for the three points that would otherwise collide
NUDGE = {("LR", "random-stripped"): (-46, 6), ("LR", "random-original"): (6, -11), ("HGB", "random-stripped"): (6, -12)}


def per_call_ms(calls, gru_ms, condition):
    """Slope of GRU time on GRU calls, with one intercept per condition."""
    x = np.column_stack([calls] + [(condition == c).astype(float) for c in range(len(CONDITIONS))])
    return np.linalg.lstsq(x, gru_ms, rcond=None)[0][0]


def main():
    rounds = []
    for index, name in enumerate(CONDITIONS):
        timing = json.loads((EVIDENCE / name / "timing.json").read_text(encoding="utf-8"))
        assert timing["batch"] == BATCH
        for r in timing["rounds"]:
            # the three recorded stages are the whole measured interval
            assert abs(r["features_ms"] + r["route_ms"] + r["gru_ms"] - r["total_ms"]) < 0.01
            rounds.append((index, r["name"], r["calls"], r["route_ms"], r["gru_ms"]))
    condition = np.array([r[0] for r in rounds])
    policy = np.array([r[1] for r in rounds])
    calls = np.array([r[2] for r in rounds], float)
    stage_ms = np.array([r[3] for r in rounds])
    gru_ms = np.array([r[4] for r in rounds])

    slope = per_call_ms(calls, gru_ms, condition)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):  # resample rounds within each condition
        pick = np.concatenate(
            [rng.choice(np.flatnonzero(condition == c), (condition == c).sum()) for c in range(len(CONDITIONS))]
        )
        draws.append(per_call_ms(calls[pick], gru_ms[pick], condition[pick]))
    low, high = np.percentile(draws, [2.5, 97.5])
    assert 0 < low <= slope <= high

    rows = []
    for index, name in enumerate(CONDITIONS):
        summary = json.loads((EVIDENCE / name / "timing.json").read_text(encoding="utf-8"))["summary"]
        for family, label in STAGE.items():
            mask = (condition == index) & np.char.startswith(policy, family)
            cost = float(np.median(stage_ms[mask]))
            best = summary[family + "_boundary"]
            rows.append(
                {
                    "condition": name,
                    "stage": label,
                    "stage_ms": cost,
                    "breakeven_calls_saved_pct": cost / slope / BATCH * 100,
                    "breakeven_low_pct": cost / high / BATCH * 100,
                    "breakeven_high_pct": cost / low / BATCH * 100,
                    "boundary_calls_saved_pct": best["calls_saved_pct"],
                    "boundary_time_saved_pct": best["time_saved_pct"],
                }
            )
    result = {
        "source": "stage-level rounds in <condition>/timing.json (4 conditions x 7 policies x 5 repeats)",
        "model": "gru_ms = intercept(condition) + per_call_ms * calls; breakeven = stage_ms / per_call_ms",
        "rounds": len(rounds),
        "gru_per_call_ms": slope,
        "gru_per_call_ms_ci95": [low, high],
        "rows": rows,
    }
    out = EVIDENCE / "summary/breakeven.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"GRU {slope:.4f} ms/call (95% {low:.4f}-{high:.4f}), {len(rounds)} rounds")
    for r in rows:
        print(
            "{condition:16} {stage:3} stage {stage_ms:5.2f} ms  breakeven {breakeven_calls_saved_pct:5.1f}% "
            "({breakeven_low_pct:.1f}-{breakeven_high_pct:.1f})  boundary: calls -{boundary_calls_saved_pct:.1f}% "
            "time {boundary_time_saved_pct:+.2f}%".format(**r)
        )
    figure(rows)


def figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    names = {
        "random-original": "무작위·원래",
        "random-stripped": "무작위·통일",
        "domain-original": "도메인·원래",
        "domain-stripped": "도메인·통일",
    }
    for stage, marker, face in (("LR", "o", "white"), ("HGB", "s", "#252525")):
        part = [r for r in rows if r["stage"] == stage]
        ax.scatter(
            [r["boundary_calls_saved_pct"] for r in part],
            [r["boundary_time_saved_pct"] for r in part],
            marker=marker, s=46, facecolor=face, edgecolor="#131313", zorder=3, label=f"{stage} 경계 5%",
        )
        for r in part:
            ax.annotate(
                names[r["condition"]], (r["boundary_calls_saved_pct"], r["boundary_time_saved_pct"]),
                textcoords="offset points", xytext=NUDGE.get((stage, r["condition"]), (6, 5)),
                fontsize=7, color="#131313",
            )
        edge = float(np.mean([r["breakeven_calls_saved_pct"] for r in part]))
        ax.axvline(edge, color="#434343", linewidth=0.9, linestyle="--" if stage == "HGB" else ":")
        ax.text(edge + 0.4, -9.3, f"{stage} 손익분기 {edge:.1f}%", fontsize=7, color="#131313")
    ax.axhline(0, color="#131313", linewidth=0.8)
    ax.set_xlim(0, 35)
    ax.set_ylim(-10, 12)
    ax.set_xlabel("GRU 호출 감소율 (%, 같은 1,000건)", fontsize=8)
    ax.set_ylabel("GRU 단독 대비 시간 단축률 (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE)


if __name__ == "__main__":
    main()
