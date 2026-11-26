"""Evidence loading, number formatting, document helpers and figures for report v2.

Runs in the report venv (python-docx, matplotlib, pandas; no TensorFlow). Every number in the
report is read here from the v1 result tables (frozen run) or the v2 experiment results.
Only the v1 docx helpers (table, field, bookmark, ReportTable, FEATURE_ROWS) and the November 23
template are imported from v1; nothing is written under the v1 folder.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from v2common import V1, V2, CODE  # noqa: E402  (same folder resolution as the experiments)

R1, R2 = V1 / "results", V2 / "results"
OUT = V2 / "report"
OUT.mkdir(exist_ok=True)
TEMPLATE_DIR = CODE / "experiments/upstream_partition"
sys.path.insert(0, str(TEMPLATE_DIR))
from make_report import field, bookmark, table  # noqa: E402  (pure docx helpers)
from table_layout import ReportTable  # noqa: E402
from feature_table import FEATURE_ROWS, example  # noqa: E402

TEAM = {"random": "team-random-original", "domain": "team-domain-original"}
BASIC = {"random": "random-original", "domain": "domain-original"}
SPLITS = ("random", "domain")
SPLIT_NAME = {"random": "URL 무작위", "domain": "도메인"}
FAM = {"logistic": "LR", "boost": "HGB"}
CASCADES = ("logistic_basic", "logistic_random", "logistic_boundary", "boost_basic", "boost_random", "boost_boundary")
NAMES = {"gru": "GRU 단독", "logistic_basic": "LR 기본", "logistic_random": "LR 무작위 5%",
         "logistic_boundary": "LR 경계 5%", "boost_basic": "HGB 기본", "boost_random": "HGB 무작위 5%",
         "boost_boundary": "HGB 경계 5%", "url_length": "URL 길이 모델", "prefix_rule": "접두어 규칙",
         "logistic_alone": "LR 단독", "boost_alone": "HGB 단독"}
POLICY = {"basic": "기본", "random": "무작위", "boundary": "경계"}


def readj(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def csv(path):
    return pd.read_csv(path)


def num(v):
    return f"{int(round(float(v))):,}"


def pct(v, d=2):
    return f"{100 * float(v):.{d}f}"


def sgn(v, d=2):
    return f"{float(v):+,.{d}f}"


def rng(a, b, fmt):
    """'a~b' of two formatted values, or one value when they print the same."""
    x, y = fmt(a), fmt(b)
    return x if x == y else f"{x}~{y}"


def f2(v, d=2):
    return f"{float(v):.{d}f}"


def ci(lo, hi, d=2, scale=1.0):
    return f"[{lo * scale:.{d}f}, {hi * scale:.{d}f}]"


class Evidence:
    """All tables the report reads, with the row lookups it needs."""

    def __init__(self):
        self.main = csv(R1 / "main-28.csv")
        self.full = csv(R1 / "full-and-paired.csv")
        self.timing = csv(R1 / "timing-summary.csv")
        self.grid = csv(R1 / "threshold-decisions.csv")
        self.seeds = csv(R1 / "seeds.csv")
        self.sens = csv(R1 / "sensitivity.csv")
        self.stage1 = csv(R1 / "stage1.csv")
        self.ablation = csv(R1 / "ablation.csv")
        self.learning = csv(R1 / "learning.csv")
        self.roles = csv(R1 / "roles.csv")
        self.cohorts = csv(R1 / "cohorts.csv").set_index("kind")
        self.pairs = csv(R1 / "timing-pairs.csv")
        self.config_pairs = csv(R1 / "configuration-pairs.csv")
        self.e1 = csv(R2 / "e1-baselines.csv")
        self.e2s = csv(R2 / "e2-miss-summary.csv")
        self.e2b = csv(R2 / "e2-miss-breakdown.csv")
        self.e2c = csv(R2 / "e2-miss-cases.csv")
        self.e3o = csv(R2 / "e3-overlap-counts.csv")
        self.e3s = csv(R2 / "e3-common-subsets.csv")
        self.e3r = csv(R2 / "e3-representation.csv")
        self.e3c = csv(R2 / "e3-representation-columns.csv")
        self.e4f = csv(R2 / "e4-prefix-flips.csv")
        self.e4o = csv(R2 / "e4-prefix-overall.csv")
        self.e4s = csv(R2 / "e4-prefix-share.csv")
        self.e5 = csv(R2 / "e5-equal-fpr.csv")
        self.e0 = csv(R2 / "e0-invalid-reasons.csv")
        # Draft builds (V2_DRAFT=1) may run before experiment 6; the final build requires it.
        # V2_E6_DIR points a layout draft at dry-run timing files; such builds are saved as drafts only.
        e6dir = Path(os.environ.get("V2_E6_DIR", R2))
        self.draft = (os.environ.get("V2_DRAFT") == "1" and not (e6dir / "e6-rate-summary.csv").exists()) or e6dir != R2
        missing = not (e6dir / "e6-rate-summary.csv").exists()
        self.e6 = None if missing else csv(e6dir / "e6-rate-summary.csv")
        self.e6v1 = None if missing else csv(e6dir / "e6-v1-comparison.csv")
        self.e6meta = None if missing else readj(e6dir / "e6-timing-meta.json")
        self.e6raw = None if missing else {s: csv(e6dir / f"e6-timing-{s}.csv") for s in SPLITS}
        for name in ("e0", "e1", "e2", "e3", "e4", "e5") + (() if self.draft else ("e6",)):
            status = readj(R2 / f"{name}-checks.json")["status"]
            assert status == "PASS", (name, status)
        assert readj(R1 / "verification.json")["status"] == "PASS"

    def m(self, kind, name):
        r = self.main[(self.main.kind == kind) & (self.main.name == name)]
        assert len(r) == 1, (kind, name)
        return r.iloc[0]

    def scope(self, kind, scope, name):
        r = self.full[(self.full.kind == kind) & (self.full.scope == scope) & (self.full.name == name) & self.full.main]
        assert len(r) == 1, (kind, scope, name)
        return r.iloc[0]

    def base(self, split, arm, model):
        r = self.e1[(self.e1.split == split) & (self.e1.arm == arm) & (self.e1.model == model)]
        assert len(r) == 1
        return r.iloc[0]

    def tau(self, kind, fam):
        return readj(V1 / kind / "routing.json")["families"][fam]["tau"]

    def n_mal(self, kind):
        return int(self.cohorts.loc[kind, "malicious"])


def metric_row(r, share):
    """Rates and confusion counts of one configuration as display strings."""
    return dict(acc=pct(r["accuracy"]), prec=pct(r["precision"]), rec=pct(r["recall"]), f1=pct(r["f1"]),
                ap=f2(r["pr_auc_ap"], 4), fpr=pct(r["fpr"]), share=share,
                tp=num(r["tp"]), fp=num(r["fp"]), tn=num(r["tn"]), fn=num(r["fn"]))


# ---------------------------------------------------------------- figures (grayscale only)

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 9.5})
    return plt


def figure_thresholds(ev, path):
    """Basic exit over every validation threshold: paired saved time (top) and recall loss (bottom)."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.2), sharex="col", layout="constrained")
    for j, split in enumerate(SPLITS):
        kind = TEAM[split]
        for fam, color, marker in (("logistic", "#1a1a1a", "o"), ("boost", "#8a8a8a", "s")):
            r = ev.grid[(ev.grid.kind == kind) & (ev.grid.family == fam) & (ev.grid.policy == "basic")]
            r = r.sort_values("calls", ascending=False)
            x = 100 * (1 - r.calls.to_numpy() / 1000)
            for ax, mid, lo, hi, scale in ((axes[0, j], "saved_ms", "saved_lo", "saved_hi", 1),
                                           (axes[1, j], "recall_loss", "loss_lo", "loss_hi", 100)):
                y = r[mid].to_numpy() * scale
                ax.plot(x, y, marker=marker, color=color, markersize=4, linewidth=1.2, label=FAM[fam])
                ax.vlines(x, r[lo] * scale, r[hi] * scale, color=color, linewidth=1.4)
                missing = r[hi].isna().to_numpy()
                if missing.any():
                    ax.scatter(x[missing], y[missing], marker="x", s=60, color="black", zorder=5)
                chosen = np.isclose(r.tau.to_numpy(), ev.tau(kind, fam))
                ax.scatter(x[chosen], y[chosen], s=90, facecolors="none", edgecolors="black", linewidths=1.2, zorder=6)
                ax.grid(axis="y", color="#dddddd", lw=.6)
        axes[0, j].set_title(SPLIT_NAME[split] + " 분할 · 팀 정제")
        axes[0, j].axhline(0, linestyle="--", color="#222222", lw=1)
        axes[0, j].legend(loc="best", ncol=2)
        axes[1, j].axhline(1, linestyle="--", color="#222222", lw=1)
        axes[1, j].set_xlabel("1,000건 배치의 GRU 호출 감소율 (%)")
        axes[0, j].set_ylabel("절약 시간 (ms / 1,000건)")
        axes[1, j].set_ylabel("악성 재현율 손실 (%p)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_rates(ev, path):
    """Recheck rate against measured saved time and recovered malicious URLs (experiment 6)."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.8), sharex=True, layout="constrained")
    for j, split in enumerate(SPLITS):
        d = ev.e6[(ev.e6.split == split) & (ev.e6.family != "gru")]
        for fam, base_color in (("logistic", "#1a1a1a"), ("boost", "#8a8a8a")):
            for policy, style, marker in (("random", "--", "^"), ("boundary", "-", "o")):
                part = pd.concat([d[(d.family == fam) & (d.policy == "basic")],
                                  d[(d.family == fam) & (d.policy == policy)]]).sort_values("rate")
                x = 100 * part.rate.to_numpy()
                label = f"{FAM[fam]} {POLICY[policy]}"
                axes[0, j].errorbar(x, part.saved_ms, yerr=[part.saved_ms - part.saved_lo, part.saved_hi - part.saved_ms],
                                    color=base_color, linestyle=style, marker=marker, markersize=4, capsize=2, lw=1.1, label=label)
                axes[1, j].plot(x, part.recovered_malicious.fillna(0), color=base_color, linestyle=style,
                                marker=marker, markersize=4, lw=1.1, label=label)
        axes[0, j].axhline(0, linestyle=":", color="#222222", lw=1)
        axes[0, j].set_title(SPLIT_NAME[split] + " 분할 · 팀 정제")
        axes[0, j].set_ylabel("GRU 대비 절약 시간 (ms / 1,000건)")
        axes[1, j].set_ylabel("기본 분기 대비 되찾은 악성 (건)")
        axes[1, j].set_xlabel("재검사율 (정상 종료 후보 중 %)")
        axes[1, j].set_xticks([0, 1, 5, 10, 20])
        for ax in axes[:, j]:
            ax.grid(axis="y", color="#dddddd", lw=.6)
    axes[0, 0].legend(loc="best", ncol=2, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)
