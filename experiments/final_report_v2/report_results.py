"""Chapter 5, the summary/conclusion sentences and the appendix of report v2, all computed from files."""
import re

import numpy as np
import pandas as pd

from report_base import (V1, OUT, TEAM, BASIC, SPLITS, SPLIT_NAME, FAM, NAMES, CASCADES, POLICY, readj,
                         num, pct, sgn, f2, ci, rng, metric_row, figure_thresholds, figure_rates)

SHORT = {"충족·단축": "충족·단축", "유보": "유보", "시간 또는 미탐 기준 미충족": "미충족", "기준": "기준"}

RATE_ORDER = [("basic", 0.0), ("random", 0.01), ("boundary", 0.01), ("random", 0.05), ("boundary", 0.05),
              ("random", 0.10), ("boundary", 0.10), ("random", 0.20), ("boundary", 0.20)]


def josa(word, pair):
    """Korean particle after a word ending in a Latin letter, digit or Hangul syllable."""
    last = word.strip()[-1]
    if "가" <= last <= "힣":
        final = (ord(last) - 0xAC00) % 28 != 0
    elif last.isdigit():
        final = last in "013678"
    else:
        final = last.upper() in "LMNR"
    return word + (pair[0] if final else pair[1])


# ---------------------------------------------------------------- chapter 2 and 3 helpers

def author_published():
    """Accuracies the author's notebook printed: last internal val_accuracy and the separate-evaluation accuracy."""
    nb = readj(V1 / "source/Model_Training.ipynb")["cells"]
    log = lambda i: "\n".join("".join(o.get("text", [])) for o in nb[i]["outputs"])
    return float(re.findall(r"val_accuracy: ([0-9.]+)", log(9))[-1]), float(re.search(r"Accuracy: ([0-9.]+)", log(10)).group(1))


def author_counts():
    """Published class counts of the author's separate evaluation, read from the confusion matrix the notebook printed."""
    import re as _re
    nb = readj(V1 / "source/Model_Training.ipynb")["cells"]
    text = "".join("".join(o.get("text", [])) for o in nb[10]["outputs"])
    rows = [[int(v) for v in r.split()] for r in _re.findall(r"\[\s*(\d+\s+\d+)\s*\]", text)][:2]
    ext = readj(V1 / "author-validation/results.json")["external"]
    assert sum(rows[0]) == ext["tp"] + ext["fn"]      # first row is the malicious class, as in the reproduction
    return sum(rows[0]), sum(rows[1])


def reproduction():
    auth = readj(V1 / "author-validation/results.json")
    internal, ext = auth["validation"], auth["external"]
    nb = readj(V1 / "source/Model_Training.ipynb")["cells"]

    def logged(i):
        return "\n".join("".join(o.get("text", [])) for o in nb[i]["outputs"])
    orig_val = float(re.findall(r"val_accuracy: ([0-9.]+)", logged(9))[-1])
    comparison = [("내부 검증 · 정확도", orig_val, internal["accuracy"])]
    for key, printed, name in (("accuracy", "Accuracy", "정확도"), ("weighted_precision", "Precision", "가중 정밀도"),
                               ("weighted_recall", "Recall", "가중 재현율"), ("weighted_f1", "F1 Score", "가중 F1")):
        value = float(re.search(printed + r": ([0-9.]+)", logged(10)).group(1))
        comparison.append(("별도 평가 · " + name, value, ext[key]))
    final_fit = readj(V1 / "author-final/trained.json")["history"]["accuracy"][-1]
    rows = [(n, pct(a), pct(b), sgn(100 * (b - a))) for n, a, b in comparison]
    rows.append(("전체 데이터 · 20회차 학습 정확도", "기록 없음", pct(final_fit), "—"))
    return rows, auth


def cleaning_rows():
    c = readj(V1 / "team-random-original/cleaning.json")
    assert c == {**c, "stages": readj(V1 / "team-domain-original/cleaning.json")["stages"]}
    st = {s["rule"]: s for s in c["stages"]}
    raw = readj(V1 / "source-manifest.json")["raw"]
    rows = [["공개 원본 (CSV 3개)", "—", "—", num(raw["rows"])]]
    labels = [("invalid_url", "① 잘못된 URL 제외 (제어·서식 문자, 지원하지 않는 스킴, 잘못된 호스트·포트)"),
              ("raw_label_conflict", "② 원본 URL 라벨 충돌 제외"),
              ("identical_raw_label_repeat", "③ 같은 원본 URL·라벨 반복 제거"),
              ("normalized_label_conflict", "④ 정규화 URL 라벨 충돌 제외"),
              ("normalized_repeat", "⑤ 정규화 URL 반복 제거"),
              ("unusable_input", "⑥ 모델 입력을 만들 수 없는 행 제외"),
              ("character_numeric_label_conflict", "⑦ 앞 52자+수치 특징 라벨 충돌 제외"),
              ("character_numeric_repeat", "⑧ 앞 52자+수치 특징 반복 제거")]
    for key, title in labels:
        s = st[key]
        rows.append([title, num(s["removed"]["rows"]), f"{num(s['removed']['malicious'])} / {num(s['removed']['benign'])}",
                     num(s["remaining"])])
    tokens = [s for s in c["stages"] if s["rule"].startswith("token_round")]
    before = st["character_numeric_repeat"]["remaining"]
    conflict = sum(s["conflict"] for s in tokens)
    repeat = sum(s["repeat"] for s in tokens)
    rows.append(["⑨ 토큰 50칸+수치 특징 라벨 충돌 제외 (반복 검사 누계)", num(conflict), "—", num(before - conflict)])
    rows.append(["⑩ 토큰 50칸+수치 특징 반복 제거 (반복 검사 누계)", num(repeat), "—", num(before - conflict - repeat)])
    assert before - conflict - repeat == tokens[-1]["remaining"]
    s = st["unstable_normalization"]
    rows.append(["⑪ 정규화를 두 번 하면 달라지는 URL 제외", num(s["removed"]["rows"]),
                 f"{num(s['removed']['malicious'])} / {num(s['removed']['benign'])}", num(s["remaining"])])
    fin = c["final"]
    mal = sum(v["malicious"] for v in fin.values())
    ben = sum(v["benign"] for v in fin.values())
    assert mal + ben == s["remaining"]
    rows.append(["최종 (학습·검증·시험 합계)", "—", f"악성 {num(mal)} / 정상 {num(ben)}", num(mal + ben)])
    return rows


def control_text(ev):
    r = ev.e0.set_index("reason")
    ctrl = r.loc["control_or_format_character"]
    other = r[r.index.str.startswith("invalid_port_or_other")]
    return (f"제어문자는 지우거나 바꾸지 않고 행 전체를 뺐다. 유니코드 범주가 제어(Cc) 또는 서식(Cf)인 문자가 URL 어디에든 있으면 ①에서 제외했고, "
            f"이런 행은 {num(ctrl.rows)}행(서로 다른 URL {num(ctrl.urls)}개, 악성 {num(ctrl.malicious)}·정상 {num(ctrl.benign)})이었다. "
            f"양 끝 공백만 잘라낸 뒤 URL을 해석했다. ①의 나머지는 http·https가 아닌 스킴을 명시한 URL {num(r.loc['unsupported_scheme'].rows)}행, "
            f"호스트가 없거나 호스트에 공백이 있는 URL {num(r.loc['invalid_host'].rows)}행, 포트가 숫자가 아니거나 IPv6 괄호가 잘못된 URL {num(other.rows.sum())}행이다. "
            f"정규화는 스킴과 호스트를 소문자로 바꾸고 경로 끝의 슬래시를 지운다. 경로의 대소문자, 쿼리, 프래그먼트, 명시한 포트와 사용자 정보는 그대로 둔다. "
            f"⑦~⑩은 학습·검증·시험을 나누지 않고 세 집합 전체에서 검사했고, 토큰 비교는 두 분할의 학습 어휘 가운데 하나에서라도 같으면 같은 그룹으로 묶어 중복이 없어질 때까지 반복했다.")


def basic_text(ev):
    o = ev.e3o.set_index("kind")
    return ("기본 정제는 ②·③과 같은 원본 URL 규칙만 쓴다. 그 뒤 학습·검증·시험 각 집합 안에서 모델 문자열(문자 입력을 만들기 전의 전처리 문자열)이 같은 행은 라벨과 관계없이 첫 행만 남긴다. "
            "11월 13일 교수님께 보고한 754,678개 정리와 같은 수준이다. 잘못된 URL 제외와 정규화를 하지 않고, 집합 사이의 같은 모델 입력도 확인하지 않는다(5.8절). "
            f"반대로 팀 정제는 ‘집합 안 같은 모델 문자열’ 규칙 대신 수치 특징까지 포함한 키로 검사하므로, 모델 문자열이 같아도 수치 특징이 다르면 남는다. 이런 시험 행은 팀 정제 URL 무작위 "
            f"{num(o.loc[TEAM['random'], 'test_repeated_model_text'])}행, 도메인 {num(o.loc[TEAM['domain'], 'test_repeated_model_text'])}행이고 기본 정제에는 "
            f"{num(o.loc[BASIC['random'], 'test_repeated_model_text'] + o.loc[BASIC['domain'], 'test_repeated_model_text'])}행이다. "
            + basic_extra_text())


def basic_extra_text():
    parts = []
    for split in SPLITS:
        c = readj(V1 / BASIC[split] / "cleaning.json")
        left = c["stages"][-1]["remaining"]
        removed = sum(r["mixed_text_removed"] + r["same_label_text_removed"] for r in c["first_text"])
        kept = sum(r["remaining"] for r in c["first_text"])
        assert kept + removed == left
        parts.append((split, removed, kept, left))
    return ("기본 정제의 ‘집합 안 같은 모델 문자열’ 규칙은 원본 URL 라벨 충돌과 반복만 지운 뒤 남은 " + num(parts[0][3]) + "행(잘못된 URL을 먼저 빼는 팀 정제 ③과 다름)에서 "
            + ", ".join(f"{SPLIT_NAME[s]} 분할 {num(r)}행" for s, r, k, l in parts) + "을 더 빼 각각 "
            + ", ".join(f"{num(k)}행" for s, r, k, l in parts) + "을 남겼다.")


def representation_text(ev):
    r = ev.e3r[ev.e3r.split == "random"].set_index("role")
    d = ev.e3r[ev.e3r.split == "domain"].set_index("role")
    c = ev.e3c[(ev.e3c.split == "random") & (ev.e3c.role == "test")].set_index("feature").rows_differ
    small = ", ".join(f"{k} {num(v)}" for k, v in c.items() if k not in ("count_dir", "url_length"))
    total_tok = int(ev.e3r.token_rows_differ.sum())
    return (f"팀 정제 데이터에서 같은 행의 원본 URL과 정규화 URL을 비교하면, 두 분할의 모든 역할에서 문자 토큰이 다른 행은 {num(total_tok)}행이었다. "
            f"토큰을 만들 때 대소문자와 기호를 지우므로 정규화가 토큰을 바꾸지 않는다. 수치 특징은 URL 무작위 분할의 학습 {num(r.loc['train'].numeric_rows_differ)}행, "
            f"검증 {num(r.loc['validation'].numeric_rows_differ)}행, 시험 {num(r.loc['test'].numeric_rows_differ)}행, 도메인 분할의 학습 {num(d.loc['train'].numeric_rows_differ)}행, "
            f"검증 {num(d.loc['validation'].numeric_rows_differ)}행, 시험 {num(d.loc['test'].numeric_rows_differ)}행에서 달랐다. "
            f"URL 무작위 시험에서 바뀐 열은 대부분 경로 끝 슬래시 제거의 영향을 받는 count_dir({num(c['count_dir'])}행)와 url_length({num(c['url_length'])}행)였고, "
            f"나머지 열은 {small}행이었다. 따라서 두 정제의 입력 표기 차이는 사실상 수치 특징 일부에 한정된다.")


def role_rows(ev):
    rows = []
    for split in SPLITS:
        r = ev.roles[ev.roles.kind == TEAM[split]].set_index("role")
        total = r.n.sum()
        for role, name in (("train", "학습"), ("validation", "검증"), ("test", "시험")):
            x = r.loc[role]
            rows.append([f"{SPLIT_NAME[split]} · {name} ({pct(x.n / total, 1)}%)", num(x.n), num(x.malicious), num(x.benign), num(x.domains)])
    return rows


def split_overlap_text(ev):
    a, b = (ev.cohorts.loc[TEAM[s]] for s in SPLITS)
    return (f"URL 무작위 분할에서는 학습과 시험에 같은 등록 도메인이 {num(a.common_train_test_domains)}개 함께 있었다. 도메인 분할에서는 {num(b.common_train_test_domains)}개로, "
            f"같은 도메인을 외워 얻는 이점을 줄인 강건성 확인용 분할이다. 두 분할 모두 시험 URL이 같은 정제의 학습·검증에 그대로 들어간 경우는 없었고, "
            f"두 정제의 공통 시험 URL이 상대 정제의 학습·검증에 들어간 경우도 {num(int(ev.cohorts.loc[[TEAM[s] for s in SPLITS] + [BASIC[s] for s in SPLITS]].cross_leaks.sum()))}건이었다.")


def stage1_rows(ev):
    rows = []
    for split in SPLITS:
        sel = readj(V1 / TEAM[split] / "stage1/t1-selected.json")["selection"]
        for fam, label, key in (("logistic", "LR", "C"), ("boost", "HGB", "최대 잎")):
            for e in sel[fam]["entries"]:
                setting = e["setting"]
                rows.append([f"{SPLIT_NAME[split]} · {label}", f"{key} = {setting:g}", f2(e["validation_ap"], 4),
                             "선택" if setting == sel[fam]["winner"] else ""])
    return rows


# ---------------------------------------------------------------- shared lists (report table 11 and the Notion page)

EXPERIMENTS = [
    ["주 실험", "팀 정제 × 두 분할, GRU 단독과 캐스케이드 6개의 탐지 지표와 1%p·시간 판정", "5.1, 5.2"],
    ["실험 1", "단순 기준 모델 2개와 1단계 단독 2개의 탐지 지표(5.1절 결과표에 함께 실음), GRU 단독 지표의 재계산 대조", "5.1, 5.7"],
    ["실험 2", "추가 미탐의 유형과 대표 사례, 조기 종료로 사라진 GRU 오탐", "5.5"],
    ["실험 3", "기본 정제 시험에 남은 ‘학습·검증과 같은 입력’의 영향", "5.8"],
    ["실험 4", "고정 모델에서 접두어만 지웠을 때 판정 변화의 정답 대조", "5.7"],
    ["실험 5", "같은 오탐률·같은 재현율에서 GRU 기준 이동과 비교", "5.6"],
    ["실험 6", "재검사율 1·5·10·20%의 실측 처리 시간", "5.3, 부록 E"],
    ["보조 E", "무작위 재검사 seed 30회", "5.3"],
    ["보조 B", "접두어를 지우고 다시 학습한 모델(결과 요약)", "5.7"],
    ["보조 C, D", "1단계에서 특징 9개 제외, 학습량 곡선", "부록 A, B"],
]
EXPERIMENT_ORIGIN = ("주 실험과 보조 B~E는 모델을 학습할 때 함께 실행한 결과이고, 실험 1~6은 그 뒤 고정 모델과 저장된 점수로 새로 분석한 결과다. "
                     "5.1절의 11개 구성 표는 주 실험의 7개 운용 방식과 실험 1에서 고정 모델·규칙으로 다시 계산한 4개 비교 구성을 합친 것이다. "
                     "실험 1~6은 시험 결과를 본 뒤 정한 탐색 분석이다. 모델 가중치와 주 실험의 1단계 분기 임계값은 고정했고, 실험 5에서는 비교용 GRU 판정 기준을 검증 집합에서 새로 골랐으며, 실험 6에서는 재검사율을 달리했다.")


def rate_label(fam, policy, rate):
    return f"{FAM[fam]} {'기본 0%' if policy == 'basic' else POLICY[policy] + ' ' + f'{rate:.0%}'}"


def policy_time_pairs():
    """Measurement A, team cleaning: paired (same repeat) total time of each 5% recheck minus the basic branch, bootstrap seed 509."""
    out = []
    for split in SPLITS:
        raw = pd.read_csv(V1 / f"timing-{split}.csv")
        raw = raw[raw.main]
        for fam in ("logistic", "boost"):
            b = raw[(raw.kind == TEAM[split]) & (raw.family == fam)].pivot(index="repeat", columns="policy", values="total_ms")
            assert len(b) == 30
            for pol in ("random", "boundary"):
                d = (b[pol] - b.basic).to_numpy()
                draws = np.random.default_rng(509).integers(0, len(d), (10000, len(d)))
                lo, hi = np.quantile(np.median(d[draws], axis=1), [.025, .975])
                out.append(dict(split=split, family=fam, policy=pol, diff=float(np.median(d)), lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out)


def prefix_change_text():
    pf = pd.read_csv(V1 / "results/prefix-fixed.csv")
    pf = pf[pf.kind.isin([TEAM[s] for s in SPLITS]) & (pf.group == "with_prefix") & (pf.name == "gru")].set_index("kind")
    r = [pf.loc[TEAM[s]] for s in SPLITS]
    assert all(x.numeric_changes == x.rows for x in r)
    return ("같은 행·같은 순서에서 접두어가 있는 시험 행 가운데 문자 토큰이 바뀐 행은 " + "·".join(f"{num(x.token_changes)}행" for x in r)
            + "(문자 입력을 만들 때 http(s)://를 이미 지우므로), 수치 특징이 바뀐 행은 " + "·".join(f"{num(x.numeric_changes)}행" for x in r)
            + "(전부), GRU 판정이 바뀐 행은 " + "·".join(f"{num(x.changed)}행" for x in r) + "이었다(URL 무작위·도메인 순).")


def prefix_loss(ev):
    f = ev.e4f[(ev.e4f.name == "gru") & (ev.e4f.group == "with_prefix")].set_index("split")
    return {s: (int(f.loc[s].lost_detection), int(f.loc[s].detected_before), int(f.loc[s].changed)) for s in SPLITS}


def settings_text(ev):
    best = {s: decisions(ev, s).sort_values("total_ms").iloc[0] for s in SPLITS}
    e6 = ev.e6[ev.e6.family != "gru"]
    met = {(s, f): ((e6.split == s) & (e6.family == f) & (e6.decision == "충족·단축")).sum() for s in SPLITS for f in ("logistic", "boost")}
    n = {(s, f): ((e6.split == s) & (e6.family == f)).sum() for s in SPLITS for f in ("logistic", "boost")}
    assert all(met[s, "logistic"] == n[s, "logistic"] for s in SPLITS) and met["domain", "boost"] < n["domain", "boost"]
    for s in SPLITS:
        for f in ("logistic", "boost"):
            g = ev.grid[(ev.grid.kind == TEAM[s]) & (ev.grid.family == f) & (ev.grid.policy == "basic") & (ev.grid.tau > ev.tau(TEAM[s], f))]
            assert len(g) and (g.loss_hi >= .01).all()
    return (f"설정 선택. 측정 A의 주 구성에서 전체 시간 중앙값이 가장 짧은 구성은 URL 무작위 분할의 {NAMES[best['random']['name']]}, 도메인 분할의 {NAMES[best['domain']['name']]}이었다. "
            f"재검사율 1·5·10·20%까지 넓히면(측정 B) LR 1단계는 두 분할 모두 {n['random', 'logistic']}개 구성 전부 ‘충족·단축’이었지만, HGB 1단계는 URL 무작위 {met['random', 'boost']}개, "
            f"도메인 {met['domain', 'boost']}개만 ‘충족·단축’이었다. 처음 보는 도메인이 많은 환경에서 재검사를 함께 쓴다면 LR 1단계의 시간 이점이 더 안정적이었다. "
            "놓친 악성을 줄이는 데는 5% 재검사 예산에서 경계 재검사가 두 분할과 두 1단계 모델의 네 조합 모두 무작위 재검사 30회 평균보다 악성을 더 되찾았다. 분기 임계값은 시험한 12개 후보 가운데 검증 선택값보다 높은 후보에서 네 조건 모두 재현율 손실의 95% 구간 상한이 1%p를 넘었으므로, 검증에서 고른 값보다 크게 올리지 않는 편이 안전하다. "
            + speed_note(ev))


def speed_note(ev):
    mA = ev.main[ev.main.kind.isin(TEAM.values()) & (ev.main.family != "gru")]
    slow = ev.e6[(ev.e6.family != "gru") & (ev.e6.saved_hi < 0)]
    return (f"시간 이득 자체도 작다. 측정 A의 주 캐스케이드는 GRU 단독보다 {f2(mA.saved_pct.min(), 1)}~{f2(mA.saved_pct.max(), 1)}% 빨랐고, "
            + ("재검사율을 넓힌 측정 B에서는 " + ", ".join(f"{e6_label(r)}가 1,000건당 {f2(-r.saved_ms)} ms 더 걸렸다" for r in slow.itertuples()) + ". " if len(slow) else "")
            + "이 정도 시간 이득이 추가 미탐을 감수할 만한지는 운영 목적에 따라 정해야 한다.")


# ---------------------------------------------------------------- summary sentences

def decisions(ev, split):
    m = ev.main[(ev.main.kind == TEAM[split]) & (ev.main.family != "gru")]
    return m


def e6_label(r):
    return f"{SPLIT_NAME[r.split]} {FAM[r.family]} {POLICY[r.policy]} {r.rate:.0%}"


def session_changes(ev):
    """0%/5% cascades timed in both measurements (A: frozen-run timing, B: experiment 6) whose decision differs: (e6 row, decision A)."""
    out = []
    for c in ev.e6v1.itertuples():
        if c.name == "gru":
            continue
        fam, policy = c.name.split("_")
        r = ev.e6[(ev.e6.split == c.split) & (ev.e6.family == fam) & (ev.e6.policy == policy) & np.isclose(ev.e6.rate, c.rate)]
        assert len(r) == 1
        a = ev.m(TEAM[c.split], c.name).decision
        if a != r.iloc[0].decision:
            out.append((r.iloc[0], a))
    return out


def changes_sentence(changes):
    return ", ".join(f"{e6_label(r)}는 측정 A에서 ‘{SHORT[a]}’, 측정 B에서 ‘{SHORT[r.decision]}’였다" for r, a in changes)


def summary(ev):
    out = []
    for split in SPLITS:
        m = decisions(ev, split)
        ok = (m.decision == "충족·단축").sum()
        g = ev.m(TEAM[split], "gru")
        out.append(f"RQ1 · {SPLIT_NAME[split]} 분할. {'캐스케이드 6개 모두' if ok == 6 else f'캐스케이드 6개 가운데 {ok}개'}가 재현율 손실 1%p 미만과 시간 단축을 95% 구간으로 함께 충족했다. "
                   f"추가 미탐은 실제 악성 {num(g.tp + g.fn)}건 중 {num(m.additional_misses.min())}~{num(m.additional_misses.max())}건(손실 {pct(m.recall_loss.min())}~{pct(m.recall_loss.max())}%p)이었고, "
                   f"1,000건당 절약 시간은 {f2(m.saved_ms.min())}~{f2(m.saved_ms.max())} ms(GRU 단독 {f2(g.total_ms)} ms의 {f2(m.saved_pct.min(), 1)}~{f2(m.saved_pct.max(), 1)}%)였다.")
        if ev.e6 is not None:
            ch = [c for c in session_changes(ev) if c[0].split == split]
            if ch:
                out[-1] += " 이 시간 판정은 측정 A의 값이며, 같은 구성을 다시 잰 측정 B와 비교하면 " + changes_sentence(ch) + "(5.3절)."
    r = {s: (ev.m(TEAM[s], "logistic_basic"), ev.m(TEAM[s], "boost_basic")) for s in SPLITS}
    t = ev.main[ev.main.kind.isin(TEAM.values()) & (ev.main.family != "gru")].groupby("family").score_ms.median()
    saved = {s: (ev.main[(ev.main.kind == TEAM[s]) & (ev.main.family == "logistic")].saved_ms.mean(),
                 ev.main[(ev.main.kind == TEAM[s]) & (ev.main.family == "boost")].saved_ms.mean()) for s in SPLITS}
    assert saved["random"][1] > saved["random"][0] and saved["domain"][0] > saved["domain"][1]
    out.append(f"RQ2 · LR과 HGB. 기본 분기의 GRU 전달 비율은 URL 무작위에서 LR {pct(r['random'][0].t2_call_rate, 1)}%, HGB {pct(r['random'][1].t2_call_rate, 1)}%, "
               f"도메인에서 LR {pct(r['domain'][0].t2_call_rate, 1)}%, HGB {pct(r['domain'][1].t2_call_rate, 1)}%였다. HGB가 GRU를 덜 불렀지만 1단계 계산이 1,000건당 약 {f2(t['boost'], 1)} ms로 LR(약 {f2(t['logistic'], 2)} ms)보다 길어, "
               f"절약 시간이 더 큰 쪽은 URL 무작위에서 HGB, 도메인에서 LR이었다.")
    seeds = ev.seeds[ev.seeds.kind.str.startswith("team")]
    parts, top = [], 0.0
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            s = seeds[(seeds.kind == TEAM[split]) & (seeds.family == fam)].recovered_malicious
            b = ev.m(TEAM[split], fam + "_boundary").recovered_malicious
            assert b > s.mean()
            top = max(top, b / ev.n_mal(TEAM[split]))
            parts.append(f"{SPLIT_NAME[split]} {FAM[fam]} {num(b)}건 대 {f2(s.mean(), 1)}건")
    bnd = [ev.m(TEAM[s], f + "_boundary") for s in SPLITS for f in ("logistic", "boost")]
    pp = policy_time_pairs()
    out.append("RQ3 · 같은 5% 예산에서 경계 재검사가 되찾은 악성은 무작위 재검사 seed 30회(보조 E) 평균보다 네 경우 모두 많았다(" + ", ".join(parts) + "). "
               f"다만 되찾은 수는 실제 악성의 {pct(top)}% 이하였다. 5% 재검사는 전체 시험에서 다시 검사한 {rng(min(r.rechecks for r in bnd), max(r.rechecks for r in bnd), num)}건만큼 GRU 호출을 늘렸고, "
               f"경계 재검사의 신규 오탐은 {rng(min(r.new_false_positives for r in bnd), max(r.new_false_positives for r in bnd), num)}건이었다. "
               f"측정 A에서 5% 재검사는 같은 1단계의 기본 분기보다 1,000건당 {f2(pp['diff'].min())}~{f2(pp['diff'].max())} ms 더 걸렸고, 8개 비교 가운데 {int((pp.lo > 0).sum())}개에서 차이의 95% 구간이 0보다 컸다.")
    e6 = ev.e6
    if e6 is None:
        out.append("[초안: 실험 6 결과 대기]")
    else:
      rnd = e6[(e6.split == "random") & (e6.family == "boost") & (e6.policy == "boundary")].set_index("rate")
      # Claims below must hold in the measured data; the build stops otherwise.
      combos = list(e6[(e6.family != "gru") & (e6.policy != "basic")].groupby(["split", "family", "policy"]))
      ups = 0
      for (s, f, p), g in combos:
          g = g.sort_values("rate")
          base0 = e6[(e6.split == s) & (e6.family == f) & (e6.policy == "basic")].iloc[0].saved_ms
          assert g.recovered_malicious.is_monotonic_increasing and g.iloc[-1].saved_ms < base0
          ups += int((g.saved_ms.diff() > 0).any())
      assert len(combos) == 8 and ups > 0
      out.append(f"RQ4 · 재검사율을 1%에서 20%로 높이면 되찾는 악성은 여덟 조합(두 분할 × 1단계 2개 × 선정 방식 2개) 모두 늘었고, 20%의 절약 시간 중앙값은 여덟 조합 모두 기본 분기보다 작았다. "
               f"다만 중간 비율에서는 {ups}개 조합의 절약 시간이 한 번 이상 다시 늘어, 비율에 따라 일정하게 줄지는 않았다. 예를 들어 URL 무작위 HGB 경계 재검사는 1%에서 20%로 올리면 되찾은 악성이 "
               f"{num(rnd.loc[0.01].recovered_malicious)}건에서 {num(rnd.loc[0.20].recovered_malicious)}건으로, 신규 오탐이 {num(rnd.loc[0.01].new_false_positives)}건에서 {num(rnd.loc[0.20].new_false_positives)}건으로 늘었고, "
               f"1,000건당 절약 시간은 1·5·10·20%에서 {'·'.join(f2(rnd.loc[x].saved_ms) for x in (0.01, 0.05, 0.10, 0.20))} ms였다(5.3절). 임계값은 검증에서 고른 값 주변의 좁은 범위에서만 두 조건을 함께 충족했다(5.4절). "
               f"도메인 분할에서는 GRU 단독의 오탐률이 {pct(ev.m(TEAM['domain'], 'gru').fpr)}%로 URL 무작위({pct(ev.m(TEAM['random'], 'gru').fpr)}%)보다 높았고, HGB의 GRU 전달 비율이 높아져 HGB 캐스케이드의 절약 시간이 LR보다 작았다.")
      dh = e6[(e6.split == "domain") & (e6.family == "boost")]
      dh_base = dh[dh.policy == "basic"].iloc[0]
      dh_held = dh[(dh.policy != "basic") & (dh.decision != "충족·단축")]
      if len(dh_held):
          out[-1] += (f" 실험 6에서 도메인 HGB는 기본 분기의 절약 시간이 1,000건당 {f2(dh_base.saved_ms)} ms뿐이어서, 재검사 구성 8개 가운데 {len(dh_held)}개("
                      + ", ".join(f"{POLICY[r.policy]} {r.rate:.0%}" for r in dh_held.sort_values(["policy", "rate"]).itertuples())
                      + ")가 시간 기준에서 ‘충족·단축’이 아니었다.")
          assert (dh_held.loss_hi < .01).all()
    e5 = ev.e5[ev.e5.name.isin(CASCADES)]
    rnd5 = e5[e5.split == "random"]
    assert (rnd5.recall_diff_lo_pp > 0).all()
    dom5 = e5[e5.split == "domain"]
    hgb5, lr5 = dom5[dom5.name.str.startswith("boost")], dom5[dom5.name.str.startswith("logistic")]
    assert len(hgb5) == 3 and (hgb5.fpr_diff_lo_pp > 0).all() and (lr5.fpr_diff_lo_pp <= 0).all()
    rdiff = 100 * (hgb5.test_gru_recall_equal_recall - hgb5.test_cascade_recall)
    assert (rdiff > 0).all()
    out.append(f"같은 오탐률 비교. URL 무작위 분할에서 GRU 판정 기준을 각 캐스케이드의 검증 오탐률에 맞춰 올리면, 시험 재현율이 캐스케이드보다 "
               f"{f2(rnd5.recall_diff_gru_minus_cascade_pp.min())}~{f2(rnd5.recall_diff_gru_minus_cascade_pp.max())}%p 높았고 여섯 구간 모두 0보다 컸다. "
               f"즉 URL 무작위 분할에서 캐스케이드의 낮은 오탐률은 GRU 기준을 올려도 얻을 수 있는 수준이었고, 이 분할에서 캐스케이드의 이점은 시간 절약이었다. "
               f"도메인 분할에서는 달랐다. 검증 재현율을 맞춘 GRU 단독의 시험 오탐률이 HGB 캐스케이드보다 {f2(hgb5.fpr_diff_gru_minus_cascade_pp.min())}~{f2(hgb5.fpr_diff_gru_minus_cascade_pp.max())}%p 높았고 세 구간 모두 0보다 컸다(LR 세 구성은 구간이 0을 포함). "
               f"이때 시험 재현율은 HGB 캐스케이드가 GRU보다 {f2(rdiff.min())}~{f2(rdiff.max())}%p 낮았다. 처음 보는 도메인에서는 HGB 1단계가 재현율을 조금 내주고 GRU 오탐 일부를 막았다(5.6절).")
    pr = ev.base("random", "team", "prefix_rule")
    ln = ev.base("random", "team", "url_length")
    out.append(f"지름길 점검. URL 무작위 분할에서 URL 길이 모델은 정확도가 {pct(ln.accuracy, 1)}%였고 ROC-AUC가 {f2(ln.roc_auc, 3)}로 0.5보다 낮아 악성을 잘 가려내지 못했지만, ‘http(s)://로 시작하면 악성’이라는 접두어 규칙은 "
               f"정확도 {pct(pr.accuracy, 1)}%였다. 고정 모델에서 접두어만 지우면, 접두어가 있고 GRU가 원래 잡던 악성 가운데 "
               + ", ".join(f"{SPLIT_NAME[s]} {num(a)}/{num(b)}건({pct(a / b)}%)" for s, (a, b, c) in prefix_loss(ev).items())
               + "을 놓쳤다. 판정이 바뀐 접두어 URL의 " + "·".join(f"{pct(a / c, 1)}%" for a, b, c in prefix_loss(ev).values())
               + "가 이런 탐지 손실이어서, 모델이 접두어 표기에 크게 기대고 있음을 확인했다(5.7절).")
    return out


def conclusion(ev):
    out = []
    for split in SPLITS:
        m = decisions(ev, split)
        best = m.sort_values("total_ms").iloc[0]
        out.append(f"{SPLIT_NAME[split]} 분할의 팀 정제 데이터에서 캐스케이드 6개는 모두 GRU 단독 대비 재현율 손실 1%p 미만과 시간 단축을 95% 구간으로 함께 충족했다"
                   if (m.decision == "충족·단축").all() else
                   f"{SPLIT_NAME[split]} 분할의 팀 정제 데이터에서 캐스케이드 6개 가운데 {(m.decision == '충족·단축').sum()}개가 기준을 충족했다")
        out[-1] += (f". 1,000건 처리 시간 중앙값이 가장 짧은 구성({NAMES[best['name']]})은 {f2(best.total_ms)} ms였고, 같은 반복끼리 비교한 GRU 단독 대비 절약 시간 중앙값은 {f2(best.saved_ms)} ms "
                    f"({f2(best.saved_pct, 1)}%)였으며, 실제 악성 {num(best.tp + best.fn)}건 가운데 {num(best.additional_misses)}건을 더 놓쳤다.")
        ch = [] if ev.e6 is None else [c for c in session_changes(ev) if c[0].split == split]
        if ch:
            out[-1] += " 다만 이 시간 판정은 측정 A의 값이며, " + changes_sentence(ch) + "."
    e5 = ev.e5[ev.e5.name.isin(CASCADES)]
    hgb5 = e5[(e5.split == "domain") & e5.name.str.startswith("boost")]
    m = ev.main[ev.main.kind.isin(TEAM.values()) & (ev.main.family != "gru")]
    g = ev.main[ev.main.kind.isin(TEAM.values()) & (ev.main.family == "gru")]
    share = (g.input_ms / g.total_ms)
    out.append(f"절약 폭은 1,000건당 {f2(m.saved_ms.min())}~{f2(m.saved_ms.max())} ms(GRU 단독 시간의 {f2(m.saved_pct.min(), 1)}~{f2(m.saved_pct.max(), 1)}%)로 작다. GRU 단독에서도 입력 준비(정규화·수치 특징·문자 토큰)가 전체 시간의 {rng(share.min(), share.max(), lambda v: pct(v, 0))}%를 차지하고, 이 가운데 정규화와 수치 특징은 모든 URL에 필요하다. 그래서 GRU 호출을 줄여도 줄어드는 몫이 제한된다. "
               "또 URL 무작위 분할에서는 검증 오탐률을 맞춘 GRU 단독이 시험 재현율도 더 높아, 이 분할에서 캐스케이드의 이점은 탐지 개선이 아니라 작은 시간 절약이었고 그 대가로 추가 미탐이 생겼다. "
               f"도메인 분할의 HGB 캐스케이드는 검증 재현율을 맞춘 GRU 단독보다 시험 오탐률이 {f2(hgb5.fpr_diff_gru_minus_cascade_pp.min())}~{f2(hgb5.fpr_diff_gru_minus_cascade_pp.max())}%p 낮아(시험 재현율은 "
               f"{f2((100 * (hgb5.test_gru_recall_equal_recall - hgb5.test_cascade_recall)).min())}~{f2((100 * (hgb5.test_gru_recall_equal_recall - hgb5.test_cascade_recall)).max())}%p 낮음), 처음 보는 도메인에서 오탐을 줄이는 효과도 있었다.")
    seeds = ev.seeds[ev.seeds.kind.str.startswith("team")]
    beaten = []
    for s in SPLITS:
        for f in ("logistic", "boost"):
            k = int((seeds[(seeds.kind == TEAM[s]) & (seeds.family == f)].recovered_malicious > ev.m(TEAM[s], f + "_boundary").recovered_malicious).sum())
            if k:
                beaten.append(f"{SPLIT_NAME[s]} {FAM[f]}에서는 30회 중 {k}회")
    out.append("경계 재검사는 같은 5% 예산에서 무작위 재검사 seed 30회의 평균보다 악성을 더 되찾았지만 양은 작았다"
               + (f"({', '.join(beaten)} 무작위가 경계보다 많이 되찾았다)" if beaten else "") + ". "
               "재검사율을 높이면 되찾는 악성은 늘고 20%의 절약 시간은 기본 분기보다 작아져, 운영자는 허용할 미탐과 시간 사이에서 비율을 고르게 된다.")
    out.append(settings_text(ev))
    out.append("기본 정제 데이터의 GRU는 팀 정제보다 재현율이 높았지만, 학습·검증과 같은 입력을 시험에서 빼도 차이가 거의 그대로였다(5.8절). 이 차이는 같은 입력의 반복보다 정제가 바꾼 학습 자료와 표기의 영향으로 보인다. "
               "접두어 규칙만으로도 정확도 84% 안팎이 나오는 데이터이므로, 다른 시점·출처의 데이터에서 다시 확인해야 한다.")
    return out


def demo_text(ev):
    c = ev.e2c[(ev.e2c.split == "random")]
    both = c.groupby("url").name.nunique()
    url = both[both == 2].index[0]
    rows = c[c.url == url].set_index("name")
    lr, hgb = rows.loc["logistic_boundary"], rows.loc["boost_boundary"]
    return (f"발표 시연은 다섯 사례(정상 조기 종료, 즉시 GRU 전달, 종료 후보 재검사, 재검사로 되찾은 악성, 조기 종료로 놓친 악성)로 구성한다. "
            f"마지막 사례인 URL 무작위 분할 팀 정제의 시험 URL {url}(정답 악성)은 GRU 단독의 악성 점수가 {f2(lr.gru_malicious, 4)}로 악성이지만, "
            f"1단계 점수가 LR {f2(lr.stage1_malicious, 3)}(임계값 {f2(lr.tau)}), HGB {f2(hgb.stage1_malicious, 3)}(임계값 {f2(hgb.tau)})로 낮아 두 경계 재검사 구성 모두에서 정상으로 끝났다. "
            f"주소는 클릭할 수 없게 hxxp와 [.]로 바꿔 표시한다.")


# ---------------------------------------------------------------- chapter 5

def chapter5(rep):
    ev = rep.ev
    rep.chapter(5)
    rep.p("이 장의 수치는 팀 정제 데이터의 전체 시험이 기준이다. 기본 정제와의 비교는 5.8절에 모았다. 백분율은 소수점 둘째 자리, 시간은 1,000건당 ms다.")
    section_metrics(rep, ev)
    section_decisions(rep, ev)
    section_recheck(rep, ev)
    section_thresholds(rep, ev)
    section_misses(rep, ev)
    section_equal_fpr(rep, ev)
    section_shortcuts(rep, ev)
    section_cleaning(rep, ev)


def section_metrics(rep, ev):
    rep.sub("탐지 지표")
    order = ["gru", "url_length", "prefix_rule", "logistic_alone", "boost_alone", *CASCADES]
    for split in SPLITS:
        kind = TEAM[split]
        g = ev.m(kind, "gru")
        rates, counts = [], []
        for name in order:
            if name in ("gru", *CASCADES):
                r = ev.m(kind, name)
                share = pct(r.t2_call_rate, 1)
            else:
                r = ev.base(split, "team", name)
                share = "0.0"
            d = metric_row(r, share)
            rates.append([NAMES[name], d["acc"], d["prec"], d["rec"], d["f1"], d["ap"], d["fpr"], d["share"]])
            extra = ["—", "—", "—"]
            if name in CASCADES:
                extra = [num(r.additional_misses),
                         "—" if name.endswith("basic") else num(r.recovered_malicious),
                         "—" if name.endswith("basic") else num(r.new_false_positives)]
            counts.append([NAMES[name], d["tp"], d["fp"], d["tn"], d["fn"], *extra])
        n, mal = int(g.n), int(g.tp + g.fn)
        rep.t(f"{SPLIT_NAME[split]} 분할 · 전체 시험 {num(n)}건(악성 {num(mal)}건)의 탐지 지표 (%, PR-AUC는 0~1)",
              ["구성", "정확도", "정밀도", "재현율", "F1", "PR-AUC", "오탐률", "GRU 전달"], rates,
              [3.3, 1.8, 1.8, 1.8, 1.7, 1.9, 1.8, 1.9])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 혼동행렬과 재검사 (건)",
              ["구성", "TP", "FP", "TN", "FN", "추가 미탐", "복구 악성", "신규 오탐"], counts,
              [3.3, 1.8, 1.6, 1.8, 1.6, 2.0, 1.95, 1.95])
    rep.p("TP는 악성을 악성으로, FP는 정상을 악성으로, TN은 정상을 정상으로, FN은 악성을 정상으로 판정한 건수다. 재현율은 악성 재현율 TP/(TP+FN), 오탐률은 FP/(FP+TN)이다. "
          "PR-AUC는 악성 점수 순위로 계산한 평균 정밀도다. 캐스케이드는 1단계에서 끝난 URL에 가장 낮은 점수를 주고 계산했다. 접두어 규칙은 점수가 0과 1뿐이다. "
          "GRU 전달은 시험 URL 가운데 GRU까지 간 비율이며, 단순 기준 모델과 1단계 단독은 GRU를 쓰지 않는다. 복구 악성과 신규 오탐은 같은 1단계의 기본 분기와 비교해 재검사가 새로 바꾼 건수다.")
    for split in SPLITS:
        kind = TEAM[split]
        g = ev.m(kind, "gru")
        h, l = ev.base(split, "team", "boost_alone"), ev.base(split, "team", "logistic_alone")
        c = ev.main[(ev.main.kind == kind) & ev.main.name.isin(CASCADES)]
        rep.p(f"{SPLIT_NAME[split]} 분할에서 GRU 단독의 정확도는 {pct(g.accuracy)}%, 재현율은 {pct(g.recall)}%, 오탐률은 {pct(g.fpr)}%였다. "
              f"1단계만 쓰면 HGB 단독 {pct(h.accuracy)}%, LR 단독 {pct(l.accuracy)}%로 GRU보다 크게 낮았다. 캐스케이드 여섯 개의 정확도는 {pct(c.accuracy.min())}~{pct(c.accuracy.max())}%, "
              f"재현율은 {pct(c.recall.min())}~{pct(c.recall.max())}%로 GRU 단독에 가까웠다. 조기 종료가 GRU의 오탐 일부도 함께 없애 캐스케이드의 오탐률은 {pct(c.fpr.min())}~{pct(c.fpr.max())}%로 GRU 단독보다 낮았다.")


def section_decisions(rep, ev):
    rep.sub("1%p 허용치와 시간 판정")
    for split in SPLITS:
        kind = TEAM[split]
        rows, stages = [], []
        for name in ("gru", *CASCADES):
            r = ev.m(kind, name)
            if name == "gru":
                rows.append([NAMES[name], "0", "0.00 [0.00, 0.00]", f2(r.total_ms), "기준", "—", "기준"])
            else:
                rows.append([NAMES[name], num(r.additional_misses), f"{f2(100 * r.recall_loss)} {ci(r.loss_lo, r.loss_hi, 2, 100)}",
                             f2(r.total_ms), f"{sgn(r.saved_ms)} {ci(r.saved_lo, r.saved_hi)}", f2(r.saved_pct, 1), r.decision])
            stages.append([NAMES[name], f2(r.total_ms), f"{f2(r.min_ms)}~{f2(r.max_ms)}", f2(r.input_ms), f2(r.score_ms), f2(r.route_ms), f2(r.gru_ms), num(r.calls)])
        rep.t(f"{SPLIT_NAME[split]} 분할 · GRU 단독 대비 추가 미탐, 재현율 손실과 1,000건 시간 (95% 구간, 측정 A). 단축률은 반복마다 GRU 단독 대비 줄어든 비율의 30회 중앙값",
              ["구성", "추가 미탐 (건)", "재현율 손실 (%p)", "전체 (ms)", "절약 (ms)", "단축률 (%)", "판정"], rows, [2.6, 1.6, 3.2, 1.5, 3.4, 1.5, 2.2])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 1,000건당 단계별 시간 중앙값과 전체 시간의 30회 범위 (ms, 측정 A)",
              ["구성", "전체", "전체 범위", "입력", "1단계", "분기·재검사", "GRU", "GRU 호출"], stages, [2.9, 1.5, 2.6, 1.6, 1.5, 2.3, 1.5, 2.1])
    m = ev.main[ev.main.kind.isin(TEAM.values()) & (ev.main.family != "gru")]
    share = (m.input_ms / m.total_ms)
    assert (m.decision == "충족·단축").all()
    st = m.groupby("family").score_ms
    rep.p(f"두 분할의 캐스케이드 12개가 모두 ‘충족·단축’이었다. 입력은 모든 URL의 정규화·수치 특징 계산과 GRU로 보낸 URL의 문자 토큰 계산을 합한 시간이며, 전체 시간의 {rng(share.min(), share.max(), lambda v: pct(v, 0))}%를 차지했다. "
          f"1단계는 LR이 {f2(st.min()['logistic'])}~{f2(st.max()['logistic'])} ms, HGB가 {f2(st.min()['boost'])}~{f2(st.max()['boost'])} ms였다. HGB는 GRU 호출을 더 많이 줄였지만 1단계 계산이 길어, 도메인 분할에서는 LR보다 덜 절약했다.")
    pp = policy_time_pairs()
    added = [int(ev.m(TEAM[s], f + "_boundary").calls - ev.m(TEAM[s], f + "_basic").calls) for s in SPLITS for f in ("logistic", "boost")]
    assert (pp["diff"] > 0).all()
    rep.p(f"절약 시간은 1,000건당 {rng(m.saved_ms.min(), m.saved_ms.max(), f2)} ms 수준이다. 같은 분할·1단계에서 5% 재검사 구성과 기본 분기를 같은 반복끼리 비교하면, "
          f"재검사 구성이 1,000건당 {f2(pp['diff'].min())}~{f2(pp['diff'].max())} ms 더 걸렸고 8개 비교 가운데 {int((pp.lo > 0).sum())}개에서 95% 구간(짝 부트스트랩 10,000회)이 0보다 컸다. "
          f"재검사한 URL만큼 1,000건 안의 GRU 호출이 {rng(min(added), max(added), num)}건 늘기 때문이다.")


def section_recheck(rep, ev):
    rep.sub("재검사: 같은 예산의 무작위와 경계, 재검사율")
    seeds = ev.seeds
    rows = []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            kind = TEAM[split]
            s = seeds[(seeds.kind == kind) & (seeds.family == fam)]
            assert len(s) == 30
            r, b = ev.m(kind, fam + "_random"), ev.m(kind, fam + "_boundary")
            cand = int(ev.m(kind, fam + "_basic").n - ev.m(kind, fam + "_basic").t2_calls)
            assert int(b.rechecks) == int(r.rechecks) == cand * 5 // 100
            rows.append([f"{SPLIT_NAME[split]} · {FAM[fam]}", num(cand), f"{num(b.rechecks)} ({pct(b.rechecks / cand)}%)",
                         f"{num(r.recovered_malicious)} / {f2(s.recovered_malicious.mean(), 1)} [{num(s.recovered_malicious.min())}~{num(s.recovered_malicious.max())}] / {num(b.recovered_malicious)}",
                         f"{num(r.new_false_positives)} / {f2(s.new_false_positives.mean(), 1)} [{num(s.new_false_positives.min())}~{num(s.new_false_positives.max())}] / {num(b.new_false_positives)}"])
    rep.t("5% 예산의 종료 후보, 재검사, 복구 악성과 신규 오탐 (건, 전체 시험). 복구·오탐은 무작위 seed 0 / seed 30회 평균 [최소~최대] / 경계",
          ["분할 · 1단계", "종료 후보", "재검사 (비율)", "복구 악성", "신규 오탐"], rows, [3.2, 1.7, 2.9, 4.1, 4.1])
    above, ratios = [], []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            s = seeds[(seeds.kind == TEAM[split]) & (seeds.family == fam)].recovered_malicious
            b = ev.m(TEAM[split], fam + "_boundary").recovered_malicious
            assert b > s.mean()
            if b > s.max():
                above.append(f"{SPLIT_NAME[split]} {FAM[fam]}")
            ratios.append(b / ev.m(TEAM[split], fam + "_basic").additional_misses)
    rep.p("두 방식은 같은 종료 후보에서 같은 건수를 다시 검사했다. 경계 재검사는 네 경우 모두 무작위 seed 30회의 평균보다 악성을 더 되찾았고, "
          f"{', '.join(above)}에서는 30회의 최댓값보다도 많았다. 신규 오탐은 방식에 따라 큰 차이가 없었다. "
          f"경계 재검사가 되찾은 악성은 기본 분기 추가 미탐의 {pct(min(ratios), 0)}~{pct(max(ratios), 0)}%였다.")
    if ev.e6 is None:
        rep.p("[초안: 실험 6 결과 대기]")
        return
    e6 = ev.e6
    base = e6[e6.policy == "basic"].set_index(["split", "family"]).batch_calls
    one = e6[np.isclose(e6.rate, 0.01)]
    added = [int(r.batch_calls - base[(r.split, r.family)]) for r in one.itertuples()]
    rep.p(f"재검사율 1·5·10·20%의 실제 처리 시간은 실험 6(측정 B)에서 팀 정제 두 분할만 따로 쟀다. 측정 A와 같은 1,000건, 같은 측정 경로, 예열 1회와 30회 무작위 순서다. "
          f"재검사 예산은 1,000건 안의 종료 후보로 계산하므로 1%에서 1,000건 안의 실제 재검사는 {rng(min(added), max(added), num)}건이었다. 탐지 값은 같은 비율을 전체 시험에 적용해 계산했다. "
          f"측정 B에서 GRU 단독의 1,000건 처리 시간 중앙값은 URL 무작위 {f2(e6[(e6.split == 'random') & (e6.family == 'gru')].iloc[0].total_ms)} ms, "
          f"도메인 {f2(e6[(e6.split == 'domain') & (e6.family == 'gru')].iloc[0].total_ms)} ms였다. 측정 B의 구성별 전체·단계별 시간은 부록 E에 있다.")
    if ev.e6 is None:
        rep.p("[초안: 실험 6 결과 대기]")
        return
    for split in SPLITS:
        d = ev.e6[ev.e6.split == split]
        g = d[d.family == "gru"].iloc[0]
        rows = [["GRU 단독", "1,000", "—", "기준", "—", "0", "—", "기준"]]
        for fam in ("logistic", "boost"):
            for policy, rate in RATE_ORDER:
                r = d[(d.family == fam) & (d.policy == policy) & np.isclose(d.rate, rate)].iloc[0]
                label = rate_label(fam, policy, rate)
                rows.append([label, num(r.batch_calls), num(r.rechecks) if policy != "basic" else "0",
                             f"{sgn(r.saved_ms)} {ci(r.saved_lo, r.saved_hi)}",
                             num(r.recovered_malicious) if policy != "basic" else "—", num(r.additional_misses),
                             num(r.new_false_positives) if policy != "basic" else "—", SHORT[r.decision]])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 재검사율별 실측 시간과 탐지 (실험 6, 측정 B). 호출·시간은 같은 1,000건 30회, 재검사·탐지는 전체 시험",
              ["구성", "GRU 호출", "재검사", "절약 (ms) [95%]", "복구 악성", "추가 미탐", "신규 오탐", "판정"], rows,
              [3.5, 1.6, 1.6, 3.5, 1.2, 1.25, 1.25, 2.0])
    figure_rates(ev, OUT / "rates.png")
    rep.img(OUT / "rates.png", "재검사율에 따른 절약 시간(위, 세로선은 95% 구간)과 기본 분기 대비 되찾은 악성(아래). 0%는 기본 분기다. 실선은 경계, 점선은 무작위 재검사다.")
    rep.p(rates_text(ev))
    rep.p(slow_text(ev))


def slow_repeats(ev):
    """Per split: rows above each specification's Q3 + 1.5 IQR, and repeats where every specification is above it."""
    out = {}
    for split, d in ev.e6raw.items():
        g = d.groupby(["family", "policy", "rate"]).total_ms
        q1, q3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
        hi = d.total_ms > q3 + 1.5 * (q3 - q1)
        per = hi.groupby(d.repeat).sum()
        n_spec = d.groupby(["family", "policy", "rate"]).ngroups
        out[split] = (int(hi.sum()), len(d), sorted(int(r) for r in per[per == n_spec].index), d.total_ms.min(), d.total_ms.max())
    return out


def rates_text(ev):
    d = ev.e6[ev.e6.family != "gru"]
    met = d[d.decision == "충족·단축"]
    held = d[d.decision != "충족·단축"]
    for (split, fam, policy), g in d[d.policy != "basic"].groupby(["split", "family", "policy"]):
        zero = d[(d.split == split) & (d.family == fam) & (d.policy == "basic")].iloc[0].saved_ms
        assert g[np.isclose(g.rate, 0.20)].iloc[0].saved_ms < zero, (split, fam, policy)
    cmp_ = ev.e6v1
    parts = []
    def why(r):
        if r.decision != "시간 또는 미탐 기준 미충족":
            return ""
        assert r.saved_hi < 0 or r.loss_lo > .01
        return ": GRU 단독보다 느림" if r.saved_hi < 0 else ": 손실 1%p 초과"

    for decision in ("유보", "시간 또는 미탐 기준 미충족"):
        items = held[held.decision == decision]
        if len(items):
            parts.append(f"‘{SHORT[decision]}’ {len(items)}개(" + ", ".join(
                e6_label(r) + why(r) for r in items.sort_values(["split", "family", "policy", "rate"]).itertuples()) + ")")
    assert sum(len(held[held.decision == d]) for d in ("유보", "시간 또는 미탐 기준 미충족")) == len(held)
    assert (held.loss_hi < .01).all()      # every non-met specification fails on time, not on the recall loss
    changes = session_changes(ev)
    assert all(r.saved_ms < 1 and ev.m(TEAM[r.split], f"{r.family}_{r.policy}").saved_ms < 1 for r, _ in changes)
    return (f"실험 6의 캐스케이드 사양 36개(두 분할 × 1단계 2개 × 9개) 가운데 {len(met)}개가 ‘충족·단축’이었고, " + ("나머지는 " + ", ".join(parts) + "였다. " if parts else "")
            + ("이들은 모두 재현율 손실 상한이 1%p 미만이었고 시간 조건만 충족하지 못했다. " if len(held) else "")
            + f"모든 분할·1단계·선정 방식에서 20%의 절약 시간 중앙값은 기본 분기(0%)보다 작았다. 두 측정에 모두 들어간 0%·5% 구성 {len(cmp_)}개는 1,000건 안의 GRU 호출 수가 모두 같았고, "
            f"절약 시간의 95% 구간은 {int(cmp_.intervals_overlap.sum())}개에서 두 측정이 서로 겹쳤다. "
            + (f"그러나 두 측정에 공통인 캐스케이드 {int((cmp_.name != 'gru').sum())}개 가운데 판정이 달라진 것은 {len(changes)}개였다. {changes_sentence(changes)}. 이 구성은 두 측정 모두 절약 폭이 1 ms 미만이었다. " if changes else "판정도 두 측정에서 모두 같았다. ")
            + "전체 시간의 절댓값은 측정마다 조금 달라 두 측정의 ms를 직접 비교하지 않는다.")


def slow_text(ev):
    sr = slow_repeats(ev)
    whole = [f"{SPLIT_NAME[s]} {', '.join(map(str, v[2]))}번째" for s, v in sr.items() if v[2]]
    text = (f"측정 B에서 각 구성의 30회 분포로 본 이상치(3사분위수 + 1.5 × 사분위범위 초과)는 "
            + ", ".join(f"{SPLIT_NAME[s]} {num(v[1])}행 중 {num(v[0])}행(전체 범위 {f2(v[3])}~{f2(v[4])} ms)" for s, v in sr.items()) + "이었다. ")
    if whole:
        text += (f"{', '.join(whole)} 반복에서는 모든 구성이 함께 느려져, 측정 중 다른 부하가 있었던 흔적으로 보인다. "
                 "절약 시간은 같은 반복 안의 GRU 단독과의 짝차이로 계산하므로 반복 전체에 걸친 지연의 영향은 줄어들지만 원인은 기록하지 못했다. 판정에는 모든 반복을 포함했다.")
    return text


def section_thresholds(rep, ev):
    rep.sub("1단계 임계값 민감도")
    figure_thresholds(ev, OUT / "thresholds.png")
    rep.img(OUT / "thresholds.png", "검증 후보 임계값 12개의 기본 분기. 위는 1,000건의 절약 시간, 아래는 전체 시험의 재현율 손실이며 세로선은 95% 구간이다. 동그라미는 검증에서 고른 임계값, ×는 손실 상한을 추정할 수 없는 점이다.")
    rows = []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            g = ev.grid[(ev.grid.kind == TEAM[split]) & (ev.grid.family == fam) & (ev.grid.policy == "basic") & (ev.grid.tau > 0)]
            ok = g[g.decision == "충족·단축"]
            rows.append([f"{SPLIT_NAME[split]} · {FAM[fam]}", f2(ev.tau(TEAM[split], fam)),
                         ", ".join(f"{t:g}" for t in sorted(ok.tau)) or "없음",
                         ", ".join(f"{t:g}" for t in sorted(g[g.loss_hi >= 0.01].tau)) or "없음"])
    rep.t("임계값별 판정 요약 (기본 분기, 임계값 0 제외)", ["분할 · 1단계", "선택 임계값", "충족·단축인 임계값", "손실 상한 1%p 이상인 임계값"], rows,
          [3.6, 2.2, 4.4, 5.8])
    be = {}
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            g = ev.grid[(ev.grid.kind == TEAM[split]) & (ev.grid.family == fam) & (ev.grid.policy == "basic")].sort_values("tau")
            assert g.fpr.is_monotonic_decreasing
            be[split, fam] = 1000 - int(g[g.saved_lo > 0].iloc[0].calls)
    assert min(be["random", "boost"], be["domain", "boost"]) > max(be["random", "logistic"], be["domain", "logistic"])
    rep.p("임계값을 높이면 GRU 호출이 줄어 대체로 절약 시간이 커졌지만(인접한 후보 사이에서는 예외가 있었다), 재현율 손실도 빠르게 커졌다. 오탐률은 임계값을 높일수록 낮아졌다(부록 C). 검증에서 고른 임계값은 두 조건을 함께 충족하는 좁은 범위에 있었다. "
          "각 점의 구간은 개별 비교용이며, 여러 임계값을 동시에 본 보정은 하지 않았다.")
    rep.p(f"NoScope가 단계별 비용과 통과율을 함께 본 것처럼 [8], 임계값마다 GRU 호출 감소와 절약 시간을 함께 봤다. 후보 임계값 가운데 절약 시간 구간이 처음 0보다 커진 곳에서 1,000건 중 줄어든 GRU 호출은 "
          f"LR이 URL 무작위 {num(be['random', 'logistic'])}건·도메인 {num(be['domain', 'logistic'])}건, HGB가 {num(be['random', 'boost'])}건·{num(be['domain', 'boost'])}건이었다. "
          "HGB는 1단계 계산이 길어 GRU 호출을 더 많이 줄여야 시간 이득이 났다.")


def level_label(group, level):
    fixed = {("prefix", "http(s):// 있음"): "접두어 있음", ("prefix", "없음"): "접두어 없음",
             ("www", "www. 있음"): "www. 있음", ("www", "없음"): "www. 없음",
             ("ip_host", "IP 주소"): "IP 주소 호스트", ("ip_host", "도메인 이름"): "도메인 이름 호스트"}
    if (group, level) in fixed:
        return fixed[(group, level)]
    return {"length_quartile": "길이 ", "source_file": "출처 "}.get(group, "") + level


def section_misses(rep, ev):
    rep.sub("추가 미탐의 유형과 대표 사례")
    groups = [("prefix", "접두어"), ("www", "www."), ("ip_host", "호스트"), ("length_quartile", "길이"),
              ("has_path_or_query", ""), ("source_file", "출처")]
    for split in SPLITS:
        b = ev.e2b[ev.e2b.split == split]
        rows = []
        for g, title in groups:
            part = b[(b.group == g) & (b.name == "logistic_basic")]
            levels = sorted(part.level) if g == "length_quartile" else list(part.level)
            for level in levels:
                cells = [level_label(g, level)]
                for name in ("logistic_basic", "boost_basic"):
                    r = b[(b.group == g) & (b.name == name) & (b.level == level)].iloc[0]
                    if name == "logistic_basic":
                        cells.append(num(r.gru_detected))
                    cells.append(f"{num(r.additional_misses)} ({f2(r.miss_rate_pct)}%)")
                rows.append(cells)
        q = "·".join(f"{float(v):g}" for v in ev.e2s[(ev.e2s.split == split) & (ev.e2s.name == "logistic_basic")].iloc[0].train_length_quartiles.strip("[]").split(","))
        rep.t(f"{SPLIT_NAME[split]} 분할 · 기본 분기의 추가 미탐 유형. 미탐 (해당 유형 안의 미탐률)",
              ["유형 · 수준", "GRU가 잡은 악성", "LR 기본", "HGB 기본"], rows, [6.2, 3.1, 3.35, 3.35])
        rep.p(f"길이 Q1~Q4는 학습 URL 길이의 사분위수 {q}자를 경계로 나눴고, 경계값과 같은 길이는 낮은 구간에 넣었다. 출처는 공개 원본 CSV 이름이다.")
    s = ev.e2s
    def get(split, name, col):
        return s[(s.split == split) & (s.name == name)].iloc[0][col]
    b = ev.e2b
    def share(split, name, group, level):
        return b[(b.split == split) & (b.name == name) & (b.group == group) & (b.level == level)].iloc[0].share_of_misses_pct
    both = get('random', 'boost_basic', 'miss_another_file_without_prefix')
    total = get('random', 'boost_basic', 'additional_misses')
    rep.p(f"추가 미탐은 접두어가 없는 URL과 df_another_validation.csv에서 온 URL에 몰렸다. URL 무작위 HGB 기본의 추가 미탐 {num(total)}건 가운데 "
          f"{num(both)}건({pct(both / total, 1)}%)이 이 파일에서 온 접두어 없는 URL이었다. "
          f"LR 기본의 추가 미탐은 긴 URL(4사분위)에 몰렸다(URL 무작위 {f2(share('random', 'logistic_basic', 'length_quartile', 'Q4'), 1)}%, 도메인 {f2(share('domain', 'logistic_basic', 'length_quartile', 'Q4'), 1)}%). "
          "IP 주소 호스트와, 접두어를 지운 뒤 www.로 시작하는 URL은 거의 놓치지 않았다. 표의 ‘www. 있음’은 정규화한 URL에서 맨 앞의 http(s)://를 지운 뒤 www.로 시작하는 경우다. 1단계가 접두어·길이 같은 표면 특징으로 ‘정상 같다’고 본 악성을 GRU가 문자 패턴으로 잡은 경우로 해석할 수 있다. TLD별 표는 부록에 있다.")
    rows = []
    for split in SPLITS:
        for name in CASCADES:
            r = s[(s.split == split) & (s.name == name)].iloc[0]
            rows.append([f"{SPLIT_NAME[split]} · {NAMES[name]}", num(r.additional_misses), num(r.false_positives_removed_by_early_exit),
                         num(r.gru_false_positives)])
    rep.t("조기 종료가 바꾼 GRU 판정 (건). 놓친 악성과 사라진 오탐", ["분할 · 구성", "추가 미탐", "사라진 GRU 오탐", "GRU 단독 오탐"], rows,
          [5.2, 3.5, 3.8, 3.5])
    ov = {sp: int(get(sp, "overlap_basic", "additional_misses")) for sp in SPLITS}
    rep.p(f"조기 종료는 GRU가 악성으로 오판할 정상 URL도 함께 정상으로 끝냈다. 도메인 분할 HGB 기본은 추가 미탐 {num(get('domain', 'boost_basic', 'additional_misses'))}건과 맞바꿔 GRU 오탐 "
          f"{num(get('domain', 'boost_basic', 'false_positives_removed_by_early_exit'))}건을 없앴다. GRU 단독이 잡은 악성 가운데 LR 기본과 HGB 기본이 모두 추가로 놓친 것은 URL 무작위 {num(ov['random'])}건, 도메인 {num(ov['domain'])}건으로, 두 구성의 추가 미탐은 대부분 서로 달랐다.")
    rows = []
    for r in ev.e2c.itertuples():
        rows.append([f"{SPLIT_NAME[r.split]} · {NAMES[r.name]}", r.url, f"{f2(r.stage1_malicious, 4)} / {f2(r.tau)}", f2(r.gru_malicious, 4)])
    rep.t("경계 재검사 구성의 추가 미탐 대표 사례 (원본 행 ID의 SHA-256 순 3건씩)", ["분할 · 구성", "URL (무력화 표기)", "1단계 점수 / 임계값", "GRU 점수"], rows,
          [3.6, 8.0, 2.6, 1.8])
    rep.p("주소는 클릭할 수 없도록 http를 hxxp로, 점을 [.]로 바꾸고 70자에서 잘랐다. 모두 정답이 악성이고 GRU 단독은 악성으로 판정했다. 1단계 점수가 임계값보다 낮아 모두 정상 종료 후보에 들었지만, 후보 가운데 점수 상위 5%에 들지 못해 GRU로 전달되지 않았다.")
    assert (ev.e2c.stage1_malicious < ev.e2c.tau).all()


def section_equal_fpr(rep, ev):
    rep.sub("같은 오탐률에서의 비교")
    e = ev.e5
    for split in SPLITS:
        rows, rows2 = [], []
        for name in CASCADES:
            r = e[(e.split == split) & (e.name == name)].iloc[0]
            rows.append([NAMES[name], pct(r.val_cascade_fpr), f2(r.gru_threshold_equal_fpr, 4),
                         f"{pct(r.test_gru_fpr_equal_fpr)} / {pct(r.test_cascade_fpr)}",
                         f"{pct(r.test_gru_recall_equal_fpr)} / {pct(r.test_cascade_recall)}",
                         f"{sgn(r.recall_diff_gru_minus_cascade_pp)} {ci(r.recall_diff_lo_pp, r.recall_diff_hi_pp)}"])
            rows2.append([NAMES[name], pct(r.val_cascade_recall), f2(r.gru_threshold_equal_recall, 4),
                          f"{pct(r.test_gru_recall_equal_recall)} / {pct(r.test_cascade_recall)}",
                          f"{pct(r.test_gru_fpr_equal_recall)} / {pct(r.test_cascade_fpr)}",
                          f"{sgn(r.fpr_diff_gru_minus_cascade_pp)} {ci(r.fpr_diff_lo_pp, r.fpr_diff_hi_pp)}"])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 검증 오탐률을 맞춘 GRU 단독과 캐스케이드의 시험 결과 (%)",
              ["구성", "검증 오탐률 (목표)", "GRU 기준", "시험 오탐률 GRU / 캐스케이드", "시험 재현율 GRU / 캐스케이드", "재현율 차이 (%p) [95%]"], rows,
              [2.7, 1.9, 1.6, 3.0, 3.0, 3.8])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 검증 재현율을 맞춘 GRU 단독과 캐스케이드의 시험 결과 (%)",
              ["구성", "검증 재현율 (목표)", "GRU 기준", "시험 재현율 GRU / 캐스케이드", "시험 오탐률 GRU / 캐스케이드", "오탐률 차이 (%p) [95%]"], rows2,
              [2.7, 1.9, 1.6, 3.0, 3.0, 3.8])
    rnd = e[(e.split == "random") & e.name.isin(CASCADES)]
    dom = e[(e.split == "domain") & e.name.isin(CASCADES)]
    dl = dom[dom.name.str.startswith("logistic")]
    dh = dom[dom.name.str.startswith("boost")]
    assert (rnd.recall_diff_lo_pp > 0).all() and (rnd.fpr_diff_hi_pp < 0).all() and (dl.recall_diff_lo_pp > 0).all()
    assert ((dh.recall_diff_lo_pp < 0) & (dh.recall_diff_hi_pp > 0)).all() and (dh.fpr_diff_lo_pp > 0).all()
    rep.p(f"GRU 기준은 악성 점수가 그 값보다 클 때 악성으로 보는 기준이며, 검증 점수 가운데서 골랐다. 차이는 시험에서 GRU 단독의 값에서 캐스케이드의 값을 뺀 것이다. 95% 구간은 URL 무작위 분할에서 재현율은 악성 행, 오탐률은 정상 행을, 도메인 분할에서는 해당 행이 있는 등록 도메인을 다시 뽑아 구했다. "
          f"URL 무작위 분할에서는 검증 오탐률을 맞추면 시험 오탐률도 거의 같았다. 이때 GRU 단독의 시험 재현율이 {f2(rnd.recall_diff_gru_minus_cascade_pp.min())}~{f2(rnd.recall_diff_gru_minus_cascade_pp.max())}%p 높았다. "
          f"반대로 검증 재현율을 맞추면 GRU 단독의 시험 오탐률이 {f2(-rnd.fpr_diff_gru_minus_cascade_pp.max())}~{f2(-rnd.fpr_diff_gru_minus_cascade_pp.min())}%p 낮았다. 여섯 구성 모두 두 차이의 95% 구간이 0을 포함하지 않았다. "
          f"도메인 분할에서는 검증에서 맞춘 오탐률이 시험에서 벌어졌다(예: HGB 기본의 시험 오탐률 GRU {pct(dh.iloc[0].test_gru_fpr_equal_fpr)}%, 캐스케이드 {pct(dh.iloc[0].test_cascade_fpr)}%). LR 캐스케이드와 비교할 때 GRU의 재현율이 {f2(dl.recall_diff_gru_minus_cascade_pp.min())}~{f2(dl.recall_diff_gru_minus_cascade_pp.max())}%p 높았지만, "
          f"HGB 캐스케이드와는 재현율 차이의 구간이 0을 포함했고, 검증 재현율을 맞추면 GRU의 시험 오탐률이 {f2(dh.fpr_diff_gru_minus_cascade_pp.min())}~{f2(dh.fpr_diff_gru_minus_cascade_pp.max())}%p 높았다.")
    assert (dom.fpr_diff_gru_minus_cascade_pp > 0).all() and ((dl.fpr_diff_lo_pp < 0) & (dl.fpr_diff_hi_pp > 0)).all()
    rep.p("따라서 URL 무작위 분할에서 캐스케이드가 GRU 단독보다 낮은 오탐률을 보인 것은 GRU 기준을 올려도 얻을 수 있는 효과였고, 검증 오탐률을 맞춘 GRU 단독이 시험에서 악성을 더 잡았다. "
          "도메인 분할에서 검증 재현율을 맞추면 여섯 구성 모두 시험 오탐률이 GRU 단독보다 낮았지만, 오탐률 차이의 95% 구간이 0보다 큰 것은 HGB 세 구성뿐이고 LR 세 구성은 0을 포함했다. 구간은 고정한 모델과 기준에서의 시험 표본 변동만 나타내며, 다시 학습하거나 기준을 다시 고를 때의 변동은 포함하지 않는다. 학습에 없던 도메인에서는 1단계의 수치 특징이 GRU의 일부 오탐을 막는 역할을 한 것으로 보이지만, 한 번의 학습 결과이므로 일반화하지 않는다.")


def prefixless_text(ev):
    f = ev.e4f[ev.e4f.group == "without_prefix"]
    moved = f[f.changed > 0]
    assert set(moved.name) <= {"logistic_boundary", "boost_boundary"}
    return ("접두어가 없는 행은 입력이 그대로지만, 경계 재검사에서는 접두어 있는 행의 1단계 점수가 바뀌면서 종료 후보와 재검사 선정이 함께 달라져 판정이 바뀐 행이 있었다("
            + ", ".join(f"{SPLIT_NAME[r.split]} {NAMES[r.name]} {num(r.changed)}건" for r in moved.itertuples()) + ").")


def section_shortcuts(rep, ev):
    rep.sub("지름길 특징 점검")
    rows = []
    for split in SPLITS:
        for model in ("url_length", "prefix_rule"):
            r = ev.base(split, "team", model)
            rows.append([f"{SPLIT_NAME[split]} · {NAMES[model]}", pct(r.accuracy), pct(r.recall), pct(r.fpr), f2(r.roc_auc, 3)])
    rep.t("단순 기준 모델의 시험 성능 (%, ROC-AUC는 0~1)", ["분할 · 모델", "정확도", "재현율", "오탐률", "ROC-AUC"], rows, [5.2, 2.7, 2.7, 2.7, 2.7])
    g = ev.base("random", "team", "gru")
    ln = [ev.base(sp, "team", "url_length") for sp in SPLITS]
    rep.p(f"URL 길이 모델은 거의 모든 URL을 정상으로 판정했고(재현율 {pct(min(x.recall for x in ln), 1)}~{pct(max(x.recall for x in ln), 1)}%), ROC-AUC는 {f2(min(x.roc_auc for x in ln), 3)}~{f2(max(x.roc_auc for x in ln), 3)}로 0.5보다 낮았다. 팀 정제 URL 무작위 시험에서 악성과 정상의 URL 길이(수치 특징 14번) 중앙값은 각각 {num(g.median_url_length_malicious)}자와 {num(g.median_url_length_benign)}자로 비슷했다. "
          f"이번 URL 길이 모델은 두 분할 모두 악성을 잘 가려내지 못했다. 반면 접두어 규칙은 학습 없이 정확도 {pct(min(ev.base(sp, 'team', 'prefix_rule').accuracy for sp in SPLITS), 1)}~{pct(max(ev.base(sp, 'team', 'prefix_rule').accuracy for sp in SPLITS), 1)}%를 냈다.")
    rows = []
    for split in SPLITS:
        s = ev.e4s[ev.e4s.split == split]
        for part, title in (("train", "학습"), ("test", "시험")):
            for prefix in ("http(s):// 있음", "없음"):
                r = s[(s.part == part) & (s.prefix == prefix)].iloc[0]
                rows.append([f"{SPLIT_NAME[split]} · {title} · {prefix}", num(r.n), num(r.malicious), pct(r.malicious_share, 1)])
    rep.t("원본 URL의 접두어 유무별 악성 비율 (팀 정제)", ["분할 · 역할 · 접두어", "행", "악성", "악성 비율 (%)"], rows, [6.4, 3.2, 3.2, 3.2])
    sh = ev.e4s
    w, wo = sh[sh.prefix != "없음"].malicious_share, sh[sh.prefix == "없음"].malicious_share
    rep.p(f"접두어가 있는 URL은 두 분할의 학습과 시험 모두 {pct(w.min(), 1)}~{pct(w.max(), 1)}%가 악성이었고, 없는 URL은 {pct(wo.min(), 1)}~{pct(wo.max(), 1)}%만 악성이었다. 공개 데이터에서 악성 URL이 주로 접두어를 붙인 형태로 수집된 결과로 보인다. 모델이 이 표기 관행을 배웠는지 확인하려고, 고정 모델에서 시험 URL의 맨 앞 http(s)://만 한 번 지우고 다시 판정했다(실험 4. 학습 직후 저장한 점수 사용). 접두어가 없는 행의 입력은 바뀌지 않는다. " + prefix_change_text())
    for split in SPLITS:
        f = ev.e4f[(ev.e4f.split == split) & (ev.e4f.group == "with_prefix")]
        o = ev.e4o[ev.e4o.split == split]
        rows = []
        for name in ("gru", *CASCADES):
            r = f[f.name == name].iloc[0]
            q = o[o.name == name].iloc[0]
            rows.append([NAMES[name], num(r.lost_detection), num(r.new_detection), num(r.removed_false_positive), num(r.new_false_positive),
                         sgn(q.recall_change_pp), sgn(q.fpr_change_pp)])
        n = f.iloc[0]
        rep.t(f"{SPLIT_NAME[split]} 분할 · 접두어만 지운 고정 모델의 판정 변화. 접두어 있는 시험 {num(n.rows)}행(악성 {num(n.malicious)}행)의 건수와 전체 시험의 변화",
              ["구성", "잃은 탐지", "새 탐지", "사라진 오탐", "새 오탐", "재현율 (%p)", "오탐률 (%p)"], rows, [3.0, 2.2, 1.9, 2.3, 1.9, 2.35, 2.35])
    fr = ev.e4f[(ev.e4f.group == "with_prefix") & (ev.e4f.name == "gru")].set_index("split")
    rep.p(f"잃은 탐지는 실제 악성인데 접두어를 지우자 정상으로 바뀐 건수, 사라진 오탐은 실제 정상인데 악성 판정이 정상으로 바뀐 건수다. 접두어를 지우자 GRU 단독은 접두어 있는 악성 가운데 "
          f"URL 무작위 {num(fr.loc['random'].lost_detection)}건, 도메인 {num(fr.loc['domain'].lost_detection)}건을 놓쳤고, 사라진 오탐은 {num(fr.loc['random'].removed_false_positive)}건과 {num(fr.loc['domain'].removed_false_positive)}건뿐이었다. "
          "판정 변화는 거의 모두 실제 악성을 놓치는 쪽이었다. 문자 토큰은 접두어를 원래 지우므로 바뀌지 않았고, 수치 특징의 접두어 관련 값(abnormal_url, count-http(s), url_length, hostname_length, count-letters)이 바뀐 결과다. "
          "모델이 접두어라는 수집 표기에 크게 기대고 있다는 뜻이며, 실제 서비스처럼 접두어 유무가 라벨과 무관한 입력에서는 성능이 떨어질 수 있다. "
          + prefixless_text(ev))
    b_rng = {}
    for split in SPLITS:
        rows = []
        k = TEAM[split]
        alt = k.replace("original", "stripped")
        ch = [100 * (ev.scope(alt, "prefix_common", n).recall - ev.scope(k, "prefix_common", n).recall) for n in ("gru", *CASCADES)]
        b_rng[split] = f"{min(ch):+.2f}~{max(ch):+.2f}".replace("-", "−")
        pairs = ev.pairs.set_index(["comparison", "base", "name"])
        for name in ("gru", *CASCADES):
            o, s = ev.scope(k, "prefix_common", name), ev.scope(alt, "prefix_common", name)
            p = pairs.loc[("remove_prefix", k, name)]
            rows.append([NAMES[name], sgn(100 * (s.recall - o.recall)), f"{sgn(p.delta_ms)} {ci(p.lo_ms, p.hi_ms)}", p.direction])
        rep.t(f"보조 B · {SPLIT_NAME[split]} 분할 · 접두어를 지우고 다시 학습한 모델의 원래 표기 대비 변화",
              ["구성", "재현율 변화 (%p)", "시간 변화 (ms) [95%]", "시간 방향"], rows, [3.6, 3.6, 5.4, 3.4])
    a = ev.scope(TEAM["random"], "prefix_common", "gru")
    b = ev.scope(TEAM["domain"], "prefix_common", "gru")
    rep.p(f"보조 B는 모델 학습 때 학습·검증·시험 URL의 접두어를 모두 지운 뒤 같은 절차로 처음부터 다시 학습한 결과다. 두 표기에 모두 남은 시험(URL 무작위 {num(a.n)}건, 도메인 {num(b.n)}건)에서 비교했다. "
          f"다시 학습하면 재현율 변화가 URL 무작위에서 {b_rng['random']}%p로 거의 그대로였고, 도메인에서 {b_rng['domain']}%p로 낮아졌다. "
          "다만 두 표기의 모델은 학습 행과 문자 어휘도 달라, 이 하락 폭을 접두어 자체의 기여분으로 나눌 수는 없다. 고정 모델이 접두어에 기대는 정도는 앞의 접두어 제거 판정 변화가 보여 준다.")


def section_cleaning(rep, ev):
    rep.sub("정제 수준 비교: 기본 정제와 팀 정제")
    rep.p("이 절은 같은 분할 배정에서 정제만 다른 두 데이터의 결과를 나란히 놓는다. 두 정제는 학습 자료, 학습 어휘, 수치 특징 입력이 함께 다르므로 차이를 한 원인으로 돌릴 수 없다.")
    for split in SPLITS:
        a, b = ev.cohorts.loc[BASIC[split]], ev.cohorts.loc[TEAM[split]]
        o = ev.e3o.set_index("kind")
        rows = [[t, num(a[k]), num(b[k]), sgn(b[k] - a[k], 0)] for t, k in
                (("학습 행", "train"), ("검증 행", "validation"), ("시험 행", "test"), ("시험 악성", "malicious"), ("시험 정상", "benign"),
                 ("두 정제의 공통 시험 URL", "shared_main_test"))]
        rows.insert(5, ["시험 악성 비율 (%)", pct(a.malicious / a.test), pct(b.malicious / b.test), sgn(100 * (b.malicious / b.test - a.malicious / a.test))])
        rows.append(["학습과 시험의 공통 등록 도메인", num(a.common_train_test_domains), num(b.common_train_test_domains),
                     sgn(b.common_train_test_domains - a.common_train_test_domains, 0)])
        for title, col in (("전체 시험 중 학습·검증과 입력이 같은 행", "exact_full"), ("공통 시험 중 학습·검증과 입력이 같은 행", "exact_common")):
            x, y = o.loc[BASIC[split], col], o.loc[TEAM[split], col]
            rows.append([title, num(x), num(y), sgn(y - x, 0)])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 두 정제의 역할과 시험 구성", ["지표", "기본 정제", "팀 정제", "차이 (팀−기본)"], rows, [7.0, 3.0, 3.0, 3.0])
    for split in SPLITS:
        rows, rows2 = [], []
        for name in ("gru", *CASCADES):
            x, y = ev.m(BASIC[split], name), ev.m(TEAM[split], name)
            assert x.decision == y.decision
            rows.append([NAMES[name], f"{pct(x.accuracy)} / {pct(y.accuracy)}", f"{pct(x.recall)} / {pct(y.recall)}",
                         f"{pct(x.fpr)} / {pct(y.fpr)}", f"{num(x.additional_misses)} / {num(y.additional_misses)}", x.decision])
            x, y = ev.scope(BASIC[split], "common", name), ev.scope(TEAM[split], "common", name)
            rows2.append([NAMES[name], f"{pct(x.accuracy)} / {pct(y.accuracy)}", f"{pct(x.recall)} / {pct(y.recall)}",
                          sgn(100 * (y.recall - x.recall)), f"{pct(x.fpr)} / {pct(y.fpr)}"])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 각 정제의 전체 시험 지표. 기본 정제 / 팀 정제 (%, 건)", ["구성", "정확도", "재현율", "오탐률", "추가 미탐", "판정 (두 정제)"], rows,
              [2.9, 3.0, 3.0, 2.6, 2.3, 2.2])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 공통 시험 {num(x.n)}건의 지표. 기본 정제 / 팀 정제 (%)", ["구성", "정확도", "재현율", "재현율 차이 (%p)", "오탐률"], rows2,
              [3.0, 3.4, 3.4, 2.8, 3.4])
    for split in SPLITS:
        gx, gy = ev.m(BASIC[split], "gru"), ev.m(TEAM[split], "gru")
        rows = []
        for name in ("gru", *CASCADES):
            x, y = ev.m(BASIC[split], name), ev.m(TEAM[split], name)
            sv = ["기준", "기준"] if name == "gru" else [f"{sgn(v.saved_ms)} {ci(v.saved_lo, v.saved_hi)}" for v in (x, y)]
            d = "—" if name == "gru" else f"{sgn(100 * (x.fpr - gx.fpr))} / {sgn(100 * (y.fpr - gy.fpr))}"
            rate = "—" if name == "gru" else f"{f2(x.saved_pct, 1)} / {f2(y.saved_pct, 1)}"
            rows.append([NAMES[name], f"{num(x.fp)} / {num(y.fp)}", d, *sv, rate])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 두 정제의 정상 오탐과 1,000건 절약 시간 (전체 시험, 시간은 측정 A). 오탐 건수, GRU 단독 대비 오탐률 변화와 단축률은 기본 정제 / 팀 정제",
              ["구성", "오탐 (건)", "오탐률 변화 (%p)", "절약 · 기본 (ms) [95%]", "절약 · 팀 (ms) [95%]", "단축률 (%)"], rows, [2.9, 2.5, 2.4, 3.2, 3.2, 1.8])
    cc = ev.pairs[ev.pairs.comparison == "change_cleaning"]
    assert (cc.lo_ms > 0).all()
    ga, gb = ev.scope(BASIC["random"], "common", "gru"), ev.scope(TEAM["random"], "common", "gru")
    da, db = ev.scope(BASIC["domain"], "common", "gru"), ev.scope(TEAM["domain"], "common", "gru")
    rep.p(f"같은 시험 URL에서 비교해도 기본 정제의 GRU가 재현율이 높았다. URL 무작위 공통 시험 {num(ga.n)}건(악성 {num(ga.malicious)}건)에서 {pct(ga.recall)}% 대 {pct(gb.recall)}%, "
          f"도메인 공통 시험 {num(da.n)}건에서 {pct(da.recall)}% 대 {pct(db.recall)}%였다. 기본 정제는 오탐률도 낮았다. 두 정제 모두 캐스케이드 6개가 기준을 충족했다. "
          f"측정 A에서 기본 정제 조건은 1,000건당 {f2(cc.delta_ms.min(), 1)}~{f2(cc.delta_ms.max(), 1)} ms 짧았다(팀−기본 짝차이, 기본 정제의 구성별 단계 시간은 부록 F). 기본 정제는 정규화를 하지 않지만, 두 조건은 모델·어휘·GRU 전달 경로도 달라 이 차이를 정규화 비용만으로 볼 수는 없다.")
    rows = []
    names = {"all_common": "공통 시험 전체", "without_exact_input_overlap": "같은 입력 행 제외", "only_exact_input_overlap": "같은 입력 행만",
             "without_prefix52_overlap": "앞 52자+수치 동일 행 제외", "only_prefix52_overlap": "앞 52자+수치 동일 행만"}
    for split in SPLITS:
        s = ev.e3s[ev.e3s.split == split].set_index("subset")
        for key, title in names.items():
            r = s.loc[key]
            rows.append([f"{SPLIT_NAME[split]} · {title}", f"{num(r.basic_n)} ({num(r.basic_malicious)})",
                         f"{pct(r.basic_recall)} / {pct(r.team_recall)}" if r.basic_malicious else "—",
                         sgn(r.recall_diff_pp) if r.basic_malicious else "—",
                         f"{pct(r.basic_accuracy)} / {pct(r.team_accuracy)}"])
    rep.t("실험 3 · 공통 시험에서 ‘기본 정제 학습·검증과 같은 입력’ 행을 빼거나 그 행만 남긴 GRU 비교", ["분할 · 부분집합", "행 (악성)", "재현율 기본 / 팀 (%)", "차이 (%p)", "정확도 기본 / 팀 (%)"], rows,
          [5.2, 2.9, 3.0, 1.7, 3.2])
    s = ev.e3s[ev.e3s.split == "random"].set_index("subset")
    o = ev.e3o.set_index("kind")
    rep.p(f"기본 정제는 집합 사이의 같은 모델 입력을 확인하지 않는다. 그래서 기본 정제 시험에는 학습·검증과 토큰 50칸과 수치 특징 18개가 모두 같은 행이 URL 무작위 {num(o.loc[BASIC['random'], 'exact_full'])}행, "
          f"도메인 {num(o.loc[BASIC['domain'], 'exact_full'])}행 남았다. 팀 정제에는 이런 행이 없다. ‘앞 52자+수치’가 같은 행은 모두 이 ‘같은 입력’ 행에 포함됐다. "
          f"URL 무작위 공통 시험에서 이 {num(s.loc['only_exact_input_overlap'].basic_n)}행을 빼도 두 정제의 GRU 재현율 차이는 {sgn(s.loc['all_common'].recall_diff_pp)}%p에서 {sgn(s.loc['without_exact_input_overlap'].recall_diff_pp)}%p로 거의 그대로였다. "
          "같은 입력 행은 두 모델 모두 거의 다 맞혔다. 따라서 기본 정제의 재현율 우위는 학습에서 본 입력이 시험에 다시 나온 효과로 설명되지 않는다. 남은 원인 후보는 정제가 바꾼 학습 행과 어휘, 정규화 URL로 계산한 수치 특징이지만, 이번 설계로는 이들을 나누어 측정할 수 없다.")


# ---------------------------------------------------------------- appendix

def appendix(rep):
    ev = rep.ev
    rep.chapter(8)
    rep.sub("A. 1단계에서 특징 9개를 뺀 모델 (보조 C)", False)
    rows = []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            full = ev.stage1[(ev.stage1.kind == TEAM[split]) & (ev.stage1.family == fam)].iloc[0]
            red = ev.ablation[(ev.ablation.kind == TEAM[split]) & (ev.ablation.family == fam)].iloc[0]
            rows.append([f"{SPLIT_NAME[split]} · {FAM[fam]}", f2(full.pr_auc_ap, 3), f2(red.pr_auc_ap, 3), sgn(red.pr_auc_ap - full.pr_auc_ap, 3),
                         pct(full.recall), pct(red.recall)])
    rep.t("특징 9개 제외 전후 1단계 단독의 시험 AP와 재현율 (팀 정제)", ["분할 · 1단계", "AP 전체", "AP 9개 제외", "AP 차이", "재현율 전체 (%)", "재현율 9개 제외 (%)"], rows,
          [3.4, 2.1, 2.4, 2.1, 2.9, 3.1])
    omitted = ", ".join(readj(V1 / TEAM["random"] / "controls-final.json")["logistic_omit_columns"]["omitted"])
    rep.p(f"뺀 열은 {omitted}다. 접두어·길이·파싱에 민감한 열이며, 1단계를 처음부터 다시 적합했다. 11월 24일에 추가한 보조 실험으로 확장 선언의 요구 항목은 아니다.")
    rep.sub("B. 1단계 학습량 곡선 (보조 D)", False)
    rows = []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            c = ev.learning[(ev.learning.kind == TEAM[split]) & (ev.learning.family == fam)].sort_values("fraction")
            rows.append([f"{SPLIT_NAME[split]} · {FAM[fam]}", *[f2(v, 4) for v in c.validation_ap]])
    rep.t("학습 표본 비율별 1단계 검증 AP (팀 정제)", ["분할 · 1단계", "10%", "25%", "50%", "75%", "100%"], rows, [4.0, 2.4, 2.4, 2.4, 2.4, 2.4])
    rep.p("선택한 1단계를 학습 표본의 일부로 다시 적합한 결과다. 이번 네 조건에서는 학습 표본을 10%에서 100%로 늘려도 검증 AP 변화가 작았다.")
    rep.sub("C. 임계값별 기본 분기의 전체 측정값 (측정 A)", False)
    rows = []
    for split in SPLITS:
        for fam in ("logistic", "boost"):
            g = ev.grid[(ev.grid.kind == TEAM[split]) & (ev.grid.family == fam) & (ev.grid.policy == "basic")].sort_values("tau")
            for r in g.itertuples():
                loss = f"{f2(100 * r.recall_loss)} " + ("[상한 없음]" if pd.isna(r.loss_hi) else ci(r.loss_lo, r.loss_hi, 2, 100))
                assert r.tau > 0 or (r.recall_loss == 0 and r.loss_hi == 0)
                rows.append([f"{SPLIT_NAME[split]} · {FAM[fam]} {r.tau:g}", num(r.calls), f"{sgn(r.saved_ms)} {ci(r.saved_lo, r.saved_hi)}",
                             loss, pct(r.fpr), SHORT[r.decision]])
    rep.t("임계값별 1,000건 GRU 호출, 절약 시간과 전체 시험 재현율 손실·오탐률(%)", ["분할 · 1단계 임계값", "GRU 호출", "절약 (ms) [95%]", "손실 (%p) [95%]", "오탐률", "판정"], rows,
          [4.0, 1.45, 3.75, 3.5, 1.4, 1.9])
    rep.p("판정의 ‘미충족’은 유보가 아니면서 절약 시간 구간 전체가 0 미만이거나 재현율 손실 구간 전체가 1%p를 넘는 경우다. 임계값 0은 모든 URL을 GRU로 보내므로 손실이 정확히 0이고, 1단계 계산만큼 시간이 늘어 시간 조건으로만 판정한다. "
          "‘상한 없음’은 도메인 분할에서 임계값이 0보다 큰데 추가 미탐이 0건이어서 상한을 추정할 수 없는 경우이며 ‘유보’다.")
    rep.sub("D. 추가 미탐의 TLD별 분포 (실험 2)", False)
    rows = []
    for split in SPLITS:
        b = ev.e2b[(ev.e2b.split == split) & (ev.e2b.group == "tld")]
        for level in b[b.name == "logistic_basic"].level:
            cells = [f"{SPLIT_NAME[split]} · {level}"]
            for name in ("logistic_basic", "boost_basic"):
                r = b[(b.name == name) & (b.level == level)].iloc[0]
                if name == "logistic_basic":
                    cells.append(num(r.gru_detected))
                cells.append(f"{num(r.additional_misses)} ({f2(r.miss_rate_pct)}%)")
            rows.append(cells)
    rep.t("TLD별 GRU가 잡은 악성과 기본 분기의 추가 미탐 (미탐률)", ["분할 · TLD", "GRU가 잡은 악성", "LR 기본", "HGB 기본"], rows, [5.2, 3.6, 3.6, 3.6])
    rep.p("TLD는 등록 도메인의 마지막 이름표다. 분할마다 GRU가 잡은 시험 악성이 많은 TLD 6개를 미탐을 보기 전에 정했고, IP 주소와 점 없는 호스트는 따로, 나머지는 기타로 묶었다.")
    if ev.e6 is not None:
        appendix_e(rep, ev)
    rep.sub("F. 기본 정제의 1,000건 단계별 시간 (측정 A)", False)
    for split in SPLITS:
        rows = []
        for name in ("gru", *CASCADES):
            r = ev.m(BASIC[split], name)
            rows.append([NAMES[name], f2(r.total_ms), f"{f2(r.min_ms)}~{f2(r.max_ms)}", f2(r.input_ms), f2(r.score_ms), f2(r.route_ms), f2(r.gru_ms), num(r.calls)])
        rep.t(f"{SPLIT_NAME[split]} 분할 · 기본 정제의 1,000건당 단계별 시간 중앙값과 전체 시간의 30회 범위 (ms, 측정 A)",
              ["구성", "전체", "전체 범위", "입력", "1단계", "분기·재검사", "GRU", "GRU 호출"], rows, [2.9, 1.5, 2.6, 1.6, 1.5, 2.3, 1.5, 2.1])
    rep.p("팀 정제의 같은 값은 5.2절에 있다. 기본 정제 캐스케이드의 절약 시간·95% 구간과 단축률은 5.8절 표에 있다.")


def appendix_e(rep, ev):
    rep.sub("E. 재검사율별 1,000건 단계별 시간 (실험 6, 측정 B)", False)
    for split in SPLITS:
        d = ev.e6[ev.e6.split == split]
        specs = [("GRU 단독", d[d.family == "gru"].iloc[0])]
        for fam in ("logistic", "boost"):
            for policy, rate in RATE_ORDER:
                specs.append((rate_label(fam, policy, rate), d[(d.family == fam) & (d.policy == policy) & np.isclose(d.rate, rate)].iloc[0]))
        rows = [[label, f2(r.total_ms), f"{f2(r.min_ms)}~{f2(r.max_ms)}", f2(r.input_ms), f2(r.score_ms), f2(r.route_ms), f2(r.gru_ms), num(r.batch_calls)] for label, r in specs]
        rep.t(f"{SPLIT_NAME[split]} 분할 · 재검사율별 1,000건당 단계별 시간 중앙값과 전체 시간의 30회 범위 (ms, 측정 B)", ["구성", "전체", "전체 범위", "입력", "1단계", "분기·재검사", "GRU", "GRU 호출"], rows,
              [3.4, 1.4, 2.6, 1.5, 1.4, 2.3, 1.4, 2.0])
    rep.p("각 값은 같은 1,000건을 30회 반복한 시간의 중앙값이고, 전체 범위는 30회의 최솟값~최댓값이다. 단계별 중앙값의 합은 전체 시간의 중앙값과 조금 다를 수 있다. 입력은 모든 URL의 정규화·수치 특징과 GRU로 보낸 URL의 문자 토큰을 만드는 시간이다.")
