"""Report the validation decision and held-out outcomes at each recheck budget."""
import numpy as np

from report_data import AB, ROOT, csv, kinds, label, num, readj


FAMILIES = (("logistic", "LR"), ("boost", "HGB"))


def selection(report):
    report.t("분기 임계값과 재검사 예산을 정한 방법", ["항목", "선정 기준"], [
        ["분기 임계값 후보", "보정한 1단계 악성 점수를 기준으로 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50을 검증 집합에서 비교했다."],
        ["임계값 선택", "재검사 없이 GRU 단독 대비 악성 재현율 손실이 1%p 이내인 후보 중 GRU 호출이 가장 적은 값을 택했다. 호출 수가 같으면 낮은 임계값을 택했다. 점수가 임계값 이상이면 즉시 GRU로 보낸다."],
        ["재검사 예산 5%", "11월 24일 팀 합의와 최종 계획에서 정상 종료 후보의 5%를 배정했다. 같은 수만큼 추가 검사할 때 어떤 선정 방식이 악성을 더 되찾는지 비교하기 위한 예산이다. 운영에 최적인 비율은 정하지 않았다."],
        ["무작위 선정", "후보 수에 재검사율을 곱해 소수점 이하는 버린다. 시드와 URL 식별자를 결합한 SHA-256 해시가 작은 순으로 그 건수만큼 뽑는다. 주 실험은 시드 0, 무작위 변동 확인은 시드 0~29를 사용했다."],
        ["경계 선정", "같은 후보를 악성 점수가 높은 순으로 정렬해 무작위 방식과 같은 건수를 뽑는다. 동점은 식별자 순으로 정한다. 가장 낮은 선정 점수는 후보 집합과 예산에 따라 달라진다."],
    ], [3.4, 12.6])
    report.p("두 정책 모두 재검사 대상은 GRU의 판정을 따른다. GRU가 출력하는 정상 확률이 0.5 이상이면 정상으로 판정한다.")
    rows = []
    for split in ("random", "domain"):
        for family, name in FAMILIES:
            for form, form_name in (("original", "원래 표기"), ("stripped", "접두어 제거 재학습")):
                values = []
                for kind in kinds(split):
                    gate = readj(kind.replace("original", form) + "/routing.json")["families"][family]
                    best = min((r for r in gate["grid"] if r["recall_loss"] <= .01),
                               key=lambda r: (r["t2_calls"], r["tau"]))
                    assert best["tau"] == gate["tau"]
                    values.append(best["tau"])
                rows.append([label(split) + " · " + name, form_name, *[f"{x:.2f}" for x in values]])
    report.t("검증에서 확정한 1단계 분기 임계값 (악성 점수 0~1)",
             ["분할 · 모델", "학습 표기", *AB], rows, [3.0, 3.9, 4.55, 4.55])
    report.p("시험 URL의 접두어만 바꾼 고정 모델 실험에는 원래 표기의 임계값을 그대로 적용했다.")


def outcomes(report):
    report.sub("종료 후보를 얼마나 다시 검사할 것인가")
    sensitivity = csv("sensitivity").set_index(["kind", "family", "tau", "rate", "policy"])
    assert len(sensitivity) == 864 and sensitivity.index.is_unique
    report.p("원래 표기로 학습한 각 모델에서 검증으로 고른 임계값을 고정하고, 정상 종료 후보의 0·1·5·10·20%를 다시 검사했다. 0%는 기본 분기다. 전체 시험에서 12개 임계값을 훑은 864개 조합 중, 실제 운용 조건으로 선택한 임계값의 72개 결과를 다음 표에 모았다.")
    report.p("표의 네 수치는 차례대로 다시 검사한 URL, 기본 분기에서 놓쳤다가 되찾은 악성, 같은 GRU 단독보다 더 놓친 악성, 기본 분기보다 새로 오탐한 정상 URL의 건수다. 1단계 점수는 0~1이며 최저·최고 점수는 소수점 아래 여섯 자리로 반올림했다.")
    for split in ("random", "domain"):
        pair = kinds(split)
        for family, name in FAMILIES:
            thresholds = [readj(k + "/routing.json")["families"][family]["tau"] for k in pair]
            ranges = []
            for kind, tau in zip(pair, thresholds):
                with np.load(ROOT / kind / "scores.npz") as scores:
                    p = scores[family]
                    ids = scores["id"]
                    c = np.flatnonzero(p < tau)
                    n = len(c) // 20
                    selected = c[np.lexsort((ids[c], -p[c]))[:n]]
                    observed = sensitivity.loc[(kind, family, tau, .05, "boundary")]
                    assert n == observed.rechecks and len(c) == len(p) - observed.t2_calls + n
                    assert int(((scores["y"][selected] == 1) & (scores["gru"][selected] > .5)).sum()) == observed.recovered_malicious
                    ranges.append((len(c), n, p[selected].min(), p[selected].max()))
            context = " · ".join(f"{proc} {num(report.ev.pop.loc[k,'test'])}건(악성 {num(report.ev.pop.loc[k,'malicious'])}건)" for proc,k in zip(("원저자", "팀"), pair))
            report.p(f"{label(split)} 분할의 {name} 전체 시험. {context}.")
            report.t(f"{label(split)} · {name}의 5% 경계 선정 점수", ["항목", *AB], [
                ["분기 임계값", *[f"{t:.2f}" for t in thresholds]],
                ["종료 후보 → 재검사 (건)", *[f"{num(c)} → {num(n)}" for c,n,_,_ in ranges]],
                ["실제 최저~최고 점수", *[f"{lo:.6f}~{hi:.6f}" for _,_,lo,hi in ranges]]
            ], [4.3, 5.85, 5.85])
            rows = []
            for rate in (0., .01, .05, .10, .20):
                policies = (("basic", "기본"),) if rate == 0 else (("random", "무작위"), ("boundary", "경계"))
                for policy, title in policies:
                    vals = [sensitivity.loc[(k, family, t, rate, policy)] for k,t in zip(pair, thresholds)]
                    fields = ("rechecks", "recovered_malicious", "additional_misses", "new_false_positives")
                    cells = [" / ".join(num(v[f]) for f in fields) for v in vals]
                    diff = " / ".join(f"{int(vals[1][f]-vals[0][f]):+,}" for f in fields)
                    rows.append([f"{rate:.0%} {title}", *cells, diff])
            report.t(f"{label(split)} · {name} 재검사율별 결과 (건). 재검사 / 악성 복구 / 추가 미탐 / 신규 정상 오탐",
                     ["비율 · 선정", *AB, "차이 (팀−원저자)"], rows, [2.5,4.6,4.6,4.3])
    selected_tau = readj("team-random-original/routing.json")["families"]["boost"]["tau"]
    example = sensitivity.loc[("team-random-original", "boost", selected_tau, .2, "boundary")]
    baseline = sensitivity.loc[("team-random-original", "boost", selected_tau, .05, "boundary")]
    report.p(f"URL 무작위 분할에서 팀 정제 HGB의 경계 예산을 5%에서 20%로 늘리자 복구한 악성은 {int(baseline.recovered_malicious)}건에서 {int(example.recovered_malicious)}건으로 늘었다. 기본 분기 대비 신규 정상 오탐도 {int(baseline.new_false_positives)}건에서 {int(example.new_false_positives)}건으로 늘었다. 이 두 비율의 시간 차이는 실측하지 않았다.")
    report.t("재검사율에 따른 탐지 평가와 시간 실측 범위", ["조건", "탐지 평가", "전체 처리 시간 실측"], [
        ["원래 표기 · 기본 분기(0%)", "전체 시험, 임계값 12개", "측정. 공통 1,000건을 예열한 뒤 30회 짝측정"],
        ["원래 표기 · 무작위·경계 5%", "전체 시험, 임계값 12개", "측정. 검증에서 선택한 임계값에서 1,000건 × 30회"],
        ["원래 표기 · 무작위·경계 1·10·20%", "전체 시험, 임계값 12개", "미측정. 이 비율의 처리 시간이나 최적 속도 순위는 알 수 없다."],
        ["접두어 제거 재학습", "선택한 임계값의 기본·무작위·경계 5%", "선택한 임계값의 세 구성과 GRU 단독을 1,000건 × 30회 측정"],
    ], [4.1,5.0,6.9])
