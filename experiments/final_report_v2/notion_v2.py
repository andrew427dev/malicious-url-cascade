"""Notion page body (HTML for pasting, Markdown for review): a short summary of the report with its setup and method,
built from the same evidence and shared lists as the report so the numbers cannot drift."""
import html
import re

from report_base import OUT, V1, TEAM, BASIC, SPLITS, SPLIT_NAME, NAMES, CASCADES, Evidence, readj, num, pct, sgn, f2, ci
import report_results as results

TITLE = "악성 URL 탐지 최종 실험 보고서 v2"


def tables(ev):
    main = [["분할 · 구성", "정확도 (%)", "재현율 (%)", "오탐률 (%)", "GRU 전달 (%)", "추가 미탐 (건)", "절약 시간 (ms) [95%]", "판정"]]
    for split in SPLITS:
        for name in ("gru", *CASCADES):
            r = ev.m(TEAM[split], name)
            main.append([f"{SPLIT_NAME[split]} · {NAMES[name]}", pct(r.accuracy), pct(r.recall), pct(r.fpr), pct(r.t2_call_rate, 1),
                         "0" if name == "gru" else num(r.additional_misses),
                         "기준" if name == "gru" else f"{sgn(r.saved_ms)} {ci(r.saved_lo, r.saved_hi)}", r.decision])
    return [("팀 정제 주 결과 (전체 시험, 시간은 측정 A의 1,000건 30회)", main)]


# Sentences of the report's summary kept on the page: (paragraph start, sentence count, kept indexes).
PICK = [("RQ1 · URL 무작위", 3, (0, 1, 2)), ("RQ1 · 도메인", 4, (0, 1, 2, 3)), ("RQ2", 3, (0, 1, 2)), ("RQ3", 4, (0, 1, 2, 3)),
        ("RQ4", 6, (0, 1, 3, 5)), ("같은 오탐률 비교", 7, (0, 1, 2, 3, 4, 5)), ("지름길 점검", 4, (0, 1, 2, 3))]


def short_results(ev):
    """RQ answers for the page: chosen sentences of the report's own summary, so wording and numbers stay the report's."""
    paras = results.summary(ev)
    assert len(paras) == len(PICK), len(paras)
    out = []
    for p, (head, n, keep) in zip(paras, PICK):
        ss = re.split(r"(?<=\.)\s+", p.strip())
        assert p.startswith(head) and len(ss) == n, (head, len(ss))
        out.append(" ".join(ss[i] for i in keep))
    g = {(c, s): ev.scope(k[s], "common", "gru") for c, k in (("basic", BASIC), ("team", TEAM)) for s in SPLITS}
    assert all(g["basic", s].recall > g["team", s].recall for s in SPLITS)
    out.append("정제 수준 비교(실험 3). 두 정제에 모두 남은 같은 시험 URL에서 GRU 단독의 재현율은 기본 정제가 팀 정제보다 높았다("
               + ", ".join(f"{SPLIT_NAME[s]} 공통 시험 {num(g['basic', s].n)}건에서 {pct(g['basic', s].recall)}% 대 {pct(g['team', s].recall)}%" for s in SPLITS)
               + "). 두 정제는 학습 자료·어휘·입력 표기가 함께 달라 이 차이를 한 원인으로 나눌 수 없다(5.8절).")
    out.append("재검사율별 결과(5.3절), 단순 기준 모델과 1단계 단독(5.1·5.7절), 정제 수준 비교(5.8절)의 표와 추가 미탐 사례는 첨부 보고서에 있다.")
    return out


def setup(ev):
    raw = readj(V1 / "source-manifest.json")["raw"]["rows"]
    *_, last, final = results.cleaning_rows()
    tests = ", ".join(f"{SPLIT_NAME[s]} {num(ev.m(TEAM[s], 'gru').n)}건(악성 {num(ev.n_mal(TEAM[s]))}건)" for s in SPLITS)
    taus = ", ".join(f"{SPLIT_NAME[s]} LR {f2(ev.tau(TEAM[s], 'logistic'))}·HGB {f2(ev.tau(TEAM[s], 'boost'))}" for s in SPLITS)
    items = [
        f"데이터: 원저자 공개 CSV 3개({num(raw)}행)를 합쳐 팀 정제를 했다. 잘못된 URL(제어·서식 문자 등)을 빼고, 정규화(스킴·호스트 소문자, 끝 슬래시 제거) 값의 라벨 충돌 목록을 먼저 구한 뒤 "
        f"원본 URL 라벨 충돌 제외 → 같은 원본 URL 반복 제거 → 정규화 URL 라벨 충돌 제외 → 정규화 URL 반복 제거 → 모델 입력(문자 입력·수치 특징) 기준 충돌·반복 제거 → 정규화를 두 번 하면 달라지는 {last[1]}행 제외 순으로 지웠다. 최종 {final[3]}행({final[2]}).",
        "비교용 기본 정제: 원본 URL 라벨 충돌과 반복만 지우고, 집합마다 모델 문자열이 같은 행은 첫 행만 남긴 데이터다(5.8절).",
        f"분할: 정제 전에 URL마다 역할을 배정해 두 정제가 같은 배정을 쓴다. URL 무작위 분할은 정상·악성·라벨 충돌을 층으로 둔 70/15/15(seed 42·43), 도메인 분할은 같은 등록 도메인을 한 역할에만 둔다. 시험 규모는 {tests}.",
        "모델: 원저자 구조의 GRU(문자 3그램 50칸 + 수치 특징 18개), 1단계 로지스틱 회귀(LR)·히스토그램 기반 그래디언트 부스팅(HGB)(수치 특징 18개, 검증 집합에서 시그모이드 보정), 단순 기준 모델(URL 길이 모델, ‘http(s)://로 시작하면 악성’ 접두어 규칙).",
        "구성(분할마다 11개): GRU 단독, 캐스케이드 6개(LR·HGB × 기본 분기·무작위 재검사 5%·경계 재검사 5%), 단순 기준 모델 2개, 1단계 단독 2개.",
    ]
    exps = [f"{e}: {d} (보고서 {loc})" for e, d, loc in results.EXPERIMENTS]
    method = [
        "캐스케이드: 수치 특징으로 1단계 악성 점수를 구하고, 분기 임계값보다 낮으면 정상으로 끝내며 나머지만 GRU로 보낸다. 재검사는 정상으로 끝낼 후보의 일부를 GRU로 다시 보낸다. 무작위는 seed와 URL 식별자의 SHA-256 순, 경계는 1단계 점수가 높은 순으로 같은 건수를 뽑는다.",
        f"임계값: 검증 집합에서 GRU 단독 대비 재현율 손실이 1%p 이내인 후보 가운데 GRU 호출이 가장 적은 값({taus}). 시험 집합은 최종 결과 계산에만 썼다.",
        "탐지 지표: 정확도, 정밀도, 악성 재현율, F1, PR-AUC, 오탐률, 혼동행렬, GRU 전달 비율. 재검사는 되찾은 악성과 신규 오탐을 함께 센다.",
        "시간: 같은 CPU에서 분할마다 고정한 시험 URL 1,000건을 예열 1회 뒤 30회, 구성 순서를 섞어 쟀다. BLAS·OpenMP 스레드는 1개로 제한했고 TensorFlow는 스레드 환경변수를 1로 지정했다(실제 스레드 수는 기록하지 않음). "
        "절약 시간은 같은 반복의 ‘GRU 단독 시간 − 캐스케이드 시간’의 중앙값이고, 95% 구간은 30개 반복을 짝째로 10,000회 다시 뽑아 구했다. "
        "측정 A는 모델 학습 때 네 조건(두 정제 × 두 표기)의 주 구성과 원래 표기 두 정제의 임계값 후보 12개를 함께 잰 것, 측정 B(실험 6)는 팀 정제 두 분할의 재검사율 1·5·10·20%를 따로 잰 것이다.",
        "판정: 재현율 손실 95% 구간의 상한이 1%p 미만이고 절약 시간 95% 구간의 하한이 0보다 크면 ‘충족·단축’, 구간이 기준에 걸치면 ‘유보’, 구간 전체가 기준을 벗어나면 ‘미충족’이다. 손실 구간은 URL 무작위가 Wilson, 도메인이 등록 도메인 묶음 4,000회 재표집이다. 도메인 분할에서 임계값이 0보다 큰데 추가 미탐이 0건이면 손실 상한을 추정할 수 없어 ‘유보’로 둔다.",
        "같은 오탐률 비교(실험 5): 검증 오탐률이 캐스케이드 이하가 되는 가장 작은 GRU 판정 기준(보조로, 검증 악성 탐지 수가 캐스케이드 이상이 되는 가장 큰 기준)을 검증 집합에서 골라 시험 결과를 비교했다.",
    ]
    return items, exps, method


def content(ev):
    intro = [
        "이 보고서는 공개 악성 URL 탐지 모델(문자 3그램 GRU + 수치 특징) 앞에 가벼운 1단계 분류기를 두는 캐스케이드가, 악성 재현율 손실을 1%p 미만으로 지키면서 처리 시간을 줄이는지 확인했다. "
        "주 실험은 팀 정제 데이터이고 URL 무작위 분할과 도메인 분할을 같은 비중으로 다룬다. 모델과 분기 임계값은 분할마다 한 번 학습·선택해 고정했다.",
        "결과는 공개 CSV 세 개를 합쳐 새로 나누고 정제한 데이터에서의 평가이며, 원저자 저장소의 수치와 직접 비교할 수 없다. 원저자 재현과 발표값 참고 비교는 첨부 보고서 2장 표 4·5에 있다.",
    ]
    items, exps, method = setup(ev)
    concl = [results.settings_text(ev),
             "한계: 실험 1~6을 설계하기 전에 시험 결과를 이미 봤고, 같은 검증 집합을 여러 선택에 썼으며, 학습 seed가 하나이고, 배치 처리 시간만 쟀다(보고서 6장)."]
    return intro, items, exps, method, short_results(ev), concl, tables(ev)


def to_html(intro, items, exps, method, summary, concl, tabs):
    e = html.escape
    out = [f"<p>{e(p)}</p>" for p in intro]
    out.append("<h2>실험 구성</h2><ul>" + "".join(f"<li>{e(b)}</li>" for b in items) + "</ul>")
    out.append("<h3>실험 목록</h3><ul>" + "".join(f"<li>{e(b)}</li>" for b in exps) + f"</ul><p>{e(results.EXPERIMENT_ORIGIN)}</p>")
    out.append("<h2>방법</h2><ul>" + "".join(f"<li>{e(b)}</li>" for b in method) + "</ul>")
    out.append("<h2>주요 결과</h2>")
    out += [f"<p>{e(p)}</p>" for p in summary]
    out.append("<h2>결론</h2>")
    out += [f"<p>{e(p)}</p>" for p in concl]
    for caption, rows in tabs:
        out.append(f"<h3>{e(caption)}</h3>")
        out.append("<table><thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
        for r in rows[1:]:
            out.append("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>")
        out.append("</tbody></table>")
    out.append("<h2>첨부</h2><p>악성 URL 탐지 최종 실험 보고서 v2.docx, 악성 URL 탐지 최종 실험 보고서 v2.pdf</p>")
    return "\n".join(out)


def to_markdown(intro, items, exps, method, summary, concl, tabs):
    out = [f"# {TITLE}", "", *[p + "\n" for p in intro]]
    out += ["## 실험 구성", "", *[f"- {b}" for b in items], "", "### 실험 목록", "", *[f"- {b}" for b in exps], "", results.EXPERIMENT_ORIGIN, ""]
    out += ["## 방법", "", *[f"- {b}" for b in method], ""]
    out += ["## 주요 결과", "", *[p + "\n" for p in summary], "## 결론", "", *[p + "\n" for p in concl]]
    for caption, rows in tabs:
        out += [f"### {caption}", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        out += ["| " + " | ".join(map(str, r)) + " |" for r in rows[1:]]
        out.append("")
    out += ["## 첨부", "", "악성 URL 탐지 최종 실험 보고서 v2.docx, 악성 URL 탐지 최종 실험 보고서 v2.pdf"]
    return "\n".join(out)


if __name__ == "__main__":
    ev = Evidence()
    parts = content(ev)
    (OUT / "notion-v2.html").write_text(to_html(*parts), encoding="utf8")
    (OUT / "notion-v2.md").write_text(to_markdown(*parts), encoding="utf8")
    print(OUT / "notion-v2.md")
