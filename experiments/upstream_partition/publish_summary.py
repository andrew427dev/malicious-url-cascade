"""Build the team's readable web summary from the same verified report tables."""

import argparse
from html import escape
from pathlib import Path

from report_content import CHAPTERS, POLICY, GROUP, read
from feature_table import example


def build(root):
    content = read(root / "report/content.json")
    tables = [
        (caption, heads, rows)
        for page in content
        for tag, *payload in page["blocks"]
        if tag == "table"
        for caption, heads, rows in [payload]
    ]
    body = []

    def p(s):
        body.append("<p>" + escape(s) + "</p>")

    def h(s):
        body.append("<h2>" + escape(s) + "</h2>")

    def sub(s):
        body.append("<h3>" + escape(s) + "</h3>")

    def table(heads, rows):
        def tr(row, cell):
            return (
                "<tr>"
                + "".join(f"<{cell}>{escape(str(c))}</{cell}>" for c in row)
                + "</tr>"
            )

        body.append(
            "<table>"
            + tr(heads, "th")
            + "".join(tr(r, "td") for r in rows)
            + "</table>"
        )

    def original_table(prefix):
        cap, heads, rows = next(t for t in tables if t[0].startswith(prefix))
        sub(cap)
        table(heads, rows)

    tm = read(root / "random-original/timing.json")["summary"]
    policy = min((n for n in tm if n != "gru"), key=lambda n: tm[n]["ms"])
    model_name = (
        POLICY[policy]
        .replace("LR", "로지스틱 회귀(LR)")
        .replace("HGB", "히스토그램 기반 그래디언트 부스팅(HGB)")
    )
    obs = read(root / "random-original/results.json")["policies"][policy]
    delta = tm["gru"]["ms"] - tm[policy]["ms"]
    p(
        f"이번에는 학습·검증·시험 집합 안에서 전처리한 전체 URL 문자열이 같을 때 첫 행만 남겼다. 원래 표기의 무작위 시험에서 가장 빠른 캐스케이드인 {model_name}의 1,000건 처리 중앙값은 {tm[policy]['ms']:.2f} ms였다. 같은 URL의 게이트 순환 유닛(GRU) 단독은 {tm['gru']['ms']:.2f} ms로, 캐스케이드에서 {abs(delta):.2f} ms {'줄었다' if delta >= 0 else '늘었다'}. 캐스케이드는 전체 시험 {obs['n']:,}건에서 악성을 {obs['additional_misses']:,}건 더 놓쳤다."
    )
    h(CHAPTERS[0])
    p(
        "모든 주소에 신경망을 실행할 때와, 정상 가능성이 큰 주소의 검사를 앞단에서 끝낼 때의 차이를 측정했다. 비교 단위는 같은 전체 시험에서 추가로 놓친 악성 수와 같은 1,000건 배치에서 실제 절약한 시간이다."
    )
    h(CHAPTERS[1])
    p(
        "원저자는 df_final.csv와 all_top_domains_merged.csv를 합쳐 섞은 뒤 층화 80/20으로 내부 검증했다. 문자 어휘는 학습에서만 만들고 df_another_validation.csv는 별도로 평가했다. 이후 세 파일 전체로 어휘와 GRU를 새로 학습했다. 당시 분할 ID와 난수 상태는 공개되지 않았다."
    )
    original_table("표 1.")
    original_table("원저자 절차에서 단계별로")
    h(CHAPTERS[2])
    p(
        "원저자에게 정제 자료를 요청한 뒤 기한 내 회신을 받지 못해 공개 CSV를 직접 정제했다. 원저자 재현은 원문 코드대로 처리했고, 팀 비교에서는 정확히 같은 원본 URL의 상충 라벨을 분할 전에 제외했다. 최종 토큰 충돌로 행을 추가 삭제하거나 재배정하지 않았다."
    )
    original_table("표 2.")
    original_table("표 3.")
    original_table("표 4.")
    p(example(root))
    hardware = read(root / "summary/checks.json")["hardware"]
    p(
        f"문자 입력은 소문자 변환과 접두어·공백·기호 제거를 거친 문자 3그램 50칸이다. CPU는 {hardware['cpu']}이며 연산 스레드 1개, 학습 배치 128, 시간 측정 배치 1,000건이다."
    )
    original_table("표 5.")
    h(CHAPTERS[3])
    p(
        "로지스틱 회귀(LR)는 계산이 가벼운 선형 분류기이고 히스토그램 기반 그래디언트 부스팅(HGB)은 수치 특징의 비선형 관계를 학습하는 트리 모델이다. 둘은 같은 1단계의 대안이다. 같은 분할·표기의 주 비교 7개 구성은 같은 GRU를 사용한다."
    )
    sub("전체 모델 구성도")
    body.append(
        '<p><img src="https://raw.githubusercontent.com/andrew427dev/malicious-url-cascade/main/docs/upstream-partition-20261124/cascade.png" alt="원저자 GRU와 팀의 1단계 분기 및 재검사 구성도"></p>'
    )
    p(
        "18개 수치 → LR 또는 HGB → 임계값 이상은 GRU, 미만은 정상 종료 후보. 후보 중 무작위 또는 경계에 가까운 5%를 GRU에 보낸다. GRU는 선택된 주소의 문자 3그램과 18개 수치를 함께 받아 최종 판정한다."
    )
    p(
        "5%는 조기 종료 후보 수를 분모로 내림한다. 공통 검증 집합에서 모델을 선택하고 확률 보정과 임계값 결정을 마쳤다. 네 조건을 고정한 뒤 본 시험을 수행했다. 시간은 14개 구성 예열 후 5회 순서를 섞어 원본 URL에서 판정까지 측정했다."
    )
    original_table("표 6.")
    h(CHAPTERS[4])
    for n in range(7, 18):
        original_table(f"표 {n}.")
    original_table("1단계 단독")
    original_table("표 18.")
    original_table("접두어 제거 후 학습 원문")
    original_table("표 19.")
    original_table("표 20.")
    original_table("표 21.")
    joint = read(root / "summary/common-test.json")
    p(
        "두 표기의 공통 시험 URL은 "
        + ", ".join(f"{GROUP[g]} {joint[g]['common_n']:,}건" for g in GROUP)
        + "이다. 고정 모델 실험은 입력 변화에 대한 반응을, 표기 통일 후 재학습은 그 표기에서 훈련된 분류기의 성능을 측정한다."
    )
    h(CHAPTERS[5])
    original_table("표 22.")
    p(
        "초기 준비 실행에서 보조 기준선의 시험 점수가 먼저 산출됐다. 최종 비교에는 경량 모델을 다시 학습하고 네 조건의 분기 기준을 검증 집합에서 모두 고정한 뒤 계산한 수치를 사용했다."
    )
    p(
        "미탐 허용 범위를 검증 자료에서 먼저 정하고 임계값을 낮추거나 재검사 비율을 높이면 신경망이 더 많은 주소를 검사한다. 각 조건의 실제 시간과 추가 탐지 수를 함께 확인해야 한다. 이번 측정은 단일 학습 시드와 한 CPU에서 처리한 배치에 해당한다. 원저자 별도 파일도 학습·검증 원문과 겹친다."
    )
    h(CHAPTERS[6])
    for tag, *args in content[-1]["blocks"]:
        if tag == "paragraph":
            p(args[0])
    sub("재실행과 자료 위치")
    p(
        "저장소 루트에서 python experiments/upstream_partition/run_study.py <새 출력 폴더>를 실행한다. 문서 생성과 검수 명령은 실험 코드의 README.md에 있다."
    )
    p(
        "출력 폴더의 summary/performance.csv는 전체 시험 성능, summary/timing.csv는 1,000건 처리 시간이다. 조건별 timing.json에서 측정한 순서, 각 회차의 시간, 표본 식별자를 확인할 수 있다. scores.npz와 *-rows.csv.gz에 행 단위 점수와 라벨이 들어 있다."
    )
    body.append(
        '<p><a href="https://github.com/andrew427dev/malicious-url-cascade/tree/main/evidence/upstream-partition-20261124">이번 실험의 결과표와 측정 원본</a> · <a href="https://github.com/andrew427dev/malicious-url-cascade/tree/main/experiments/upstream_partition">재실행 코드와 명령</a></p>'
    )
    sub("보고서 첨부")
    return "\n".join(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    (args.root / "report/notion-body.html").write_text(
        build(args.root), encoding="utf8"
    )
