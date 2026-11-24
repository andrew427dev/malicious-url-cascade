"""Report paragraphs and tables derived from the verified run."""

import json
from statistics import mean
from feature_table import FEATURE_ROWS, example

CHAPTERS = [
    "Introduction & Motivation",
    "Original Work Summary",
    "System Setup",
    "Experiments (highlighting your extension settings)",
    "Results & Discussion (highlighting your results)",
    "Limitations & Future Work",
    "Conclusion and References",
]
POLICY = {
    "gru": "GRU 단독",
    "logistic_basic": "LR 기본",
    "logistic_random": "LR 무작위",
    "logistic_boundary": "LR 경계",
    "boost_basic": "HGB 기본",
    "boost_random": "HGB 무작위",
    "boost_boundary": "HGB 경계",
}
GROUP = {"random": "무작위", "domain": "도메인"}
FORM = {
    "original": "원래 접두 표기를 유지한 조건",
    "stripped": "접두 표기 통일 후 재학습한 조건",
}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def number(x):
    return f"{x:,}"


def pct(x):
    return f"{100 * x:.2f}"


def build(root):
    output = []

    def page(title, chapter=False):
        output.append({"title": title, "chapter": chapter, "blocks": []})

    def add(kind, *values):
        output[-1]["blocks"].append((kind, *values))

    def para(s):
        add("paragraph", s)

    def table(title, headers, rows):
        add("table", title, headers, [[str(x) for x in row] for row in rows])

    def sub(s):
        add("sub", s)

    def data(kind, name):
        return read(root / kind / (name + ".json"))

    def fastest(kind, include_gru=False):
        t = data(kind, "timing")["summary"]
        n = min((k for k in t if include_gru or k != "gru"), key=lambda k: t[k]["ms"])
        p = data(kind, "results")["policies"][n]
        return (
            n,
            t["gru"]["ms"] - t[n]["ms"],
            t[n]["time_saved_pct"],
            p["additional_misses"],
        )

    def sentence(kind):
        key, ms, rate, miss = fastest(kind)
        return (
            f"{GROUP[kind.split('-')[0]]} 분할의 {FORM[kind.split('-')[1]]}에서 가장 빠른 캐스케이드는 "
            f"{POLICY[key]}였다. GRU 단독과 비교한 1,000건 처리 시간 중앙값은 "
            f"{abs(ms):.2f} ms({abs(rate):.2f}%) {'짧았고' if ms >= 0 else '길었고'}, "
            f"전체 시험에서 악성을 {number(miss)}건 더 놓쳤다."
        )

    check = data("summary", "checks")
    assert check["status"] == "PASS"
    team = read(root / "team-source.json")
    author = data("author-validation", "results")
    sources = team["sources"]
    page("원저자 전처리 기준 악성 URL 탐지 재실험")
    add("subtitle", "컴퓨터보안 팀프로젝트 실험 보고서")
    para(
        "모든 주소를 검사하는 게이트 순환 유닛(GRU) 앞에 경량 분류기를 놓았을 때, 추가로 놓치는 악성 URL과 실제로 절약하는 시간을 측정했다."
    )
    sub("목차")
    add("toc", CHAPTERS)
    page(CHAPTERS[0], True)
    para(
        "악성 URL 탐지기는 피싱이나 악성코드 배포를 의심할 주소를 찾아낸다. 이번 실험에서는 웹페이지에 접속하지 않고 주소 문자열만으로 판정했다. 모든 주소에 같은 신경망을 실행하면 정상일 가능성이 높은 주소에도 같은 계산을 반복한다. 가벼운 첫 검사를 두면 일부 계산을 생략할 수 있지만, 그 단계에서 정상으로 처리한 악성 주소는 뒤의 신경망이 바로잡을 수 없다."
    )
    sub("연구 질문")
    para(
        "같은 시험 URL과 같은 GRU 가중치에서, 경량 분류기를 앞에 두면 악성 주소를 몇 건 더 놓치고 1,000건을 처리하는 시간은 얼마나 달라지는가?"
    )
    para(
        "1단계는 로지스틱 회귀(Logistic Regression, LR)와 히스토그램 기반 그래디언트 부스팅(Histogram-based Gradient Boosting, HGB) 중 하나를 사용한다. LR은 수치 특징의 선형 결합을 한 번 계산하므로 첫 검사 비용이 낮다. HGB는 작은 결정 트리를 차례로 더해 길이와 기호 수 같은 특징 사이의 비선형 관계를 학습한다. 분기가 정교해질 때 얻는 호출 절감이 트리를 계산하는 비용을 상쇄하는지 비교했다."
    )
    sub("같은 시험 집합에서 관측한 차이")
    para(sentence("random-original"))
    para(sentence("domain-original"))
    para(
        "표기 실험은 두 상황을 다룬다. 배포한 모델에 들어오는 주소 표기만 달라질 때의 판정 변화와, 학습할 때부터 표기를 통일한 모델의 탐지 성능·속도는 서로 다른 질문이다."
    )
    page(CHAPTERS[1], True)
    para(
        "Munna Ram의 공개 노트북은 두 입력을 한 신경망에서 결합한다. 문자열을 세 글자씩 한 칸씩 옮겨 묶고 최대 50개 토큰을 사용한다. 각 토큰은 16차원 임베딩과 16유닛 GRU를 지난다. 수치 특징 18개는 유닛 16개의 ReLU 밀집층을 지난다. 두 출력을 결합한 뒤 유닛 16개의 ReLU 밀집층과 sigmoid 출력 한 개로 정상 확률을 계산한다. GRU의 입력·순환 dropout과 각 분기 및 결합 뒤 dropout은 0.2다. 학습 매개변수는 162,481개다."
    )
    table(
        "표 1. 공개 CSV별 원본 행과 노트북에서 맡은 역할",
        ["파일", "정상", "악성", "첫 학습 블록"],
        [
            [n, number(sources[n]["benign"]), number(sources[n]["malicious"]), r]
            for n, r in zip(
                sources, ("학습·내부 검증 풀", "학습·내부 검증 풀", "별도 평가")
            )
        ],
    )
    para(
        "원저자는 앞의 두 파일을 결합해 섞은 뒤 전체 열 중복과 결측을 제거했다. 정상·악성 비율을 유지하며 80 대 20으로 나누고, 각 집합 안에서 문자 전처리 결과의 첫 행을 남겼다. 문자 어휘는 학습 집합에서만 맞췄다. 세 번째 파일은 이 학습 블록에서 버퍼 10,000행으로 순서를 섞고 128행씩 예측해 0.5 기준으로 별도 평가했다. 정상 확률이 정확히 0.5이면 정상이다."
    )
    fin = data("author-final", "data")["stages"]["train"]
    para(
        f"이후 노트북은 세 파일 전체를 합쳐 섞고 중복·결측을 제거한 뒤 새 어휘와 새 모델을 학습한다. 이번 최종 학습에는 분할 내부 중복 제거 후 {number(fin['after']['rows'])}행이 들어갔다. 이 단계에는 별도 시험 집합이 없다. 공개 셀에는 20회 학습 명령이 있고 보존된 출력은 19회차 도중에서 끝난다. 이번 실행의 20회 완료는 새 학습 이력에서 확인했다."
    )
    demos = read(root / "author-examples.json")
    matched = sum(
        a == b
        for d in demos
        for a, b in zip(d["published_predictions"], d["new_predictions"])
    )
    n_examples = sum(len(d["new_predictions"]) for d in demos)
    para(
        f"노트북의 학습 후 예시와 다시 불러온 모델의 예시까지 {n_examples}회 개별 판정했다. 이 중 {matched}회가 공개 노트북의 판정과 일치했다. 예시에는 독립 정답이 주어지지 않아 정확도 계산에는 사용하지 않았다."
    )
    sub("원문의 지표를 읽는 방법")
    para(
        "원저자 README는 내부 정확도를 98.4%로 요약했고, 공개 노트북의 20회차 로그는 98.34%다. 별도 파일의 71.11%는 별도 평가 정확도다. 당시 난수 시드, 최초 셔플과 분할 행 목록, 가중치 초기값이 공개되지 않았다. 이번 실행은 시드 42로 절차를 재현했다. 결과 표는 각 수치를 같은 역할의 집합에서 같은 지표와 비교한다."
    )
    page("원저자 학습 블록별 정제 경로")
    source_rows = []
    for kind, label in (
        ("author-validation", "첫 학습"),
        ("author-final", "최종 학습"),
    ):
        stages = data(kind, "source-stages")
        previous = stages["shuffled"]["rows"]
        for step, text in (
            ("shuffled", "결합·셔플"),
            ("exact_row_unique", "전체 열 중복 제거"),
            ("nonmissing", "결측 제거"),
        ):
            count = stages[step]["rows"]
            source_rows.append(
                [label, text, number(previous), number(count), number(previous - count)]
            )
            previous = count
        for role, step in data(kind, "data")["stages"].items():
            name = {
                "train": "학습",
                "validation": "내부 검증",
                "external": "별도 평가",
            }[role]
            source_rows.append(
                [
                    label,
                    name + " 문자열 중복 제거",
                    number(step["before"]["rows"]),
                    number(step["after"]["rows"]),
                    number(step["removed"]),
                ]
            )
    table(
        "원저자 절차에서 단계별로 남긴 행",
        ["학습 블록", "처리 또는 집합", "입력 행", "남은 행", "제외 행"],
        source_rows,
    )
    conf1 = data("author-validation", "source-stages")["conflicting_raw_urls"]
    conf2 = data("author-final", "source-stages")["conflicting_raw_urls"]
    para(
        f"첫 학습 풀에서 라벨이 상충한 원본 URL은 {number(conf1)}개, 세 파일 전체에서는 {number(conf2)}개였다. 원저자 코드는 전체 열이 같은 행을 먼저 제거하므로 같은 원문에 서로 다른 라벨이 붙은 행은 이 단계에 남는다. 각 집합 안의 문자열 중복 제거에서 어느 행이 먼저 오는지에 따라 남는 라벨도 달라질 수 있다. 별도 평가 행은 첫 학습 풀의 분할과 독립적으로 세 번째 파일에서 읽었다."
    )
    page(CHAPTERS[2], True)
    para(
        "처음에는 원저자가 공개한 세 파일을 그대로 사용하려 했다. 파일을 합쳐 보니 같은 원본 주소에 서로 다른 라벨이 붙은 경우와 파일 사이에 겹치는 주소가 있었다. 원저자에게 정제 자료를 요청했으나 회신을 받지 못해 프로젝트 기한 안에 공개 자료를 직접 정제했다."
    )
    table(
        "표 2. 팀 비교의 분할 전 처리",
        ["단계", "남은 행", "이번 단계 제외"],
        [
            ["세 공개 CSV 결합", number(team["combined"]["rows"]), "—"],
            [
                "결측 제거",
                number(team["nonmissing"]["rows"]),
                number(team["combined"]["rows"] - team["nonmissing"]["rows"]),
            ],
            [
                f"원본 URL {number(team['exact_conflicting_urls'])}개의 라벨 충돌 제외",
                number(team["after_conflicts"]["rows"]),
                number(team["conflict_rows"]["rows"]),
            ],
            [
                "동일 원본 URL·라벨의 첫 행 유지",
                number(team["after_exact_duplicates"]["rows"]),
                number(
                    team["after_conflicts"]["rows"]
                    - team["after_exact_duplicates"]["rows"]
                ),
            ],
        ],
    )
    para(
        "원저자 재현에서는 상충 라벨 전체를 따로 제외하지 않았다. 팀 비교에서는 정답을 결정할 수 없는 정확히 같은 원문만 제외했다. 원문이 다른 두 주소가 문자 전처리 후 같아져도 라벨 오류로 분류하지 않았다. 학습·검증·시험에 배정한 뒤, 각 집합 안에서만 전처리한 전체 문자열이 같은 행 중 첫 행을 남겼다."
    )
    table(
        "표 3. 원래 표기의 집합별 최종 행 수와 전처리 중복 제외 수",
        ["분할", "역할", "분할 직후", "최종", "제외"],
        [
            [
                GROUP[g],
                r,
                number(s["before"]["rows"]),
                number(s["after"]["rows"]),
                number(s["removed"]),
            ]
            for g in GROUP
            for r, s in zip(
                ("학습", "검증", "시험"),
                data(g + "-original", "data")["stages"].values(),
            )
        ],
    )
    para(
        "무작위 분할은 라벨을 층화했다. 팀이 추가한 도메인 분할은 등록 도메인 단위로 나누므로 실제 행 비율이 목표 70/15/15와 다르다. 로컬 Public Suffix List의 사설 접미사도 반영했고, 등록 접미사가 없으면 호스트를 사용했다. 호스트를 읽을 수 없는 3행은 같은 그룹으로 배정했다. 모델 입력은 원문을 그대로 사용했다."
    )
    page("수치 특징 18개의 열 순서와 계산 방법")
    para(
        "원저자의 FeatureCreation.create_features가 열을 붙이는 순서다. 원래 표기 조건에서는 문자 전처리 전 원문에 계산하며, 표기 통일 조건에서는 맨 앞 접두어만 제거한 뒤 같은 함수를 적용한다."
    )
    table(
        "표 4. 원저자 수치 입력의 열 순서",
        ["순서", "원저자 열 이름", "계산 방법"],
        [[i, n, d] for i, (n, d) in enumerate(FEATURE_ROWS, 1)],
    )
    para(example(root))
    page("문자 입력과 실행 환경")
    para(
        "문자열을 소문자로 바꾸고 http://, https://, www., 공백 한 칸을 순서대로 치환해 지운다. 양끝 공백을 정리하고 영문자와 숫자 이외 문자를 제거한다. 원저자의 치환은 문자열 전체에 적용된다. 표기 실험에서 맨 앞의 HTTP(S) 접두어만 지우는 조작과 구별된다."
    )
    para(
        "example.org/a-b와 example.org/ab는 둘 다 exampleorgab가 된다. 수치 특징이 달라도 한 집합에 있으면 먼저 들어온 행을 남긴다. 반대로 전처리 전체 문자열이 다르면 50개 토큰을 채운 결과가 같아도 둘 다 남긴다. 50은 원본 글자 수가 아니라 연속 3글자 토큰의 입력 칸 수다. 짧으면 0으로 채우며 어휘 밖 토큰은 미등록 토큰으로 표시한다."
    )
    parity = read(root / "parity.json")
    first, reverse = parity["orders"]
    para(
        f"공개 파일에서 뽑은 {parity['real_sample_rows']}행에 하이픈·접두어·긴 주소와 반복 행을 더해 정순과 역순으로 검사했다. {first['input']}행 중 정순은 {first['kept']}행, 역순은 {reverse['kept']}행을 남겼다. 보존 행 ID·라벨과 같은 어휘에서의 50개 토큰·18개 수치가 원저자의 실제 코드와 일치했다. 팀의 네 학습 조건에서는 모든 역할의 행·라벨·수치·토큰과 학습 어휘를 원본에서 다시 계산해 저장본과 대조했다."
    )
    tab = []
    for g in GROUP:
        d = data(g + "-original", "data")["overlaps"]
        for pair, z in d.items():
            tab.append(
                [
                    GROUP[g],
                    {
                        "train/validation": "학습·검증",
                        "train/test": "학습·시험",
                        "validation/test": "검증·시험",
                    }[pair],
                    number(z["url"]),
                    number(z["text"]),
                    number(z["domain"]),
                ]
            )
    table(
        "표 5. 두 집합에 공통인 고유 값 수",
        ["분할", "집합 쌍", "원본 URL", "전처리 문자열", "도메인 그룹"],
        tab,
    )
    pk = check["packages"]
    para(
        f"장비는 {check['hardware']['cpu']}, 운영체제는 {check['hardware']['operating_system']}이다. Python {check['python']}, TensorFlow {pk['tensorflow']}, Keras {pk['keras']}, scikit-learn {pk['scikit-learn']}, NumPy {pk['numpy']}, pandas {pk['pandas']}, tldextract {pk['tldextract']}를 사용했다. TensorFlow 연산 내부·연산 간 스레드 및 BLAS·OpenMP 스레드는 각각 1개다. 학습은 시드 42, 배치 128, Adam, 이진 교차엔트로피, 20회다. 원저자는 TPU 복제본 8개에서 배치 128을 사용했다."
    )
    page(CHAPTERS[3], True)
    add("figure", "그림 1. 원저자 신경망의 두 입력과 팀이 추가한 선택적 검사")
    para(
        "GRU 단독은 모든 URL의 두 입력을 계산한다. 캐스케이드는 모든 URL에서 수치 특징을 한 번 계산하고, 1단계에서 끝나지 않은 URL에만 문자 토큰을 만들어 GRU를 실행한다. LR과 HGB는 각각 별도 구성에서 같은 1단계 역할을 맡는다."
    )
    para(
        "무작위 재검사와 경계 재검사는 1단계가 정상으로 조기 종료할 후보 수의 5%를 내림한 만큼 검사한다. 경계 방식은 그중 악성 확률이 높은 순서로 고른다. 무작위 방식은 시드와 원본 행 ID로 정해진 순위를 쓴다. 같은 시험 집합에서 두 방식의 실제 재검사 수는 같다."
    )
    page("모델 선택과 실제 시간 측정")
    para(
        "LR은 표준화 후 규제 강도의 역수 C를 0.1, 1, 10에서 선택했다. HGB는 학습률 0.1, 트리 60개를 사용하고 최대 잎 수 7, 15, 31을 비교했다. 두 후보군 모두 검증 집합의 악성 평균 정밀도(AP)로 선택하고 sigmoid로 확률을 보정했다. AP는 판정 임계값을 바꿔 가며 재현율 증가분에 따라 정밀도를 가중한 지표다."
    )
    para(
        "팀 GRU는 20회 학습 중 검증 손실이 가장 작은 회차를 선택했다. 분기 임계값은 후보 12개 중 검증 악성 재현율 손실이 GRU 단독보다 1%p 이내인 값으로 제한하고, GRU 호출이 가장 적은 값을 골랐다. 동률이면 작은 임계값을 택했다. 모델·보정기·분기 기준은 각 조건의 동일한 검증 집합에서 결정했다."
    )
    rows = []
    for g in GROUP:
        k = g + "-original"
        cfg = data(k, "routing")["families"]
        st = data(k, "stage1/t1-selected")["selection"]
        rows.append(
            [
                GROUP[g],
                "원래",
                data(k, "trained")["epoch"],
                st["logistic"]["winner"],
                st["boost"]["winner"],
                cfg["logistic"]["tau"],
                cfg["boost"]["tau"],
            ]
        )
    table(
        "표 6. 검증 집합에서 선택한 설정",
        ["분할", "표기", "GRU 회차", "LR C", "HGB 잎", "LR 임계값", "HGB 임계값"],
        rows,
    )
    para(
        "정확도는 전체 정답 비율, 악성 재현율은 실제 악성 중 탐지 비율, 정상 오탐률은 실제 정상 중 악성으로 판정한 비율이다. 추가 미탐은 같은 GRU가 잡았으나 1단계를 더하면서 놓친 악성 수다. GRU 호출 건수는 GRU에 넘긴 URL 수로 센다."
    )
    para(
        "시간은 분할별 공통 시험 ID에서 고정한 원본 URL 1,000건으로 쟀다. 원문 표기 처리와 수치 특징 계산을 시작한 시점부터 선택적 문자 인코딩, GRU 추론, 최종 판정 배열 완성까지 포함했다. 모델 적재, 표본 선택, 결과 대조는 측정 밖이다. 각 분할의 두 표기 × 7개 구성 모두 1회 예열한 뒤, 14개 구성의 순서를 회차마다 함께 섞어 5회 측정했다. 한 배치를 함께 처리한 중앙값과 최솟값–최댓값을 ms로 제시한다."
    )
    page(CHAPTERS[4], True)
    table(
        "표 7. 원저자 발표와 이번 재현의 같은 역할·같은 지표 비교",
        ["평가 집합", "지표", "원저자 (%)", "이번 재현 (%)"],
        [
            [
                "내부 검증",
                "20회차 정확도",
                "98.34",
                pct(
                    data("author-validation", "trained")["history"]["val_accuracy"][-1]
                ),
            ],
            ["별도 평가", "정확도", "71.11", pct(author["external"]["accuracy"])],
            [
                "별도 평가",
                "가중 정밀도",
                "64.62",
                pct(author["external"]["weighted_precision"]),
            ],
            [
                "별도 평가",
                "가중 재현율",
                "71.11",
                pct(author["external"]["weighted_recall"]),
            ],
            ["별도 평가", "가중 F1", "67.35", pct(author["external"]["weighted_f1"])],
        ],
    )
    ext = author["external"]
    val = author["validation"]
    para(
        f"이번 20회차 내부 검증은 {number(val['n'])}건이다. 별도 파일은 자체 전처리 후 {number(ext['n'])}건으로, 정상 {number(ext['tn'] + ext['fp'])}건과 악성 {number(ext['tp'] + ext['fn'])}건이 남았다. 별도 평가 악성 재현율은 {pct(ext['recall'])}%, 정상 오탐률은 {pct(ext['fpr'])}%였다. 가중 재현율은 정상과 악성을 각 클래스 크기로 가중하므로 악성 재현율과 분모가 다르다."
    )
    d = data("author-validation", "data")
    para(
        f"이 별도 파일에서 학습 집합과 원문이 겹치는 고유 URL은 {number(d['overlaps']['train/external']['url'])}개, 내부 검증과는 {number(d['overlaps']['validation/external']['url'])}개였다. 따라서 별도 파일 전체의 결과를 처음 수집한 외부 주소에 대한 일반화 성능으로 해석하기 어렵다."
    )
    ex = author["external_without_exact_training_or_validation_url"]
    para(
        f"추가로 이번 학습·검증의 최종 행과 원문이 같은 주소를 제외하면 {number(ex['n'])}건(정상 {number(ex['benign'])}, 악성 {number(ex['malicious'])})이다. 이 부분집합의 악성 재현율은 {pct(ex['malicious_recall'])}%다. "
        + (
            "정상 분모가 0이어서 오탐률은 정의되지 않는다."
            if ex["fpr"] is None
            else f"정상 오탐률은 {pct(ex['fpr'])}%다."
        )
    )
    para(
        "팀 주 실험은 별도 파일까지 합쳐 새로 분할했다. 아래의 추가 미탐과 호출 차이는 각 팀 시험 집합 안에서 계산한다. 원저자와 팀 주 실험 사이에는 시험 대상과 학습 자료가 모두 다르다."
    )
    for g in GROUP:
        page(f"{GROUP[g]} 분할의 탐지 성능")
        k = g + "-original"
        rows = data(k, "results")["policies"]
        gr = rows["gru"]
        para(
            f"시험 {number(gr['n'])}건 중 악성 {number(gr['tp'] + gr['fn'])}건, 정상 {number(gr['tn'] + gr['fp'])}건이다."
        )
        table(
            "표 "
            + str(8 + list(GROUP).index(g))
            + ". "
            + GROUP[g]
            + " 분할, "
            + FORM["original"],
            [
                "구성",
                "정확도 %",
                "재현율 %",
                "오탐률 %",
                "미탐",
                "추가 미탐",
                "GRU 건",
                "호출 %",
            ],
            [
                [
                    POLICY[n],
                    pct(r["accuracy"]),
                    pct(r["recall"]),
                    pct(r["fpr"]),
                    number(r["fn"]),
                    number(r["additional_misses"]),
                    number(r["t2_calls"]),
                    pct(r["t2_call_rate"]),
                ]
                for n, r in rows.items()
            ],
        )
        examples = check["examples_by_source_id"][k]["boost"]
        if examples:
            para(
                "HGB 경계 재검사에서 조기 종료한 악성 주소의 원본 행 ID는 "
                + ", ".join(examples)
                + "이다. 같은 GRU를 단독으로 실행하면 이 주소들은 악성으로 탐지됐다."
            )
    for g in GROUP:
        page(f"{GROUP[g]} 분할의 실제 처리 시간")
        k = g + "-original"
        tm = data(k, "timing")["summary"]
        table(
            "표 "
            + str(10 + list(GROUP).index(g))
            + ". "
            + GROUP[g]
            + " 분할, "
            + FORM["original"]
            + " · 동일 1,000건",
            [
                "구성",
                "GRU 건",
                "호출 절감 %",
                "중앙값 ms",
                "5회 범위 ms",
                "시간 단축 %",
            ],
            [
                [
                    POLICY[n],
                    number(r["calls"]),
                    f"{r['calls_saved_pct']:.1f}",
                    f"{r['ms']:.2f}",
                    f"{r['min_ms']:.2f}–{r['max_ms']:.2f}",
                    f"{r['time_saved_pct']:.2f}",
                ]
                for n, r in tm.items()
            ],
        )
        para(
            "시간 단축률은 (GRU 단독 시간 − 해당 구성 시간) / GRU 단독 시간이다. 음수는 시간이 늘었다는 뜻이다. 표의 호출 절감률은 이 1,000건에서 계산했다. 전체 시험의 호출률과 분모가 다르다."
        )
    page("경량 분류기만으로 판정한 결과")
    rows = []
    for g in GROUP:
        res = data(g + "-original", "results")
        for fam, m in res["first_stage"].items():
            rows.append(
                [
                    GROUP[g],
                    "원래",
                    "LR" if fam == "logistic" else "HGB",
                    pct(m["accuracy"]),
                    pct(m["recall"]),
                    pct(m["fpr"]),
                    number(m["fn"]),
                    pct(m["pr_auc_ap"]),
                ]
            )
    table(
        "표 12. 1단계 단독 · 보정한 악성 확률 0.5를 기준으로 판정",
        ["분할", "표기", "모델", "정확도 %", "재현율 %", "오탐률 %", "미탐 건", "AP %"],
        rows,
    )
    para(
        "단독 모델은 시험 URL 전체를 수치 특징만으로 판정해 GRU 호출이 0건이다. 캐스케이드는 분기 임계값 이상인 주소와 재검사에 선택된 주소를 GRU로 보내므로, 1단계 단독의 악성 미탐 수와 캐스케이드의 추가 미탐 수는 의미가 다르다."
    )
    page("수치 특징 제거와 학습량에 따른 변화")
    rlist = []
    for g in GROUP:
        orig = data(g + "-original", "results")["first_stage"]
        controls = data(g + "-original", "controls-final")
        for fam in ("logistic", "boost"):
            rlist.append(
                [
                    GROUP[g],
                    "LR" if fam == "logistic" else "HGB",
                    pct(orig[fam]["pr_auc_ap"]),
                    pct(controls[fam + "_omit_columns"]["pr_auc_ap"]),
                ]
            )
    table(
        "표 13. 표기 관련 특징을 제외한 1단계 모델의 평균 정밀도",
        ["분할", "모델", "18개 특징 AP %", "9개 특징 AP %"],
        rlist,
    )
    para(
        "원래 표기에서 abnormal_url, 경로의 /·// 횟수, https·http 횟수, 원문 길이, netloc 길이, 영문을 포함한 문자 수, 첫 디렉터리 길이 등 9개 열을 제외하고 다시 학습했다."
    )
    c = data("random-original", "controls-final")
    para(
        f"원래 표기 무작위 시험에서 URL 길이 하나로 학습한 모델의 AP는 {pct(c['length']['pr_auc_ap'])}%, 접두 표기 유무만으로 악성을 판정한 단순 규칙의 정확도는 {pct(c['scheme_only']['accuracy'])}%였다. 이는 공개 데이터 안에서 형식과 라벨이 함께 변하는 정도를 보여 준다. 수집 출처가 원인인지는 이 결과만으로 알 수 없다."
    )
    table(
        "표 14. 원래 표기 무작위 분할에서 학습량에 따른 검증 AP (%)",
        ["학습 비율", "학습 행", "LR", "HGB"],
        [
            [
                f"{a['fraction']:.0%}",
                number(a["n"]),
                pct(a["validation_ap"]),
                pct(b["validation_ap"]),
            ]
            for a, b in zip(c["logistic_curve"], c["boost_curve"])
        ],
    )
    page("재검사 예산과 임계값 민감도")
    rows = []
    for g in GROUP:
        k = g + "-original"
        result = data(k, "results")["policies"]
        sw = data(k, "sensitivity")
        for fam in ("logistic", "boost"):
            a = result[fam + "_boundary"]
            seeds = sw[fam]["random_seeds"]
            vals = [r["recovered_malicious"] for r in seeds]
            rows.append(
                [
                    GROUP[g],
                    "원래",
                    "LR" if fam == "logistic" else "HGB",
                    number(a["rechecks"]),
                    number(a["recovered_malicious"]),
                    f"{mean(vals):.1f}",
                    f"{min(vals)}–{max(vals)}",
                ]
            )
    table(
        "표 15. 기본 분기에서 놓쳤으나 재검사로 되찾은 악성 URL",
        ["분할", "표기", "모델", "재검사 건", "경계 회복", "무작위 평균", "30회 범위"],
        rows,
    )
    para(
        "되찾은 건수는 기본 분기라면 정상으로 끝났지만 재검사에 뽑혀 GRU가 악성으로 판정한 주소 수다. 경계 방식 한 번과 같은 예산의 무작위 방식을 시드 0~29에서 비교했다."
    )
    k = "random-original"
    fam = "boost"
    tau = data(k, "routing")["families"][fam]["tau"]
    grid = [
        r
        for r in data(k, "sensitivity")[fam]["sensitivity"]
        if r["tau"] == tau and r["policy"] in ("basic", "boundary")
    ]
    table(
        "표 16. 원래 표기 무작위 분할 HGB의 재검사 비율별 변화",
        ["재검사 비율", "재검사 건", "악성 재현율 %", "추가 미탐", "GRU 호출 %"],
        [
            [
                f"{r['rate']:.0%}",
                number(r["rechecks"]),
                pct(r["recall"]),
                number(r["additional_misses"]),
                pct(r["t2_call_rate"]),
            ]
            for r in grid
        ],
    )
    para(
        "12개 분기 임계값과 5개 재검사 비율의 전 조합은 조건별 sensitivity.json에 저장했다. 시험 민감도는 정책의 동작을 설명하며, 본 비교의 임계값은 앞서 검증 집합에서 정한 값이다."
    )
    page("추가 실험 · 접두어를 제거한 결과")
    counts = []
    to_benign = []
    retrained = []
    paired_counts = []
    all_common = data("summary", "common-test")
    for g in GROUP:
        orig = g + "-original"
        stripped = g + "-stripped"
        pairs = data(orig, "fixed-prefix")["pairs"]
        to_benign.append(
            f"{GROUP[g]} {number(pairs['with_prefix']['flips']['gru']['malicious_to_benign'])}건"
        )
        for key, label in (("with_prefix", "있음"), ("without_prefix", "없음")):
            group = pairs[key]
            counts.append(
                [
                    GROUP[g] + " · " + label,
                    number(group["rows"]),
                    number(group["token_changes"]),
                    number(group["numeric_changes"]),
                    number(group["flips"]["gru"]["changed"]),
                ]
            )
        joint = all_common[g]
        policy = fastest(stripped)[0]
        for name in ("gru", policy):
            metrics = [joint[form][name] for form in ("original", "stripped")]
            times = [
                data(kind, "timing")["summary"][name]["ms"] for kind in (orig, stripped)
            ]
            misses = [
                metric["fn"] - joint[form]["gru"]["fn"]
                for metric, form in zip(metrics, ("original", "stripped"))
            ]
            retrained.append(
                [
                    GROUP[g],
                    POLICY[name],
                    " → ".join(pct(m["recall"]) + "%" for m in metrics),
                    " → ".join(f"{v:.2f}" for v in times),
                    " → ".join(number(v) for v in misses),
                ]
            )
        paired_counts.append(f"{GROUP[g]} {number(joint['common_n'])}건")
    para(
        "모델·어휘·보정기·임계값을 고정한 채 같은 시험 행에서 맨 앞의 http:// 또는 https://만 뗐다. 원래 순서대로 두 입력을 다시 계산했다. 표의 판정 변화는 GRU 단독에서 정상과 악성이 뒤집힌 건수다."
    )
    table(
        "표 17. 고정한 GRU에 접두 표기만 바꿔 입력한 결과",
        ["분할 · 원래 접두어", "시험 URL", "토큰 변화", "수치 변화", "판정 변화"],
        counts,
    )
    para(
        "접두어를 떼자 GRU가 악성에서 정상으로 판정을 바꾼 주소는 "
        + ", ".join(to_benign)
        + "이었다. 7개 구성의 양방향 판정 변화는 분할별 fixed-prefix.json에 저장했다."
    )
    para(
        "다음은 학습·검증·시험에서 처음부터 접두어를 떼고 모델을 새로 학습한 결과다. 화살표는 원래 표기 → 접두어 제거를 뜻한다. 분할별로 GRU 단독과 재학습 조건에서 가장 빨랐던 캐스케이드를 비교했다."
    )
    table(
        "표 18. 접두어 제거 후 재학습 · 같은 구성의 두 학습 조건 비교",
        ["분할", "구성", "악성 재현율", "시간 (ms)", "추가 미탐"],
        retrained,
    )
    identical = all(all_common[g]["ordered_ids_equal"] for g in GROUP)
    para(
        "두 표기의 공통 시험 URL은 "
        + ", ".join(paired_counts)
        + "이다. "
        + (
            "두 분할 모두 전체 시험 행·라벨·순서가 일치했다. "
            if identical
            else "탐지 성능은 공통 행에서 비교했다. "
        )
        + "시간은 같은 1,000건의 5회 중앙값이고 추가 미탐은 각 표기의 GRU 단독 대비 건수다. 저장소의 evidence/upstream-partition-20261124 아래 summary/performance.csv와 summary/timing.csv에 7개 구성의 전체 성능과 시간이 있다. 재학습 모델의 행별 점수와 라벨은 random-stripped와 domain-stripped 폴더의 scores.npz와 test-rows.csv.gz에 보관했다."
    )
    page(CHAPTERS[5], True)
    table(
        "표 19. 원래 표기에서 분할별로 가장 빠른 구성과 GRU 단독 대비 손익",
        ["분할", "표기", "구성", "절약 ms", "시간 단축 %", "추가 미탐 건"],
        [
            [GROUP[g], "원래", POLICY[n], f"{ms:.2f}", f"{r:.2f}", number(m)]
            for g in GROUP
            for n, ms, r, m in [fastest(g + "-original", include_gru=True)]
        ],
    )
    para(
        "절약 시간은 고정한 1,000건 배치의 차이이고 추가 미탐은 해당 분할의 전체 시험 결과다. 예열 후 5회 측정한 한 장비의 결과이므로 중앙값이 비슷하고 범위가 겹치는 구성의 속도 순위는 다른 부하나 배치 크기에서 달라질 수 있다."
    )
    slow = []
    for g in GROUP:
        t = data(g + "-original", "timing")["summary"]
        slow.extend(
            [
                f"{GROUP[g]}·원래 {POLICY[n]} ({r['time_saved_pct']:.2f}%)"
                for n, r in t.items()
                if r["calls"] < 1000 and r["time_saved_pct"] < 0
            ]
        )
    para(
        ("호출은 줄었으나 전체 시간이 늘어난 구성은 " + ", ".join(slow) + "였다.")
        if slow
        else "이번 네 조건에서 호출을 줄이고도 중앙값이 길어진 구성은 없었다."
    )
    para(
        "검증 때의 허용 손실 1%p는 시험 집합에서 보장되는 상한이 아니다. 운영에 필요한 미탐 상한을 먼저 정한 뒤, 별도 검증 자료에서 경계 재검사 비율을 올리거나 분기 임계값을 낮추면 더 많은 후보가 GRU로 넘어간다. 이때 회복한 악성 수와 늘어난 배치 시간을 함께 재야 한다."
    )
    selected_tau = data("random-original", "routing")["families"]["boost"]["tau"]
    sensitivity = data("random-original", "sensitivity")["boost"]["sensitivity"]
    by_rate = {
        r["rate"]: r
        for r in sensitivity
        if r["tau"] == selected_tau and r["policy"] == "boundary"
    }
    five, ten = by_rate[0.05], by_rate[0.10]
    recovered = five["additional_misses"] - ten["additional_misses"]
    more_calls = ten["t2_calls"] - five["t2_calls"]
    para(
        f"원래 표기의 무작위 분할에서 HGB의 검증 임계값 {selected_tau:g}을 유지하고 경계 재검사를 5%에서 10%로 높이면, 전체 시험에서 GRU 호출이 {number(more_calls)}건 늘고 추가 미탐은 {number(five['additional_misses'])}건에서 {number(ten['additional_misses'])}건으로 바뀌었다. 이때 되찾은 악성은 {number(recovered)}건이다. 10% 조건의 처리 시간은 별도로 측정해야 한다."
    )
    para(
        "초기 준비 실행에서 보조 기준선 시험 점수가 분기 기준 확정 전에 산출됐다. 최종 비교에서는 경량 분류기를 새로 적합하고 네 조건의 분기 기준을 검증 자료로 모두 고정한 뒤 시험 점수를 다시 계산했다. 같은 공개 자료에 대한 과거 시험 노출이 있으며, 후보 선택·보정·분기 기준에도 같은 검증 자료를 썼다. 모든 라벨의 실제 악성 여부를 독립적으로 확인하지 않았고, 각 모델의 학습 시드는 하나다. 기간과 수집 출처가 다른 데이터, 다양한 배치 크기와 단건 요청에서 다시 측정할 필요가 있다."
    )
    page(CHAPTERS[6], True)
    sub("이번 재실험의 결론")
    para(sentence("random-original"))
    para(sentence("domain-original"))
    para(
        "원래 표기에서는 무작위 분할과 등록 도메인 분할의 가장 빠른 구성이 달랐다. "
        "처리 시간을 줄인 구성도 악성 주소를 추가로 놓쳤다. 적용 환경에서 허용할 미탐 수준을 정하고, "
        "그 범위에서 분기 기준과 재검사 비율을 선택할 필요가 있다."
    )
    sub("참고문헌")
    para(
        "[1] Munna Ram. URL_CLASSIFIER. 공개 Git 저장소. 기준 커밋 835bd5e1710531407c0b5ca09da3d7c3de8d1bbc. https://github.com/munna0912/URL_CLASSIFIER"
    )
    para(
        "[2] Paul Viola and Michael Jones. Rapid Object Detection Using a Boosted Cascade of Simple Features. CVPR, 2001, vol. 1, pp. 511–518. DOI 10.1109/CVPR.2001.990517. 쉬운 입력을 앞 단계에서 처리하고 나머지에 더 복잡한 검사를 적용하는 구조를 참고했다."
    )
    return output
