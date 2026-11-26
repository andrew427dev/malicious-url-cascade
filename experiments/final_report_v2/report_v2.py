"""Build '악성 URL 탐지 최종 실험 보고서 v2' (docx) from the v1 frozen run and the v2 experiments.

Run with the report environment (python-docx; see README.md):  python -B report_v2.py
Then convert with experiments/upstream_partition/render_word.ps1 (Word COM) to PDF.
"""
import re
from datetime import datetime, timedelta, timezone

from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from report_base import (V1, OUT, TEMPLATE_DIR, TEAM, BASIC, SPLITS, SPLIT_NAME, FAM, NAMES, CASCADES,
                         Evidence, ReportTable, FEATURE_ROWS, example, table, field, bookmark, readj,
                         num, pct, sgn, f2, ci, figure_thresholds, figure_rates)
from diagram_v2 import draw
import report_results as results

TITLE = "악성 URL 탐지 최종 실험 보고서 v2"
SUBTITLE = "팀 정제 데이터에서 선택적 GRU 검사의 탐지 성능과 처리 시간"
HEAD = ["Introduction & Motivation", "Original Work Summary", "System Setup",
        "Experiments (highlighting your extension settings)", "Results & Discussion (highlighting your results)",
        "Limitations & Future Work", "Conclusion and References", "부록"]
URL_LENGTH = "특징 추출에 넣은 URL 문자열(원문)의 글자 수. 기본 정제는 원본 URL, 팀 정제는 정규화한 URL이다."
assert dict(FEATURE_ROWS)["url_length"] == "원본 URL 문자열의 글자 수."   # the shared row text this override replaces

REFS = [
    "Sahoo, D., Liu, C., Hoi, S. C. H. (2017). Malicious URL Detection using Machine Learning: A Survey. https://arxiv.org/abs/1701.07179",
    "Ma, J., Saul, L. K., Savage, S., Voelker, G. M. (2009). Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs. KDD. https://cseweb.ucsd.edu/~savage/papers/KDD09.pdf",
    "Munna Ram. URL_CLASSIFIER, Model_Training.ipynb 및 Utils, 커밋 835bd5e17105. https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc",
    "Le, H., Pham, Q., Sahoo, D., Hoi, S. C. H. (2018). URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection. https://arxiv.org/abs/1802.03162",
    "Northcutt, C. G., Athalye, A., Mueller, J. (2021). Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. https://arxiv.org/abs/2103.14749",
    "Arp, D., et al. (2022). Dos and Don’ts of Machine Learning in Computer Security. USENIX Security. https://www.usenix.org/system/files/sec22-arp.pdf",
    "Viola, P., Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. CVPR. https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf",
    "Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M. (2017). NoScope: Optimizing Neural Network Queries over Video at Scale. PVLDB 10(11), §§6.2–6.3. https://www.vldb.org/pvldb/vol10/p1586-kang.pdf",
    "컴퓨터보안 프로젝트 팀. 악성 URL 탐지 실험 코드. https://github.com/andrew427dev/malicious-url-cascade",
    "Latotzke, C., Loh, J., Gemmeke, T. (2021). Cascaded Classifier for Pareto-Optimal Accuracy-Cost Trade-Off Using off-the-Shelf ANNs. https://arxiv.org/abs/2110.14256",
    "Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. IJCAI, §1. https://cseweb.ucsd.edu/~elkan/rescale.pdf",
]


class Report:
    def __init__(self):
        self.ev = Evidence()
        self.doc = Document(TEMPLATE_DIR / "nov23-format.docx")
        self.doc.paragraphs[1].text = TITLE
        self.doc.paragraphs[2].text = SUBTITLE
        self.date = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
        self.doc.tables[0].cell(5, 1).text = self.date
        self.n = 0
        self.figures = 0
        self.sections = 0
        title = self.doc.add_paragraph("목차")
        title.runs[0].bold = True
        title.runs[0].font.size = Pt(23)
        title.paragraph_format.page_break_before = True
        for i, h in enumerate(HEAD, 1):
            p = self.doc.add_paragraph(style="toc 1")
            p.paragraph_format.tab_stops.add_tab_stop(Cm(15.9), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            p.add_run((str(i) + "  " if i < len(HEAD) else "") + h + "\t")
            field(p, f"PAGEREF section{i} \\h")

    # ------------------------------------------------------------ document primitives
    def chapter(self, i):
        text = f"{i}  {HEAD[i - 1]}" if i < len(HEAD) else HEAD[i - 1]
        p = self.doc.add_paragraph(text, style="Heading 1")
        p.paragraph_format.page_break_before = True
        bookmark(p, i)
        self.chapter_no, self.sections = i, 0

    def sub(self, text, numbered=True):
        if numbered and self.chapter_no in (4, 5):
            self.sections += 1
            text = f"{self.chapter_no}.{self.sections}  {text}"
        p = self.doc.add_heading(text, 2)
        p.paragraph_format.page_break_before = False

    def p(self, s):
        self.doc.add_paragraph(s)

    def t(self, caption, headers, rows, widths=None):
        self.n += 1
        widths = widths or [16 / len(headers)] * len(headers)
        assert abs(sum(widths) - 16) < 0.25, (caption, sum(widths))
        assert all(len(r) == len(headers) for r in rows), caption
        table(self.doc, ReportTable(f"표 {self.n}. {caption}", list(headers), [list(map(str, r)) for r in rows], widths), False)

    def img(self, file, caption, width=15.6):
        self.figures += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(file), width=Cm(width))
        self.doc.add_paragraph(f"그림 {self.figures}. {caption}", style="Caption")

    # ------------------------------------------------------------ chapters
    def ch1(self):
        ev = self.ev
        self.chapter(1)
        self.p("새로 만들어진 피싱 주소는 차단 목록에 오르기 전에도 사용자를 속일 수 있다. URL 문자열의 통계적 특징을 학습하는 방법은 웹페이지를 열기 전에 의심 주소를 골라낼 수 있다 [1, 2]. 그러나 모든 주소에 순환 신경망을 실행하면 정상 주소에도 같은 계산 비용이 든다.")
        self.p("이 보고서의 확장은 캐스케이드(선택적 2단계 검사)다. 가벼운 1단계 모델이 수치 특징 18개로 악성 가능성을 먼저 계산하고, 점수가 낮은 주소는 정상으로 일찍 끝낸다. 나머지만 원저자 구조의 GRU(게이트 순환 유닛) 모델로 보낸다 [9]. 쉬운 사례를 앞 단계에서 걸러 뒤 단계 계산을 줄이는 생각은 Viola–Jones 검출기와 NoScope에서 가져왔다 [7, 8]. 일찍 끝낸 주소 가운데 GRU라면 잡았을 악성이 있으면 그만큼 미탐이 늘어난다. 따라서 줄어든 시간과 늘어난 미탐을 함께 보고한다.")
        self.p("주 실험은 팀 정제 데이터에서 URL 무작위 분할과 도메인 분할을 같은 비중으로 다룬다. 모델과 분기 임계값은 분할마다 한 번 학습·선택해 고정했다. 이후 분석(실험 1~6)은 다시 학습하지 않고 이 고정 모델로 단순 기준 모델, 추가 미탐의 유형, 같은 오탐률 비교, 재검사율별 실측 시간을 살폈다. 정제를 덜 한 기본 정제 데이터의 결과는 5.8절에서 같은 표에 나란히 비교한다.")
        self.p("이 결과는 공개 CSV 세 개를 합쳐 새로 나누고 정제한 데이터에서의 평가다. 원저자 저장소나 논문의 수치와 직접 비교할 수 있는 값이 아니며, 원저자 발표값은 2장의 재현 비교표(표 4)와 정제 데이터 참고 비교표(표 5)에만 제시한다.")
        self.sub("연구 질문", False)
        self.t("연구 질문과 답하는 절 (11월 13일 확장 선언)", ["질문", "내용", "답하는 절"], [
            ["RQ1", "선택적 2단계 검사로 GRU 단독 대비 악성 재현율 감소를 정해진 허용 범위(팀 기준 1%p) 안에 두면서 처리 시간을 줄일 수 있는가", "5.1, 5.2, 5.6"],
            ["RQ2", "LR과 HGB 1단계는 탐지, GRU 전달 비율, 전체 시간에서 어떻게 다른가", "5.1, 5.2, 5.5"],
            ["RQ3", "같은 재검사율에서 경계 재검사가 무작위 재검사보다 악성을 더 되찾는가, 추가 비용과 오탐은 얼마인가", "5.3"],
            ["RQ4", "1단계 임계값, 재검사율, 도메인 분할이 탐지와 비용의 관계에 어떤 영향을 주는가", "5.3, 5.4, 5.8"],
        ], [1.6, 11.6, 2.8])
        self.sub("주요 결과", False)
        for s in results.summary(ev):
            self.p(s)
        self.sub("이 보고서의 용어", False)
        self.t("보고서 전체에서 같은 뜻으로 쓰는 용어", ["용어", "뜻"], [
            ["공개 원본", "원저자가 공개한 CSV 3개, 1,291,488행"],
            ["원저자 재현", "원저자 노트북의 데이터 처리·학습 절차를 재현한 결과(2장). 앞 CSV 2개를 80/20으로 나누고 세 번째 CSV로 별도 평가한다. 노트북 자체의 중복·결측 처리는 하고 팀의 정제는 하지 않는다"],
            ["기본 정제", "라벨이 충돌하는 원본 URL을 모두 빼고, 같은 URL 반복을 지운 뒤, 학습·검증·시험 각 집합 안에서 모델 문자열이 같은 행은 첫 행만 남긴 데이터"],
            ["팀 정제", "원본 URL 규칙은 기본 정제와 같고, 잘못된 URL 제외, 정규화(스킴·호스트 소문자, 경로 끝 슬래시 제거) 후 충돌·반복 제거, 세 집합 전체에서 모델 입력(앞 52자+수치, 토큰+수치)의 충돌·반복 제거를 한 데이터"],
            ["URL 무작위 / 도메인 분할", "정제 전에 원본 URL마다 학습·검증·시험 역할을 배정한 두 방식. 도메인 분할은 같은 등록 도메인을 한 역할에만 둔다"],
            ["1단계 모델", "수치 특징 18개를 읽는 로지스틱 회귀(LR) 또는 히스토그램 기반 그래디언트 부스팅(HGB)"],
            ["단순 기준 모델", "URL 길이 하나만 쓰는 로지스틱 회귀(URL 길이 모델)와 ‘http(s)://로 시작하면 악성’ 규칙(접두어 규칙)"],
            ["기본 분기 / 재검사", "기본 분기는 1단계 점수가 임계값 미만이면 정상으로 끝낸다. 재검사는 그 종료 후보의 5%를 다시 GRU로 보낸다(무작위 또는 점수가 높은 순인 경계)"],
            ["전체 시험 / 공통 시험", "각 정제가 남긴 시험 전부 / 두 정제에 모두 남은 같은 시험 URL"],
            ["추가 미탐", "GRU 단독은 악성으로 잡았지만 캐스케이드가 1단계에서 정상으로 끝내 놓친 실제 악성"],
            ["GRU 전달 비율", "시험 URL 가운데 GRU까지 보낸 비율"],
        ], [3.9, 12.1])

    def ch2(self):
        self.chapter(2)
        self.p("Munna Ram의 공개 노트북은 URL을 연속된 세 문자로 읽는 분기와 수치 특징 18개를 읽는 분기를 결합한다 [3]. 문자 임베딩 16차원과 GRU 16유닛을 쓰고, 수치 분기의 ReLU 밀집층과 두 분기를 연결한 뒤의 밀집층도 각각 16유닛이다. 출력 하나가 정상 확률을 나타낸다. URLNet은 문자와 단어 수준 표현을 학습해 URL 구조를 반영했다 [4]. 원저자 모델은 단어 분기 대신 수치 특징을 함께 쓴다.")
        self.p("원저자 노트북은 앞의 두 CSV를 섞고 전체 열이 같은 행과 결측을 지운 뒤 라벨 비율을 유지해 학습 80%, 내부 검증 20%로 나눈다. 각 집합에서 전처리한 URL 문자열이 같은 행은 첫 행만 남긴다. 세 번째 CSV는 별도 평가에 쓴다. 이와 따로 세 파일을 모두 합쳐 새 어휘와 모델을 만들며, 이 마지막 모델에는 남겨 둔 시험 집합이 없다. 평가 지표는 정확도와 가중 정밀도·재현율·F1이다.")
        manifest = readj(V1 / "source-manifest.json")
        self.t("공개 원본의 라벨과 원저자 노트북에서의 역할", ["파일", "정상", "악성", "원저자 노트북의 역할"],
               [(k.split("/")[-1], num(v["counts"]["benign"]), num(v["counts"]["malicious"]),
                 "별도 평가" if "another" in k else "80/20 분할") for k, v in manifest["files"].items()],
               [6.7, 2.45, 2.45, 4.4])
        self.sub("원저자 재현", False)
        rows, auth = results.reproduction()
        auth_val, auth_ext = results.author_published()
        self.t("원저자 공개 노트북과 원저자 재현. 집합 역할별로 대응하는 지표 [3]",
               ["집합 · 지표", "원저자 발표 (%)", "원저자 재현 (%)", "차이 (%p)"], rows, [6.4, 3.2, 3.2, 3.2])
        self.p("원저자는 최초 셔플, 분할 행 목록과 난수 초기 상태를 공개하지 않았다. 재현에는 고정 seed 42를 썼고, 원저자의 TPU 대신 CPU에서 공개 모델 생성 함수를 실행했다. 이 차이는 학습 곡선과 남는 첫 행까지 바꿀 수 있다. 공개 노트북의 전체 데이터 학습 출력은 19회차 도중에 끝나므로, 마지막 행의 재현값은 20회차 학습 중 집계한 정확도이며 시험 성능으로 쓸 수 없다.")
        valdata = readj(V1 / "author-validation/data.json")
        ext = auth["external"]
        novel = auth["external_without_exact_training_or_validation_url"]
        self.sub("재현에서 발견한 데이터 문제와 정제 이유", False)
        self.p(f"재현 모델은 별도 CSV의 악성 {num(ext['tp'] + ext['fn'])}건 가운데 {num(ext['tp'])}건만 탐지했다. 내부 학습 자료와 별도 CSV에는 같은 원본 주소가 {num(valdata['overlaps']['train/external']['url'])}개 있었다. 내부 학습·검증에 있던 주소를 뺀 별도 CSV(악성 {num(novel['malicious'])}행, 정상 {num(novel['benign'])}행)의 악성 재현율은 {pct(novel['malicious_recall'])}%였다. 따라서 별도 CSV를 독립적인 외부 검증으로 보기 어렵다.")
        self.p(f"공개 원본 {num(manifest['raw']['rows'])}행에는 서로 다른 URL {num(manifest['unique_urls'])}개가 있었다. 완전히 같은 문자열에 정상과 악성 라벨이 모두 붙은 주소는 {num(manifest['ambiguous_urls'])}개(해당 {num(manifest['ambiguous_rows'])}행)였고, 같은 URL·라벨이 반복된 여분 행은 {num(manifest['identical_label_repeats_over_all_rows'])}개였다. 같은 주소가 학습과 시험에 함께 들어가거나 시험 라벨이 서로 모순되면 모델 비교가 흔들린다 [5, 6]. 그래서 팀은 11월 11일 원저자에게 정제한 자료를 요청했고, 11월 16일까지 회신이 없어 공개 자료를 직접 정제했다. 교수님은 11월 13일 답장에서 중복 제거 전에 정규화하고 단계별 건수를 보고하도록 요청했다. 이 요구를 충족하는 팀 정제를 주 실험 데이터로 정했다.")
        self.sub("원저자 발표값과의 참고 비교", False)
        ev = self.ev
        acc = {(c, s): ev.m(k[s], "gru").accuracy for c, k in (("basic", BASIC), ("team", TEAM)) for s in SPLITS}
        am, ab = results.author_counts()
        g = {(c, s): ev.m(k[s], "gru") for c, k in (("basic", BASIC), ("team", TEAM)) for s in SPLITS}
        same = "위와 같음"
        basic_rule, team_rule = "CSV 3개 통합, 원본 URL 라벨 충돌·반복 제거, 집합 안 같은 모델 문자열 첫 행", "CSV 3개 통합, 잘못된 URL 제외, 원본·정규화 URL과 모델 입력 기준 충돌·반복 제거"
        split_text = {"random": "층화 70/15/15 (seed 42·43)", "domain": "등록 도메인 묶음 70/15/15"}
        rows = [["원저자 발표 · 내부 검증", "앞 CSV 2개, 전체 열 중복·결측 제거, 집합 안 같은 모델 문자열 첫 행", "층화 80/20 (seed 미공개)", "미공개", "원본 URL · 20회차 가중치 · TPU", pct(auth_val)],
                ["원저자 발표 · 별도 평가", "세 번째 CSV (학습 자료와 같은 원본 주소가 많음)", "별도 파일", f"{num(am)} / {num(ab)}", same, pct(auth_ext)]]
        for c, rule, inp in (("basic", basic_rule, "원본 URL"), ("team", team_rule, "정규화 URL")):
            for s in SPLITS:
                r = g[c, s]
                rows.append([f"{'기본' if c == 'basic' else '팀'} 정제 GRU · {SPLIT_NAME[s]} 시험", rule if s == "random" else same, split_text[s],
                             f"{num(r.tp + r.fn)} / {num(r.tn + r.fp)}", f"{inp} · 검증 손실 최소 에포크 · CPU" if s == "random" else same, pct(r.accuracy)])
        self.t("원저자 발표 정확도와 정제 데이터 GRU 정확도의 참고 비교. 같은 조건의 비교가 아니다", ["결과", "데이터·정제", "분할", "악성 / 정상", "수치 특징 입력 · 가중치 · 장비", "정확도 (%)"], rows,
               [3.0, 4.0, 2.4, 2.2, 2.9, 1.5])
        self.p("확장 선언과 피드백 반영 계획에서 약속한 참고 비교다. 모델 구조(문자 3그램 50칸과 수치 특징 18개, GRU 16유닛)는 모든 행이 같다. 원저자 내부 검증의 행 수와 분할 seed는 공개되지 않아 ‘미공개’로 적었다. "
               "원저자 내부 검증은 앞의 두 CSV를 80/20으로 나눈 뒤 집합마다 따로 중복을 지운 결과이고, 별도 평가는 세 번째 CSV의 결과다. 이 보고서의 시험은 세 CSV를 합쳐 원본 URL마다 역할을 먼저 배정한 뒤 각 정제에서 남은 시험 URL이다. 데이터 구성과 분할이 다르므로 정확도 차이를 성능의 향상이나 하락으로 해석할 수 없다.")

    def ch3(self):
        ev = self.ev
        self.chapter(3)
        self.sub("실행 환경", False)
        self.p("모든 학습과 측정은 Windows의 AMD Ryzen 7 7800X3D CPU 한 대에서 했다. Python 3.10.18, TensorFlow 2.12.0, NumPy 1.23.5, scikit-learn 1.2.2를 썼다. GRU에는 Adam, 이진 교차 엔트로피, 배치 128, 20에포크를 적용했다. 기본 정제·팀 정제 모델은 검증 손실이 가장 낮은 에포크의 가중치를, 원저자 재현 모델은 노트북처럼 마지막 20회차 가중치를 썼다. GRU의 입력·순환 드롭아웃과 각 분기 및 결합층 뒤 드롭아웃은 0.2이고, 학습 매개변수는 162,481개다. 시간 측정에서는 NumPy·scikit-learn의 BLAS·OpenMP 스레드를 threadpoolctl로 1개로 제한하고, TensorFlow는 가져오기 전에 연산 내부·연산 간 스레드 환경변수를 1로 지정했다. 측정 중 실제로 쓰인 스레드 수는 따로 기록하지 않았다. 보고서와 그림은 python-docx와 matplotlib로 만들었다.")
        self.sub("데이터 정제: 팀 정제의 단계별 건수", False)
        self.p("공개 원본의 URL이나 라벨을 고치지 않고, 규칙에 걸린 행을 빼기만 했다. 아래 표는 교수님이 요청한 순서(원본 → 충돌 제거 → 중복 제거 → 최종 정상·악성)로 팀 정제의 각 단계를 적은 것이다. 라벨 충돌은 같은 키에 정상과 악성이 함께 붙은 경우로, 그 키의 행을 모두 뺀다. ④의 정규화 URL 라벨 충돌 목록은 ②·③을 적용하기 전에 ①을 통과한 전체 행에서 먼저 구했고, 행은 표의 순서대로 뺐다. 반복은 같은 키·같은 라벨의 여분 행으로, 대표 한 행만 남긴다.")
        self.t("팀 정제의 단계별 제외 행과 남은 행 (두 분할 공통)", ["단계", "제외 행", "제외 악성 / 정상", "남은 행"],
               results.cleaning_rows(), [6.9, 2.6, 3.6, 2.9])
        self.p(results.control_text(ev))
        self.p(results.basic_text(ev))
        self.sub("수치 특징과 문자 입력", False)
        self.p("아래 열은 원저자 create_features의 18개 특징 정의와 열 순서다. 기본 정제는 원본 URL에서, 팀 정제는 정규화한 URL에서 특징을 계산한다. 표의 원문은 이 단계에 들어온 문자열이다. 생략된 접두어를 파서가 보충하지 않는다.")
        self.t("수치 특징 18개와 실제 계산", ["순서", "원저자 열 이름", "계산법"],
               [(i, n, URL_LENGTH if n == "url_length" else v) for i, (n, v) in enumerate(FEATURE_ROWS, 1)], [1.35, 4.25, 10.4])
        self.p("문자 입력은 수치 계산 뒤에 만든다. 문자를 소문자로 바꾸고, 문자열 어디에 있든 http://, https://, www.를 차례로 지운 뒤 공백을 없앤다. 영문자와 숫자 밖의 문자도 지운다. 한 글자씩 이동하며 연속 세 문자를 한 토큰으로 만들고 처음 50칸을 쓴다. 짧은 입력은 0으로 채우고 학습 어휘 밖 토큰에는 미등록 기호를 쓴다. 어휘는 분할마다 학습 URL로만 만들고, 검증·시험에는 그 어휘를 그대로 적용했다.")
        self.p(example(V1))
        self.p(results.representation_text(ev))
        self.sub("학습·검증·시험 분할", False)
        self.p("정제하기 전에 공개 원본의 서로 다른 URL마다 역할을 한 번 배정하고, 배정표를 파일로 저장했다. 두 정제가 같은 배정을 쓰므로, 정제가 빼지 않은 URL은 두 데이터에서 같은 역할을 가진다. URL 무작위 분할은 정상·악성·라벨 충돌을 층으로 두고 70/30(seed 42)으로 나눈 뒤 나머지를 검증과 시험으로 반씩(seed 43) 나눴다. 도메인 분할은 비공개 접미사를 포함한 오프라인 공개 접미사 목록으로 등록 도메인을 구하고, 같은 등록 도메인을 한 역할에만 두도록 같은 비율·seed로 묶음 단위 분할을 했다. 도메인 묶음은 크기가 달라 실제 행 비율은 70/15/15와 조금 다르다.")
        self.p("모든 모델은 같은 분할을 쓴다. 검증 집합은 1단계 모델 선택·확률 보정·분기 임계값 선택, GRU 에포크 선택, 실험 5의 GRU 기준 선택에만 쓰고, 시험 집합은 최종 결과 계산에만 썼다. 시험값은 모델·임계값 선택에 쓰지 않았다. 후속 분석(실험 1~6)을 설계할 때 시험 결과를 이미 본 점은 6장에 적었다.")
        self.t("팀 정제의 역할별 행 수", ["분할 · 역할", "행", "악성", "정상", "등록 도메인"], results.role_rows(ev),
               [5.2, 2.7, 2.7, 2.7, 2.7])
        self.p(results.split_overlap_text(ev))

    def ch4(self):
        ev = self.ev
        self.chapter(4)
        draw(OUT / "cascade.png", "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf")
        self.img(OUT / "cascade.png", "캐스케이드의 실제 처리 순서. 수치 특징을 먼저 계산하고, GRU로 보낸 URL에서만 문자 토큰과 GRU 출력을 계산한다.")
        self.p("두 분할 각각에서 GRU 단독과 캐스케이드 여섯 개(1단계 LR·HGB × 기본 분기·무작위 재검사 5%·경계 재검사 5%)를 같은 시험 URL로 비교한다. 여기에 단순 기준 모델 두 개와 1단계 모델 단독 두 개를 더해 구성은 분할마다 11개다.")
        self.sub("1단계 모델 선택과 확률 보정")
        self.p("1단계는 학습 집합으로 적합한 LR과 HGB다. LR은 규제 강도의 역수 C를 0.1·1·10, HGB는 최대 잎 수를 7·15·31 가운데 검증 평균 정밀도(AP)가 가장 높은 값으로 골랐다. 교수님 요청대로 고른 모델의 확률은 검증 집합에서 시그모이드 방식으로 보정했다. 모든 분기 규칙은 이 보정한 악성 점수(0~1)를 쓴다.")
        self.t("1단계 후보의 검증 AP와 선택 (팀 정제)", ["분할 · 모델", "후보", "검증 AP", "선택"], results.stage1_rows(ev),
               [4.0, 4.0, 4.6, 3.4])
        self.sub("분기 임계값과 재검사")
        self.t("분기 임계값과 재검사 예산을 정한 방법", ["항목", "선정 기준"], [
            ["분기 임계값 후보", "보정한 1단계 악성 점수에서 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50을 검증 집합으로 비교했다."],
            ["임계값 선택", "재검사 없이 GRU 단독 대비 검증 악성 재현율 손실이 1%p 이내인 후보 가운데 GRU 호출이 가장 적은 값을 골랐다. 호출 수가 같으면 낮은 값을 골랐다. 점수가 임계값 이상이면 바로 GRU로 보낸다."],
            ["재검사 예산 5%", "11월 24일 팀 합의에서 정상 종료 후보의 5%로 정했다. 같은 건수를 다시 검사할 때 어느 선정 방식이 악성을 더 되찾는지 비교하기 위한 예산이며, 운영에 최적인 비율을 뜻하지 않는다. 1·10·20%는 5.3절에서 따로 본다."],
            ["무작위 선정", "후보 수에 재검사율을 곱해 소수점 아래를 버린 건수만큼, seed와 URL 식별자를 이은 문자열의 SHA-256 값이 작은 순으로 뽑는다. 주 결과는 seed 0, 변동 확인은 seed 0~29다."],
            ["경계 선정", "같은 후보를 악성 점수가 높은 순으로 정렬해 같은 건수를 뽑는다. 동점은 식별자 순이다."],
            ["최종 판정", "GRU로 간 URL은 GRU의 악성 점수가 0.5보다 크면(정상 확률 0.5 미만) 악성이다. 1단계에서 끝난 URL은 정상이다."],
        ], [3.4, 12.6])
        self.t("검증에서 고정한 1단계 분기 임계값 (악성 점수 0~1, 팀 정제)", ["분할", "LR", "HGB"],
               [[SPLIT_NAME[s], f2(ev.tau(TEAM[s], "logistic")), f2(ev.tau(TEAM[s], "boost"))] for s in SPLITS],
               [6.0, 5.0, 5.0])
        self.sub("단순 기준 모델과 1단계 단독")
        self.p("교수님은 URL 길이 같은 단순 특징만으로도 높은 점수가 나오는지 확인해 지름길 특징을 점검하라고 요청했다. URL 길이 모델은 학습 집합에서 URL 길이(수치 특징 14번) 하나로 적합한 로지스틱 회귀이며, 악성 확률 0.5 이상을 악성으로 본다. 접두어 규칙은 원본 URL이 http:// 또는 https://로 시작하면 악성으로 본다. 1단계 LR·HGB 단독은 보정 확률 0.5 이상을 악성으로 본 결과로, GRU 없이 1단계만 쓸 때의 성능이다.")
        self.sub("같은 오탐률 비교")
        self.p("조기 종료는 GRU의 오탐 일부도 함께 없앤다. 그런데 GRU 단독도 판정 기준을 0.5보다 높이면 재현율을 조금 내주고 오탐을 줄일 수 있다. 캐스케이드의 오탐 감소가 이런 기준 이동보다 나은지 보려고, 각 캐스케이드의 검증 오탐률을 넘지 않는 가장 작은 GRU 판정 기준을 검증 점수에서 정하고 시험 재현율을 비교했다. 보조로, 검증 악성 탐지 수가 해당 캐스케이드 이상이 되는 GRU 기준 가운데 가장 큰 값을 골라 시험 오탐률도 비교했다. 기준은 검증 집합에서만 정했다. 차이의 95% 구간은 부트스트랩 10,000회로 구했다. URL 무작위 분할은 재현율에 악성 행, 오탐률에 정상 행을, 도메인 분할은 해당 행이 있는 등록 도메인 묶음을 다시 뽑는다.")
        self.sub("시간 측정 규약")
        self.p("시간은 두 정제와 두 접두어 표기에 모두 남은 시험 URL 가운데 분할마다 고정한 1,000개(seed 46)로 쟀다. 원본 문자열에서 정규화·수치 특징·문자 토큰을 만드는 시간부터 최종 판정 배열이 나올 때까지를 잰다. 모델 로딩, 파일 읽기, 정답 대조는 빼고, 재검사 예산은 이 1,000건 안에서 계산한다. 각 구성을 한 번 예열한 뒤 30회 반복하며, 회차마다 구성의 실행 순서를 섞었다(seed 47). 절약 시간은 같은 회차의 GRU 단독 시간에서 캐스케이드 시간을 뺀 값의 중앙값이다. 30개 회차를 짝째로 10,000회 복원추출해(seed 509) 중앙값의 95% 구간을 구했다.")
        self.p("시간 측정은 따로 두 번 했다. 측정 A는 네 조건(두 정제 × 두 표기)의 주 구성과, 원래 표기 두 정제의 임계값 후보 12개를 한 측정 안에서 함께 섞어 쟀고, 5.2절의 판정, 5.4절의 임계값 민감도, 5.7절 보조 B와 5.8절의 시간 비교, 부록 C·F에 쓴다. 측정 B는 실험 6으로, 팀 정제 두 분할의 재검사율 1·5·10·20%를 측정 A와 같은 1,000건·측정 경로·반복 방식으로 따로 쟀다(5.3절). 두 측정에 모두 들어간 0%·5% 구성은 서로 대조했다. 측정이 다르면 같은 구성의 ms 값도 조금 다를 수 있다.")
        self.sub("판정 기준")
        self.p("재현율 손실은 추가 미탐 수를 실제 악성 수로 나눈 값이다. 95% 구간은 URL 무작위 분할에서 Wilson 구간, 도메인 분할에서 악성 URL이 있는 등록 도메인을 묶어 4,000회 재표집한 구간이다. 캐스케이드는 재현율 손실 구간의 상한이 1%p 미만이고 절약 시간 구간의 하한이 0보다 크면 ‘충족·단축’이다. 시간 구간이 0을 포함하거나 손실 구간이 1%p를 포함하면 ‘유보’, 그 밖에 절약 시간 구간 전체가 0 미만이거나 손실 구간 전체가 1%p를 넘으면 ‘미충족’이다. 도메인 분할에서 임계값이 0보다 큰데 추가 미탐이 한 건도 없으면 도메인 재표집으로 상한을 추정할 수 없어 ‘유보’로 둔다. 임계값 0은 모든 URL을 GRU로 보내므로 손실이 정확히 0이고 시간 조건으로만 판정한다. 1%p 허용치는 팀이 정했다. 정상 오탐과 오탐률은 함께 보고하지만 판정 기준에는 넣지 않았다.")
        self.t("실험과 결과 위치", ["실험", "내용", "결과"], results.EXPERIMENTS, [2.4, 10.8, 2.8])
        self.p(results.EXPERIMENT_ORIGIN)

    def ch6(self):
        ev = self.ev
        self.chapter(6)
        self.p("시험 데이터 노출 이력. 팀은 실험 1~6을 설계하기 전에 이 공개 코퍼스의 시험 결과(주 실험 포함)를 이미 확인했다. 실험 1~6은 새 학습 없이 고정 모델의 결과를 더 자세히 본 것이지만, 분석 항목을 정할 때 시험 결과를 알고 있었다. 따라서 이 결과는 완전히 맹검인 시험이 아니다 [6].")
        self.p("검증 데이터의 재사용. 같은 검증 집합을 GRU 에포크 선택, 1단계 모델 선택, 확률 보정, 분기 임계값 선택, 같은 오탐률 기준 선택에 모두 썼다. 검증 집합에 맞춘 선택이 여럿 겹쳐 있어, 다른 검증 표본에서도 같은 설정이 선택될지와 시험 성능이 낙관적으로 추정됐는지는 확인하지 못했다.")
        self.p("남은 라벨의 정확성. 규칙으로 찾을 수 있는 라벨 충돌은 뺐지만, 남은 URL이 실제로 악성인지 외부 기관이나 사람이 다시 확인하지 않았다. 접두어 유무와 라벨이 강하게 묶여 있다는 점(5.7절)도 수집 방식에서 생긴 특성일 수 있다.")
        changes = [] if self.ev.e6 is None else results.session_changes(self.ev)
        self.p("시간 측정의 범위. 한 CPU에서 같은 1,000개를 배치로 처리한 시간만 쟀다. 단건 요청, 동시 사용자, GPU 추론의 시간은 재지 않았다. 시간을 두 번 따로 쟀으므로(4.5절) 두 측정 사이의 ms 값은 직접 비교하지 않는다. "
               + (f"절약 폭이 1 ms 미만인 구성은 측정에 따라 판정이 달라질 수 있다. 예를 들어 {results.changes_sentence(changes)}(5.3절). " if changes else "")
               + "임계값별 95% 구간은 개별 비교용이며 여러 비교를 동시에 보정한 구간이 아니다.")
        self.p("학습 seed 하나. 각 학습 조건(분할 × 정제 × 표기)에서 GRU와 1단계 모델을 한 번씩 학습했고, 난수 seed를 바꾼 반복 학습은 하지 않았다(보조 B는 접두어를 지운 표기로 따로 학습한 모델이다). 탐지 지표의 구간은 시험 표본의 변동을, 시간 구간은 같은 1,000건을 반복 측정한 변동을 나타내며, 다른 seed로 처음부터 다시 학습할 때의 변동은 포함하지 않는다. 기본 정제와 팀 정제의 차이도 학습 자료·어휘·입력 표기가 함께 바뀐 결과라 한 원인으로 나눌 수 없다(5.8절).")
        self.p("도메인 분할의 한계. 같은 등록 도메인을 한 역할에 두어도, 다른 도메인이 같은 공격 캠페인에 속하거나 수집 시점이 겹칠 수 있다.")
        self.p("후속 과제는 이후 시점에 모은 데이터로 하는 독립 시험이다. 등록 도메인과 수집 출처를 함께 나누고, 검증을 모델 선택용과 임계값용으로 나누면 위 한계 가운데 노출 이력과 검증 재사용을 줄일 수 있다. 오탐 허용 범위는 운영 목적과 오분류 비용을 확인한 뒤 정해야 한다 [11].")

    def ch7(self):
        self.chapter(7)
        for s in results.conclusion(self.ev):
            self.p(s)
        self.p("정상 오탐을 얼마나 허용할지는 운영 목적과 피해 비용을 확인한 뒤 정할 일이다. 시간을 더 줄이는 구성과 악성을 덜 놓치는 구성이 다를 수 있으므로 두 지표를 함께 읽어야 한다 [10, 11].")
        self.sub("시연", False)
        self.p(results.demo_text(self.ev))
        self.sub("참고문헌", False)
        for i, ref in enumerate(REFS, 1):
            self.p(f"[{i}] {ref}")

    def build(self):
        self.ch1()
        self.ch2()
        self.ch3()
        self.ch4()
        results.chapter5(self)
        self.ch6()
        self.ch7()
        results.appendix(self)
        self.doc.core_properties.title = TITLE
        self.doc.core_properties.author = "컴퓨터보안 팀프로젝트"
        path = OUT / (TITLE + (" (초안)" if self.ev.draft else "") + ".docx")
        self.doc.save(path)
        return path


if __name__ == "__main__":
    print(Report().build())
