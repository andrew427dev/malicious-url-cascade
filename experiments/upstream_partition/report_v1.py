"""Create an independent Korean report from the v1 evidence tables and November 23 styles."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_TAB_ALIGNMENT,WD_TAB_LEADER
from make_report import field,bookmark,table
from table_layout import ReportTable
from feature_table import FEATURE_ROWS,example
from diagram import draw
from report_data import *
import recheck_tables
import re


TITLE="악성 URL 탐지 최종 실험 보고서 v1"
HEAD=["Introduction & Motivation","Original Work Summary","System Setup","Experiments (highlighting your extension settings)","Results & Discussion (highlighting your results)","Limitations & Future Work","Conclusion and References"]
REFS=[
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
"Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. IJCAI, §1. https://cseweb.ucsd.edu/~elkan/rescale.pdf"]
class Report:
    def __init__(self):
        self.ev=Evidence()
        self.out=ROOT/"report"; self.out.mkdir(exist_ok=True)
        self.doc=Document(Path(__file__).with_name("nov23-format.docx"))
        self.doc.paragraphs[1].text=TITLE
        self.doc.paragraphs[2].text="URL 정제 절차와 선택적 GRU 검사의 탐지 성능 및 처리 시간"
        self.date=datetime.now(timezone(timedelta(hours=9))).date().isoformat()
        self.doc.tables[0].cell(5,1).text=self.date
        self.n=0
        title=self.doc.add_paragraph("목차");title.runs[0].bold=True;title.runs[0].font.size=Pt(23)
        title.paragraph_format.page_break_before=True
        for i,h in enumerate(HEAD,1):
            p=self.doc.add_paragraph(style="toc 1")
            p.paragraph_format.tab_stops.add_tab_stop(Cm(15.9),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
            p.add_run(str(i)+"  "+h+"\t");field(p,f"PAGEREF section{i} \\h")
    def chapter(self,i):
        p=self.doc.add_paragraph(f"{i}  {HEAD[i-1]}",style="Heading 1")
        p.paragraph_format.page_break_before=True;bookmark(p,i)
    def sub(self,text):
        p=self.doc.add_heading(text,2);p.paragraph_format.page_break_before=False
    def p(self,s):self.doc.add_paragraph(s)
    def t(self,caption,headers,rows,widths=None):
        self.n+=1
        widths=widths or ([4.55,4.05,4.05,3.35] if len(headers)==4 else [16/len(headers)]*len(headers))
        table(self.doc,ReportTable(f"표 {self.n}. {caption}",list(headers),list(rows),widths),False)
    def img(self,file,caption):
        p=self.doc.add_paragraph();p.paragraph_format.keep_with_next=True
        p.paragraph_format.line_spacing=1.0
        p.paragraph_format.space_before=Pt(5)
        p.paragraph_format.space_after=Pt(2)
        p.add_run().add_picture(str(file),width=Cm(15.6))
        self.doc.add_paragraph(caption,style="Caption")
    def main(self):
        ev=self.ev
        self.chapter(1)
        self.p("새로 만들어진 피싱 주소는 차단 목록에 올라가기 전에도 사용자를 속일 수 있다. URL 문자열의 통계적 특징을 학습하는 방법은 웹페이지를 열기 전에 의심 주소를 선별할 수 있다 [1, 2]. 모든 주소에 순환 신경망을 실행하면 정상 주소에도 같은 계산 비용이 든다. 일부 주소의 검사를 앞에서 끝낼 때 절약하는 시간과 추가로 놓치는 악성 주소를 함께 알아야 한다.")
        self.p("세 공개 CSV에서 주소별 역할을 정한 후 정제 절차 두 가지로 각각 GRU를 학습했다. 같은 시험 주소에서 두 모델을 비교해 시험 집합 구성이 달라지는 영향을 줄였다. 각 모델 안에서는 가중치를 고정한 채 여섯 가지 조기 종료 구성을 GRU 단독과 비교했다.")
        self.sub("주요 결과")
        self.p("GRU(게이트 순환 유닛)는 주소의 문자 순서를 읽는 신경망이다. 팀은 이 신경망 앞에서 로지스틱 회귀(LR) 또는 히스토그램 기반 그래디언트 부스팅(HGB)이 수치 특징으로 악성 가능성을 먼저 추정하도록 구현했다 [9]. 경계 재검사는 정상으로 끝내려던 후보 가운데 악성 점수가 높은 5%를 다시 GRU로 보내는 방식이다.")
        for s in ev.summary:self.p(s)
        self.chapter(2)
        self.p("Munna Ram의 공개 노트북은 URL을 연속된 세 문자로 읽는 분기와 수치 특징 18개를 읽는 분기를 결합한다 [3]. 문자 임베딩 16차원과 GRU 16유닛을 사용하고, 수치 분기의 ReLU 밀집층과 두 분기를 연결한 뒤의 밀집층도 각각 16유닛이다. 출력 하나가 정상 확률을 나타낸다. URLNet은 문자와 단어 수준 표현을 학습해 URL 구조를 반영했다 [4]. 이 실험의 GRU는 단어 분기 대신 수치 특징을 함께 사용한다.")
        self.p("원저자 재현에서는 앞의 두 CSV를 섞고 전체 열이 같은 행과 결측을 제거한 뒤 라벨 비율을 유지해 학습 80%, 내부 검증 20%로 나눴다. 세 번째 CSV는 이 모델의 별도 평가에 사용했다. 이와 독립적으로 세 파일을 모두 합친 뒤 새로운 어휘와 모델을 만들었다. 마지막 모델에는 남겨 둔 독립 시험 집합이 없다.")
        manifest=readj("source-manifest.json")
        record=readj("results/review-evidence.json")
        times=record["author_event_times"]
        fmt=lambda key: datetime.fromisoformat(times[key]).strftime("%m월 %d일 %H:%M")
        self.p(f"실제 실행에서 앞 두 파일의 모델과 세 파일 전체 모델은 독립적으로 병행 학습했다. 전체 모델의 학습 입력은 {fmt('author-final/training-inputs.json')}에 확정됐고 가중치는 {fmt('author-final/trained.json')}에 저장됐다. 그 사이 {fmt('author-validation/external-scores.npz')}에 저장한 별도 평가 점수는 앞 두 파일로 학습한 모델에서만 얻었다. 두 모델의 어휘와 가중치 파일은 각각 따로 관리했다.")
        self.t("공개 원본의 라벨과 역할",["파일","정상","악성","첫 학습에서의 역할"],[(k.split("/")[-1],num(v["counts"]["benign"]),num(v["counts"]["malicious"]),"별도 평가" if "another" in k else "80/20 분할") for k,v in manifest["files"].items()],[6.7,2.45,2.45,4.4])
        self.p("원저자는 최초 셔플, 분할 행 목록과 난수 초기 상태를 공개하지 않았다. 재현에는 고정 시드 42를 썼다. 원저자의 TPU 환경 대신 CPU에서 공개 모델 생성 함수를 실행했다. 이 차이는 학습 곡선과 남는 첫 행까지 바꿀 수 있다.")
        self.chapter(3)
        self.p("팀은 원저자에게 정제한 자료를 요청했으나 회신을 받지 못해 공개 자료를 직접 정제했다. 기준 커밋이 가리킨 Git LFS 포인터의 SHA-256과 바이트 수를 공개 CSV 실파일과 대조했다. 원본 URL이나 라벨을 소급해 수정하지 않았다. 시험 라벨 오류만으로도 모델 비교 결과가 달라질 수 있어 [5], 충돌과 반복을 나누어 기록했다.")
        self.p(f"공개 원본 {num(manifest['raw']['rows'])}행에는 서로 다른 URL {num(manifest['unique_urls'])}개가 있었다. 완전히 같은 문자열에 정상과 악성 라벨이 모두 붙은 주소는 {num(manifest['ambiguous_urls'])}개, 해당 행은 {num(manifest['ambiguous_rows'])}개였다. 같은 원본·라벨이 반복된 여분 행은 전체 원본 기준 {num(manifest['identical_label_repeats_over_all_rows'])}개였다.")
        self.sub("수치 특징과 문자 입력")
        self.p("아래 열은 공개 함수 create_features가 추가한 순서다. 원저자 전처리 기준은 원본 URL, 팀 정제 기준은 정규화한 URL을 이 함수의 입력으로 사용한다. 표에서 원문은 이 단계에 들어온 문자열이다. URL의 생략된 접두어를 파서가 자동 보충하지 않는다.")
        self.t("수치 입력 18개와 실제 계산",["순서","원저자 열 이름","계산법"],[(i,n,v) for i,(n,v) in enumerate(FEATURE_ROWS,1)],[1.35,4.25,10.4])
        self.p("수치 계산 뒤 문자를 소문자로 바꾼다. 문자열 어디에 있든 http://, https://, www.를 차례대로 지우고 일반 공백을 없앤 뒤 앞뒤 공백을 제거한다. 영문자와 숫자를 제외한 나머지 문자도 지운다. 한 글자씩 이동하며 연속 세 문자를 한 토큰으로 만들고 처음 50칸을 사용한다. 짧은 입력은 0으로 채우고 학습 어휘 밖 토큰에는 미등록 기호를 쓴다.")
        self.p(example(ROOT))
        parity=readj("parity.json");self.p(f"실제 표본 {parity['real_sample_rows']}행과 하이픈·장문·반복 예시를 합쳐 정방향과 역방향에서 각각 {parity['orders'][0]['input']}행을 검사했다. 두 순서 모두 원저자 함수와 재현 코드가 같은 {parity['orders'][0]['kept']}행을 남겼고, 라벨과 토큰·수치 배열이 원소 단위로 일치했다. 하이픈을 삭제해 같은 문자열이 되면 입력 순서상 첫 행이 살아남았다.")
        self.sub("학습 및 측정 환경")
        self.p("Windows의 AMD Ryzen 7 7800X3D CPU에서 Python 3.10.18, TensorFlow 2.12.0, NumPy 1.23.5, scikit-learn 1.2.2를 사용했다. GRU에는 Adam, 이진 교차 엔트로피, 배치 128, 20에포크를 적용했다. GRU의 입력·순환 드롭아웃과 각 분기 및 결합층 뒤 드롭아웃은 0.2다. 총 학습 매개변수는 162,481개다. 시간 측정에서는 CPU 연산 스레드를 1개로 제한했다.")
        self.chapter(4)
        self.p("정제 전에 모든 원본 URL에 하나의 역할을 부여했다. URL 무작위 분할은 정상·악성·라벨 충돌을 각각 층으로 두었고, 도메인 분할은 비공개 접미사를 포함한 오프라인 공개 접미사 목록으로 등록 도메인을 묶었다. 양쪽 모두 학습·검증·시험의 목표 비율은 70/15/15다. 도메인 묶음은 크기가 달라 실제 행 비율도 달라질 수 있다.")
        self.p("원저자 전처리 기준은 팀 비교용 절차다. 같은 원본의 라벨 충돌을 전부 빼고 같은 원본·라벨의 첫 행을 남겼다. 각 역할 안에서 전체 전처리 문자열이 같으면 첫 행만 남겼다. 2장에서 설명한 원저자 재현에는 이 원본 라벨 충돌 제거를 적용하지 않았다.")
        self.p("팀 정제 기준은 제어문자·지원하지 않는 스킴·잘못된 호스트와 포트를 거르고, 스킴과 호스트를 소문자로 바꾸며 경로 끝의 슬래시를 지운다. 경로 대소문자, 쿼리, 프래그먼트, 명시적 포트와 사용자 정보는 보존한다. 정규화 주소, 앞 52문자와 수치 특징, 실제 토큰 50칸과 수치 특징을 차례로 비교한다. 라벨 충돌 그룹은 전부 제외하고 같은 라벨은 대표 한 행을 남긴다.")
        self.p("두 분할의 학습 어휘 중 하나에서라도 최종 입력이 같은 행을 연결해 충돌 그룹을 검사했다. 같은 라벨의 토큰 중복은 정규화 주소 해시 순으로 대표를 정했다. 제외 후 학습 어휘를 다시 만들어 중복이 없어질 때까지 반복하고, 정규화를 두 번 적용했을 때 달라지는 주소도 격리해 다시 확인했다. 11월 16일에는 정제 후 분할을 다시 만들었지만 이번에는 최초 역할을 끝까지 유지했다. 따라서 비교에는 정규화한 표기에서의 특징 계산과 중복 처리 효과가 함께 들어간다.")
        self.sub("경량 분류기와 재검사")
        self.p("Viola와 Jones는 쉬운 음성 사례를 앞 단계에서 제거해 뒤의 계산을 줄였다 [7]. 여기서는 로지스틱 회귀(LR)와 히스토그램 기반 그래디언트 부스팅(HGB)을 대안적인 첫 단계로 사용했다. 검증 평균 정밀도(AP)가 가장 높은 후보를 선택하고 같은 검증 집합에서 시그모이드로 확률을 보정했다. LR의 C는 0.1·1·10, HGB의 최대 잎 수는 7·15·31이었다. 팀 GRU는 검증 손실이 가장 낮은 에포크의 가중치를 고정했다.")
        recheck_tables.selection(self)
        draw(self.out/"cascade.png","C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/malgunbd.ttf")
        self.img(self.out/"cascade.png","그림 1. 숫자 특징을 먼저 계산하고, 전달된 URL에서만 문자 토큰과 GRU 출력을 계산한다.")
        self.p("시간 측정은 두 정제 절차와 두 접두어 표기에 모두 남은 시험 URL에서 분할마다 같은 1,000개를 고정했다. 입력 순서를 유지하고 각 구성을 한 번 예열한 뒤 30회 각각에서 구성의 실행 순서를 섞었다. 원본 문자열에서 정규화·특징·문자 입력을 만드는 시간부터 최종 판정 배열이 완성될 때까지 측정했다. 모델 로딩, 파일 읽기, 표본 추출과 정답 대조는 제외했다. 재검사 예산은 시간 측정 때도 해당 배치 안에서 계산했다.")
        self.p("NoScope가 단계별 비용과 통과율을 함께 본 방법을 참고해 [8], 검증에 사용한 임계값들을 실제로 바꾸며 기본 분기의 시간을 쟀다. 시간 절약량은 같은 회차의 GRU 시간에서 캐스케이드 시간을 뺀 값의 중앙값이다. 30개 회차를 짝째로 10,000회 복원추출해 중앙값의 95% 백분위 구간을 구했다. 전체 시험의 재현율 손실은 추가 미탐 수를 실제 악성 수로 나눈 값이다. 무작위 분할은 Wilson 95% 구간, 도메인 분할은 악성 주소가 있는 등록 도메인을 묶어 4,000회 재표집한 구간이다.")
        self.p("추가 미탐 허용치 1%p는 팀이 정했다. 시간 구간이 0을 포함하거나 재현율 손실 구간이 1%p를 포함하면 판정을 유보했다. 정상 오탐과 오탐률은 함께 기록했고 판정의 탈락 기준에는 넣지 않았다.")
        self.p("GRU 단독과 임계값 0은 어떤 입력이 와도 전부 같은 GRU로 전달하므로 손실과 양 끝값이 모두 0이다. 조기 종료가 가능한 도메인 분할에서 추가 미탐이 한 건도 관측되지 않으면, 관측 도메인을 다시 뽑는 계산으로는 손실의 상한을 알 수 없다. 이때는 상한 추정과 판정을 유보했다.")
        self.results()
        self.doc.save(self.out/(TITLE+".docx"))

    def results(self):
        ev=self.ev
        self.chapter(5)
        auth=readj("author-validation/results.json")
        internal=auth["validation"];ext=auth["external"]
        nb=readj("source/Model_Training.ipynb")["cells"]
        def logged(i):
            return "\n".join("".join(o.get("text",[])) for o in nb[i]["outputs"])
        orig_val=float(re.findall(r"val_accuracy: ([0-9.]+)",logged(9))[-1])
        comparison=[("내부 검증 · 정확도",orig_val,internal["accuracy"])]
        for key,printed,name in (("accuracy","Accuracy","정확도"),("weighted_precision","Precision","가중 정밀도"),("weighted_recall","Recall","가중 재현율"),("weighted_f1","F1 Score","가중 F1")):
            value=float(re.search(printed+r": ([0-9.]+)",logged(10)).group(1))
            comparison.append(("별도 평가 · "+name,value,ext[key]))
        final_fit=readj("author-final/trained.json")["history"]["accuracy"][-1]
        original_rows=[(n,pct(a),pct(b),delta(100*(b-a))) for n,a,b in comparison]
        original_rows.append(("전체 데이터 · 20회차 학습 정확도","기록 없음",pct(final_fit),"—"))
        self.t("원저자 공개 노트북과 이번 재현. 집합 역할별로 대응하는 지표 [3]",["집합 · 지표","원저자 발표 (%)","이번 재현 (%)","차이 (%p)"],original_rows)
        self.p("공개 노트북의 전체 데이터 학습 출력은 19회차 도중에 끝난다. 마지막 행의 이번 수치는 20회차 학습 중 집계한 정확도이며, 남겨 둔 시험 자료에 대한 성능으로 사용할 수 없다.")
        valdata=readj("author-validation/data.json");finaldata=readj("author-final/data.json")
        self.p(f"첫 모델은 {num(valdata['stages']['train']['after']['rows'])}행으로 학습하고, 내부 검증 {num(internal['n'])}행과 별도 CSV {num(ext['n'])}행에 적용했다. 별도 CSV에서 악성 {num(ext['tp']+ext['fn'])}건 중 {num(ext['tp'])}건을 탐지했다. 내부 학습과 별도 CSV에 동일 원본 주소가 {num(valdata['overlaps']['train/external']['url'])}개 있었다. 별도 파일을 독립적인 외부 검증으로 해석하기 어려운 이유다.")
        novel=auth["external_without_exact_training_or_validation_url"]
        self.p(f"중복 영향을 확인한 보조 분석에서는 내부 학습·검증에 있던 정확히 같은 URL을 제외했다. 별도 파일에는 악성 {num(novel['malicious'])}행과 정상 {num(novel['benign'])}행이 남았고, 이 하위 집합의 악성 재현율은 {pct(novel['malicious_recall'])}%였다. 세 파일 전체를 합친 최종 모델은 {num(finaldata['stages']['train']['after']['rows'])}행으로 어휘부터 새로 학습해 20에포크를 마쳤다.")
        self.sub("정제로 바뀐 데이터 구성")
        audit=readj("results/review-evidence.json")
        overlap={v["kind"]:v for v in audit["input_overlap"]}
        for split in ("random","domain"):
            ks=kinds(split); ds=[ev.pop.loc[k] for k in ks]
            rows=[]
            for title,key in (("학습 행","train"),("검증 행","validation"),("시험 행","test"),("시험 악성","malicious"),("시험 정상","benign"),("학습·시험 공통 도메인","common_train_test_domains"),("양쪽 공통 시험 URL","shared_main_test"),("상대 학습·검증 혼입","cross_leaks")):
                a,b=(r[key] for r in ds); rows.append((title,num(a),num(b),num(b-a)))
            p,q=[r.malicious/r.test for r in ds]
            rows.insert(5,("시험 악성 비율 (%)",pct(p),pct(q),delta(100*(q-p))))
            for scope,text in (("full","전체 시험"),("common","공통 시험")):
                x,y=[overlap[k][scope] for k in ks]
                rows.append((text+" 중 학습·검증과 입력이 같은 행",num(x),num(y),num(y-x)))
            self.t(label(split)+" 분할의 역할과 시험 구성",["지표",*AB,"차이 (팀−원저자)"],rows)
            self.t(label(split)+" 분할의 정제 규칙별 제외 / 잔여 행",["규칙",*AB,"제외 수 차이"],ev.cleaning(split))
        self.p("대시는 적용하지 않은 규칙이다. 유효하지 않은 URL을 먼저 제외하는 팀 기준에서는 일부 원본 라벨 충돌이 앞 단계에 이미 집계된다. 원저자 기준은 라벨이 섞여도 전체 문자가 같으면 첫 행을 보존한다. 팀 기준의 토큰 충돌·반복은 두 학습 어휘에서 연결된 그룹을 대상으로 반복 검사한 누계다.")
        for split in ("random","domain"):
            self.sub(label(split)+" 분할의 탐지 성능")
            for scope,desc in (("full","각 절차가 남긴 전체 시험"),("common","양쪽에 남은 동일 시험 URL")):
                a,b=[ev.row(k,scope,"gru") for k in kinds(split)]
                sub=f"{label(split)} · {desc} (원저자 {num(a.n)}건, 팀 {num(b.n)}건)"
                if scope=="common":sub=f"{label(split)} · 공통 시험 {num(a.n)}건"
                self.t(sub+"의 정확도 / 악성 재현율 (%)",["구성",*AB,"차이 (%p)"],ev.metric_rows(split,scope,[("accuracy",100,2),("recall",100,2)]))
                self.t(sub+"의 정상 오탐 건수 / 오탐률 (%)",["구성",*AB,"차이 (건 / %p)"],ev.metric_rows(split,scope,[("fp",1,0),("fpr",100,2)]))
                if scope=="full":
                    self.t("같은 GRU 대비 추가 미탐 건수 / 재현율 손실 (%p) · "+label(split),["구성",*AB,"차이 (건 / %p)"],ev.metric_rows(split,scope,[("additional_misses",1,0),("recall_loss",100,2)]))
                    self.p(f"추가 미탐의 분모는 실제 악성 URL로, 원저자 기준 {num(a.malicious)}건, 팀 기준 {num(b.malicious)}건이다. 재검사 두 방식은 같은 수의 종료 후보를 GRU로 되보냈다. 별첨 결과표에는 정상 오탐률을 비롯해 GRU 호출 수·AP·재검사로 되찾은 악성 수도 담았다.")
            self.p(ev.summary[0 if split=="random" else 2])
            self.p("동일 URL 비교에서도 모델은 각 정제 절차가 남긴 학습 자료로 학습했다. 이 차이에는 입력 표기, 선택된 학습 행, 학습 어휘가 함께 작용한다. 공통 시험의 재검사 예산은 그 공통 집합의 종료 후보 수로 다시 계산했다.")
        self.sub("실제 처리 시간과 미탐 허용 구간")
        pairs=csv("timing-pairs").set_index(["comparison","base","name"])
        for ix,split in enumerate(("random","domain"),2):
            rows=[];d=ev.main.set_index(["kind","name"])
            for n in NAMES:
                a,b=[d.loc[(k,n)] for k in kinds(split)]
                change=pairs.loc[("change_cleaning",kinds(split)[0],n)]
                rows.append((NAMES[n],f"{a.total_ms:.2f} / {a.saved_ms:+.2f}",f"{b.total_ms:.2f} / {b.saved_ms:+.2f}",f"{change.delta_ms:+.2f} [{change.lo_ms:+.2f}, {change.hi_ms:+.2f}]"))
            self.t(label(split)+" · 공통 1,000건의 전체 시간 / 각 GRU 대비 절약 시간 (ms)",["구성",*AB,"팀−원저자 전체 시간 · 짝차이 [95% 구간]"],rows)
            interval_rows=[]
            changes=[]
            loss_rows=[]; stages=[]; ranges=[]
            for n in NAMES:
                a,b=[d.loc[(k,n)] for k in kinds(split)]
                loss_rows.append((NAMES[n],*[f"{100*v.recall_loss:.3f} [{100*v.loss_lo:.3f}, {100*v.loss_hi:.3f}]" for v in (a,b)],f"{100*(b.recall_loss-a.recall_loss):+.3f}"))
                stages.append((NAMES[n],*[" / ".join(f"{v[key]:.2f}" for key in ("input_ms","score_ms","route_ms","gru_ms")) for v in (a,b)]))
                ranges.append((NAMES[n],*[f"{v.min_ms:.2f}~{v.max_ms:.2f}" for v in (a,b)]))
                interval_rows.append((NAMES[n],f"{a.saved_lo:.2f}~{a.saved_hi:.2f} / {a.saved_pct:+.1f}",f"{b.saved_lo:.2f}~{b.saved_hi:.2f} / {b.saved_pct:+.1f}",f"{b.saved_pct-a.saved_pct:+.1f}"))
                baseline=[d.loc[(k,"gru")] for k in kinds(split)]
                dr=[(v.fp-base.fp,100*(v.fpr-base.fpr)) for v,base in zip((a,b),baseline)]
                changes.append((NAMES[n],f"{dr[0][0]:+,d} / {dr[0][1]:+.3f}",f"{dr[1][0]:+,d} / {dr[1][1]:+.3f}",f"{dr[1][0]-dr[0][0]:+,d} / {dr[1][1]-dr[0][1]:+.3f}"))
            self.t(label(split)+" · 절약 시간의 95% 구간 (ms) / 짝측정 단축률 (%)",["구성",*AB,"단축률 차이 (%p)"],interval_rows)
            self.t(label(split)+" · 같은 GRU 대비 악성 재현율 손실과 95% 구간 (%p)",["구성",*AB,"점추정치 차이"],loss_rows)
            self.t(label(split)+" · 1,000건당 입력 / 1단계 / 분기·재검사 / GRU 중앙값 (ms)",["구성",*AB],stages,[3.5,6.25,6.25])
            self.t(label(split)+" · 1,000건 전체 시간 30회의 최소–최대 (ms)",["구성",*AB],ranges,[3.5,6.25,6.25])
            self.t(label(split)+" · 같은 GRU 대비 정상 오탐 변화 (건 / %p)",["구성",*AB,"변화량 차이"],changes)
            self.t(label(split)+" · 95% 구간을 적용한 구성별 판정",["판정",*AB],ev.winning_names(split),[3.7,6.15,6.15])
            self.p(ev.summary[1 if split=="random" else 3])
            pair=csv("configuration-pairs").set_index("split").loc[split]
            family="HGB" if pair.family=="boost" else "LR"
            self.p(f"원저자 전처리 기준에서 {family} 경계 재검사와 기본 분기를 같은 회차끼리 비교했다. 경계 시간에서 기본 시간을 뺀 차이의 중앙값은 {pair.boundary_minus_basic_ms:+.2f} ms, 95% 구간은 {pair.lo_ms:+.2f}~{pair.hi_ms:+.2f} ms였다. 두 정책 사이의 속도 우위는 유보했다.")
            if split=="random":
                slow,fast=[d.loc[(kinds(split)[0],name)] for name in ("logistic_boundary","boost_boundary")]
                self.p(f"원저자 전처리 기준에서 HGB 경계 재검사는 {fast.total_ms:.2f} ms에 추가 미탐 {num(fast.additional_misses)}건, LR 경계 재검사는 {slow.total_ms:.2f} ms에 추가 미탐 {num(slow.additional_misses)}건이었다. HGB 경계의 처리 시간 중앙값이 {slow.total_ms-fast.total_ms:.2f} ms 짧았지만 전체 시험에서는 악성을 {num(fast.additional_misses-slow.additional_misses)}건 더 놓쳤다.")
            else:
                fast,slow=[d.loc[(kinds(split)[0],name)] for name in ("logistic_boundary","boost_boundary")]
                self.p(f"원저자 전처리 기준에서 LR 경계는 {fast.total_ms:.2f} ms에 추가 미탐 {num(fast.additional_misses)}건, HGB 경계는 {slow.total_ms:.2f} ms에 {num(slow.additional_misses)}건이었다. 관측 중앙값이 {slow.total_ms-fast.total_ms:.2f} ms 짧았던 LR 경계가 악성은 {num(fast.additional_misses-slow.additional_misses)}건 더 놓쳤다.")
            self.img(self.out/(split+"-break-even.png"),f"그림 {ix}. {label(split)} 분할의 기본 분기. 위는 1,000건의 시간, 아래는 전체 시험의 손실이다. 각 점은 한 임계값의 결과이며 선은 GRU 호출 감소율 순으로 연결했다. 세로선은 95% 구간이다."+(" ×는 손실 상한을 추정할 수 없는 점이다." if split=="domain" else ""))
            self.sweep_text(split)
        self.sub("1단계와 재검사의 보조 실험")
        s1=csv("stage1").set_index(["kind","family"]);abl=csv("ablation").set_index(["kind","family"])
        seed=csv("seeds");learning=csv("learning")
        for split in ("random","domain"):
            rows=[]
            for fam in ("logistic","boost"):
                ks=kinds(split); name="LR" if fam=="logistic" else "HGB"
                for text,vals,sc in (("1단계 단독 AP",[s1.loc[(k,fam),"pr_auc_ap"] for k in ks],1),
                                     ("9열 제외 후 AP",[abl.loc[(k,fam),"pr_auc_ap"] for k in ks],1),
                                     ("경계 재검사로 되찾은 악성 (건)",[ev.row(k,"full",fam+"_boundary").recovered_malicious for k in ks],1),
                                     ("무작위 30시드 평균 복구 (건)",[seed[(seed.kind==k)&(seed.family==fam)].recovered_malicious.mean() for k in ks],1)):
                    a,b=vals;digits=0 if text.startswith("경계 재검사") else 3
                    rows.append((name+" · "+text,f"{a:.{digits}f}",f"{b:.{digits}f}",f"{b-a:+.{digits}f}"))
                for f in (.1,1.):
                    v=[learning[(learning.kind==k)&(learning.family==fam)&(learning.fraction==f)].validation_ap.item() for k in ks]
                    rows.append((f"{name} · 학습 {f:.0%} 검증 AP",f"{v[0]:.3f}",f"{v[1]:.3f}",f"{v[1]-v[0]:+.3f}"))
            self.t(label(split)+" · 1단계와 재검사 진단",["평가 대상",*AB,"차이"],rows)
            for k in kinds(split):
                gains={f:abl.loc[(k,f),"pr_auc_ap"]-s1.loc[(k,f),"pr_auc_ap"] for f in ("logistic","boost")}
                selected=seed[(seed.kind==k)&(seed.family=="boost")].recovered_malicious
                recovered=ev.row(k,"full","boost_boundary").recovered_malicious
                self.p(f"{label(split)} 분할의 {proc(k)}에서 9개 열을 제외했을 때 1단계 시험 AP 변화는 LR {gains['logistic']:+.3f}, HGB {gains['boost']:+.3f}였다. 같은 HGB 종료 후보에서 무작위 재검사가 되찾은 악성은 시드 30개에 걸쳐 {num(selected.min())}~{num(selected.max())}건이었다. 경계 재검사로 되찾은 수는 {num(recovered)}건이었다.")
        self.p("AP는 악성 점수 순위에 따른 평균 정밀도다. 접두어·길이·파싱에 민감한 9개 열을 빼고 첫 단계부터 재적합했다. 학습량 곡선은 선택한 1단계 모델을 학습 표본의 10·25·50·75·100%에서 다시 적합한 결과다. 보조표에는 각 시드, 학습량 및 임계값·재검사율 조합별 측정값을 남겼다.")
        recheck_tables.outcomes(self)
        self.sub("추가 실험 ① 접두어만 제거한 고정 모델")
        fixed=csv("prefix-fixed")
        for split in ("random","domain"):
            rows=[]
            for flag,title in (("with_prefix","접두어 있음"),("without_prefix","접두어 없음")):
                for name,field,lab in (("gru","rows","전체"),("gru","token_changes","토큰 변화"),("gru","numeric_changes","수치 변화"),*[(n,"changed",NAMES[n]+" 판정 변화") for n in NAMES]):
                    vals=[fixed[(fixed.kind==k)&(fixed.group==flag)&(fixed.name==name)][field].item() for k in kinds(split)]
                    cells=[num(v) for v in vals]
                    if field=="changed":
                        for i,k in enumerate(kinds(split)):
                            v=fixed[(fixed.kind==k)&(fixed.group==flag)&(fixed.name==name)].iloc[0]
                            cells[i]+=f" ({num(v.malicious_to_benign)} / {num(v.benign_to_malicious)})"
                    rows.append((title+" · "+lab,*cells,num(vals[1]-vals[0])))
            self.t(label(split)+" · 고정 모델에서 표기 변경 전후 행·입력·판정 건수",["평가 대상",*AB,"차이 (건)"],rows)
        self.p("맨 앞의 HTTP(S) 접두어가 있는 경우에만 한 번 제거했다. 시험 행을 다시 정제하거나 버리지 않고 기존 어휘·가중치·보정기·임계값으로 두 입력을 만들었다. 무작위·경계 정책에서는 배치 전체의 후보와 예산이 달라지면 입력이 같은 개별 행의 경로도 달라질 수 있다.")
        self.p("판정 변화 행은 총건수 뒤 괄호에 악성 판정→정상 판정 / 정상 판정→악성 판정의 건수를 표시했다. 이 방향은 모델 예측의 변화이며, 실제 악성에 대한 추가 미탐을 세려면 정답 라벨도 함께 확인해야 한다.")
        self.sub("추가 실험 ② 접두어를 제거해 다시 학습한 모델")
        for split in ("random","domain"):
            rows=[];intervals=[]
            for n in NAMES:
                vals=[]
                for k in kinds(split):
                    alt=k.replace("original","stripped")
                    orig,st=[ev.row(a,"prefix_common",n) for a in (k,alt)]
                    pair=pairs.loc[("remove_prefix",k,n)]
                    vals.append((100*(st.recall-orig.recall),pair.delta_ms))
                    if k==kinds(split)[0]:intervals.append([NAMES[n]])
                    intervals[-1].append(f"{pair.lo_ms:+.2f}~{pair.hi_ms:+.2f} · {pair.direction}")
                (a,c),(b,d)=vals
                rows.append((NAMES[n],f"{a:+.2f} / {c:+.2f}",f"{b:+.2f} / {d:+.2f}",f"{b-a:+.2f} / {d-c:+.2f}"))
            a,b=[ev.row(k,"prefix_common","gru") for k in kinds(split)]
            self.t(label(split)+" · 재학습 시 원래 표기 대비 재현율 변화 (%p) / 시간 변화 (ms)",["구성",*AB,"차이"],rows)
            self.t(label(split)+" · 접두어 제거 재학습의 시간 변화 95% 구간 (ms)",["구성",*AB],intervals,[4.4,5.8,5.8])
            self.p(f"탐지 성능은 원래 표기와 제거 표기에 모두 남은 원저자 기준 {num(a.n)}건, 팀 기준 {num(b.n)}건에서 비교했다. 시간은 네 조건이 공유하는 동일 1,000건에서 측정했다. 양수인 시간 변화는 재학습 조건이 더 오래 걸렸다는 뜻이다.")
        self.chapter(6)
        self.p("학습 직전에는 입력 행·토큰·수치·어휘의 해시를 기록했다. 당시 증거에 실제 역할 배정 코드와 두 정제 절차 코드의 해시는 들어 있지 않았다. 실험 후 두 코드와 배정표의 해시를 추가하고 남은 URL의 역할과 배열을 재검사했다. 사후 해시는 학습 당시 코드의 정확한 버전까지 소급해 입증하지 못한다.")
        self.p("학습·검증과 공통 시험에 같은 원본 URL은 없었지만, 서로 다른 주소도 같은 50칸 토큰과 18개 수치로 표현될 수 있었다. 이 입력 중복은 원저자 전처리 기준에 남았다. 따라서 두 정제 절차의 재현율 차이는 데이터 구성과 표현이 함께 바뀐 결과다.")
        self.p("팀 정제에서는 정규화 주소나 모델 입력이 같은 그룹의 라벨 충돌을 모든 역할에 걸쳐 확인했다. 학습 어휘는 학습 행으로만 만들었지만 어떤 행이 평가에 남는지는 이 충돌 검사에 영향을 받는다. 공통 시험 결과도 두 절차에서 모두 살아남은 주소에 대한 조건부 비교다.")
        self.p("공개 라벨 가운데 규칙으로 찾을 수 있는 충돌을 처리했지만 남은 주소의 실제 위험 여부를 외부 기관이나 사람에게 재확인하지 못했다. 원저자의 별도 CSV와 내부 자료 사이에는 원본 주소 중복이 있고, 팀은 과거에도 이 공개 코퍼스의 시험 결과를 확인했다. 새로 학습한 모델이라도 완전히 새로운 환경에서의 맹검 성능을 보장할 수 없다 [5, 6].")
        self.p("도메인을 나누면 동일 등록 도메인을 기억해서 얻는 이점을 줄일 수 있다. 다만 서로 다른 도메인이 같은 캠페인에 속할 가능성과 수집 시점의 차이는 남는다. 검증 자료를 모델 선택·보정·임계값 선택에 함께 사용했다. 한 번 고정한 분할의 구간은 모델을 다른 시드로 처음부터 다시 학습했을 때의 변동까지 포함하지 않는다.")
        self.p("30회 시간 측정은 한 CPU에서 같은 1,000개 URL을 배치로 처리한 결과다. 단건 요청, 동시 사용자, GPU 추론의 시간은 별도로 측정해야 한다. 임계값별 95% 구간은 개별 비교용이며 여러 임계값을 동시에 보정한 구간이 아니다. 접두어와 라벨의 관계도 수집 자료의 표기 관행에 따라 달라질 수 있다. 다음 독립 시험에서는 이후 시점의 데이터를 확보해 등록 도메인과 수집 출처를 함께 나누는 방법이 적합하다.")
        self.chapter(7)
        for i in (1,3):self.p(ev.summary[i])
        self.p("같은 조건에서 시간을 더 줄이는 구성과 악성을 덜 놓치는 구성이 다를 수 있다. Latotzke 외의 정확도·비용 비교처럼 [10], 두 지표를 함께 읽어야 한다. 예상 피해를 하나의 값으로 비교하려면 유형별 오분류 비용이 필요하다 [11]. 이번 데이터에는 피해 비용이 없어 화폐 단위 순위를 계산할 근거가 없다.")
        self.p("정상 오탐의 허용 범위는 팀이 운영 목적과 피해 비용을 확인한 뒤 결정할 사항이다. 각 분할에서 가장 빠른 구성은 두 정제 절차에 적용한 총 12개 캐스케이드 중 해당 GRU 대비 시간·미탐 조건을 함께 충족한 구성의 1,000건 처리 시간 중앙값으로 정했다.")
        self.sub("참고문헌")
        self.p("발표 별첨 ‘악성 URL 탐지 시연 v1.html’에서 실제 시험 URL 네 건의 판정 경로를 다시 볼 수 있다.")
        for i,r in enumerate(REFS,1):self.p(f"[{i}] {r}")

    def sweep_text(self,split):
        sweep=csv("threshold-decisions")
        rows=[]
        for fam in ("logistic","boost"):
            for cond,desc in (("time","시간 절약"),("loss","손실 허용"),("both","두 조건 동시")):
                vals=[]
                for k in kinds(split):
                    z=sweep[(sweep.kind==k)&(sweep.family==fam)&(sweep.policy=="basic")]
                    a=z.saved_lo>0;b=z.loss_hi<.01
                    g=z[a if cond=="time" else b if cond=="loss" else (a&b)]
                    values=100*(1-g.calls/1000)
                    vals.append("없음" if g.empty else f"{values.min():.1f}–{values.max():.1f} ({len(g)}점)")
                rows.append((("LR" if fam=="logistic" else "HGB")+" · "+desc,*vals))
        self.t(label(split)+" · 기준을 충족한 측정점의 최소–최대 호출 감소율 (%)",["구간",*AB],rows,[4.6,5.7,5.7])
        self.p("범위 안에서도 개별 임계값의 판정이 다를 수 있다. 표의 범위는 측정한 점들의 양 끝이며 연속적인 모든 호출률에서 같은 효과를 보장하지 않는다.")

if __name__=="__main__": Report().main()

