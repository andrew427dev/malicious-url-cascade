"""Produce a compact shareable evidence archive and Notion text from the same verified tables."""
from pathlib import Path
import hashlib
from zipfile import ZipFile,ZIP_DEFLATED
import json
import shutil
import html
from report_data import *

GUIDE="""# 악성 URL 탐지 최종 실험 v1 재실행
이 묶음의 results에는 전체 시험, 공통 시험, 접두어 재학습 공통 시험을 구분한 결과와 보조 실험이 있다. main-28.csv는 원래 표기 주 실험 28개 결과다. threshold-decisions.csv에는 실제로 측정한 검증 임계값 후보와 각 구간이 들어 있다. recall, fpr, recall_loss는 0~1 비율이다. 100을 곱하면 % 또는 %p다. additional_misses는 GRU 단독에서 잡았지만 조기 종료로 놓친 악성 수다. 긍정 클래스는 악성이다.
 
## 시간 측정 원본
timing-random.csv와 timing-domain.csv에 1,000건당 회차·실행 순서·조건·실제 GRU 호출 수와 단계별 밀리초가 기록돼 있다. 같은 split과 repeat의 해당 kind GRU 시간을 기준으로 짝지어 비교한다. batch-*.json에는 공유 표본의 순서가 있다. 파일 로딩·어휘 생성·예열·정합성 단언은 시간에서 제외했다.
 
## 재실행
64비트 Python 3.10 및 requirements-experiment.txt에 고정한 라이브러리를 사용한다. 운영체제와 라이브러리 차이는 학습 및 시간에 영향을 미친다. 압축을 푼 폴더에서 아래 명령을 실행한다. 출력 대상은 새 경로여야 한다.
 
    python -m pip install -r requirements-experiment.txt
    python code/experiments/upstream_partition/rerun_v1.py NEW_RUN
 
실행기는 공개 CSV 세 파일·노트북·Utils를 다시 받아 해시를 대조한 뒤 입력 동등성 검사, 역할 배정, 새 어휘와 GRU 학습, 1단계 선택·보정·분기 고정, 전체·공통 평가, 예열과 30회 시간 측정 및 감사를 수행한다. 원저자 재현의 내부/별도 평가 모델과 전체 학습 모델, 원래 표기 4개와 접두어 제거 재학습 4개가 각각 따로 학습된다. 데이터의 URL은 텍스트로만 처리한다.
 
## 보고서 재생성
측정 환경과 별도로 python-docx, Pillow, pandas, matplotlib, 한글 글꼴 맑은 고딕과 Windows Word가 필요하다. URLCASCADE_RUN_DIR 환경 변수에 새 결과 폴더 절대 경로를 지정한 뒤 figure_v1.py, report_v1.py 순으로 실행한다. render_word.ps1 -Docx에 생성된 Word 파일을 주면 필드와 쪽 번호를 갱신하고 PDF로 변환한다. PDF와 Word의 표·그림은 다시 열어 확인한다.
 
## 로컬 실행 증거
조건별 폴더의 training-inputs.json과 trained.json은 학습에 쓴 행 순서·토큰·수치·어휘 및 가중치 해시를 연결한다. scores.npz와 paired-scores.npz는 행별 점수, routing.json은 검증 임계값, stage1은 모델 선택·보정 증거다. 전체 시험 성능과 고정 1,000건 배치 시간은 분모가 다르다. 외부 페이지를 열어 주소의 현재 상태를 확인하지 않았다.

## 이번 실행과 재실행 순서
이번 실행에서는 앞 두 CSV를 사용한 모델과 세 CSV 전체 모델의 학습을 병행했다. 별도 CSV 평가는 앞 두 파일로만 학습한 체크포인트로 수행했다. 재실행 진입점은 같은 두 작업을 순서대로 수행한다. results/review-evidence.json에 당시 파일 기록 시각과 가중치 해시가 있다. 당시 학습 전 기록에서 빠졌던 역할 배정·정제 코드 두 개의 해시는 사후 확인으로 표시했다. 새 재실행은 이 두 코드와 역할 배정표도 학습 시작 전에 기록한다.

## 같은 회차의 시간 비교
results/timing-pairs.csv는 같은 1,000개 URL의 회차별 시간 차이를 먼저 구한다. 정제 변경은 팀−원저자, 접두어 변경은 제거 표기−원래 표기이며 양수는 더 오래 걸렸다는 뜻이다. 30개 차이의 중앙값과 짝 부트스트랩 95% 구간을 기록했다. results/configuration-pairs.csv는 두 분할에서 선택된 경계 재검사와 같은 모델의 기본 분기 사이의 짝차이다. 도메인 재표집으로 손실 상한을 추정할 수 없는 경우 threshold-decisions.csv의 loss_hi는 비어 있고 판정은 유보다.
"""
def sha_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    e=Evidence()
    out=ROOT/"report"
    notes=RESULT/"재실행 안내 v1.md"
    notes.write_text(GUIDE,encoding="utf8")
    r=ROOT/"code"
    code=[]
    for group,names in {
        "experiments/upstream_partition":"fetch inputs train_gru team_data provenance assign_roles cohorts fit_stages evaluate run_comparison score_v1 paired_time aggregate audit_v1 review_evidence_v1 finish_v1 verify_results demo_v1 demo_screen_v1 rerun_v1 report_data report_v1 recheck_tables make_report report_content table_layout diagram feature_table figure_v1 publish_v1 check_report_v1 archive_inputs_v1 share_models_v1",
        "experiments/clean_rerun":"common prepare experiment",
        "experiments/scheme_ablation":"run verify_inputs",
        "urlcascade":"__init__ adapters features"}.items():
        code += [r/group/(s+".py") for s in names.split()]
    code += [r/"configs/upstream.json",r/"experiments/upstream_partition/nov23-format.docx",r/"experiments/upstream_partition/render_word.ps1"]
    items={str(p.relative_to(ROOT)).replace("\\","/"):p for p in code}
    for p in [*RESULT.glob("*"),*ROOT.glob("timing-*.csv"),*ROOT.glob("batch-*.json"),ROOT/"protocol-v1.json",ROOT/"source-manifest.json",ROOT/"source-tree.json",ROOT/"parity.json",ROOT/"common-audit.json",ROOT/"feature-example.json"]:
        if p.is_file():items[p.relative_to(ROOT).as_posix()]=p
    for k in [x for x in ROOT.iterdir() if x.is_dir() and (x/"trained.json").exists()]:
        for n in ("cleaning.json","data.json","routing.json","training-inputs.json","trained.json","epochs.csv","results.json","source-stages.json","stage1/t1-selected.json"):
            p=k/n
            if p.is_file():items[p.relative_to(ROOT).as_posix()]=p
    archive=out/"결과표·측정 원본·재실행 자료 v1.zip"
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=8) as z:
        for name,path in sorted(items.items()):z.write(path,name)
        z.writestr("README.md",GUIDE)
        z.writestr("requirements-experiment.txt","tensorflow==2.12.0\nnumpy==1.23.5\npandas==1.5.3\nscikit-learn==1.2.2\nscipy==1.15.3\ntldextract==5.4.0\njoblib==1.6.0\nh5py==3.16.0\n")
        z.writestr("SHA256.json",json.dumps({k:sha_file(v) for k,v in items.items()},ensure_ascii=False,indent=2))
    with ZipFile(archive) as z:
        assert z.testzip() is None
    texts=["공개 CSV 세 파일에서 URL별 학습·검증·시험 역할을 고정하고 두 정제 절차로 GRU를 각각 새로 학습했다. GRU 단독과 로지스틱 회귀(LR)·히스토그램 기반 그래디언트 부스팅(HGB)의 기본 분기, 정상 종료 후보 중 무작위·경계 5% 재검사를 비교했다. 허용치는 같은 GRU 대비 실제 악성 100건당 추가 미탐 1건(재현율 손실 1%p)이다.",*e.summary]
    texts[0]+=" 절약 시간의 95% 구간 하한이 0보다 크고 재현율 손실의 95% 구간 상한이 1%p보다 작을 때 두 조건을 충족한 것으로 판정했다."
    rows=[["분할 · 공통 시험 건수","지표",*AB,"팀−원저자"]]
    for s,split in enumerate(("random","domain")):
        a,b=kinds(split)
        x,y=[e.row(k,"common","gru") for k in (a,b)]
        for key,lab in (("recall","악성 재현율 (%)"),("accuracy","정확도 (%)")):
            rows.append([label(split)+" · "+num(x.n),lab,pct(x[key]),pct(y[key]),delta(100*(y[key]-x[key]))+"%p"])
    policy_rows=[["분할 · 구성",*AB,"팀−원저자 · 손실 (%p) / 절약 (ms)"]]
    for split in ("random","domain"):
        for n in list(NAMES)[1:]:
            records=[e.main[(e.main.kind==k)&(e.main.name==n)].iloc[0] for k in kinds(split)]
            cells=[]
            for v in records:
                loss="충족" if v.loss_hi<.01 else "초과" if v.loss_lo>.01 else "유보"
                cells.append(f"{loss} · {int(v.additional_misses):,}/{int(v.malicious):,}건 · {100*v.recall_loss:.3f}%p · {v.saved_ms:+.2f} ms")
            a,b=records
            policy_rows.append([label(split)+" · "+NAMES[n],*cells,f"{100*(b.recall_loss-a.recall_loss):+.3f} / {b.saved_ms-a.saved_ms:+.2f}"])
    parts=["<p>"+html.escape(t)+"</p>" for t in texts]
    def html_table(matrix):
        return "<table>"+"".join("<tr>"+"".join("<td>"+html.escape(str(c))+"</td>" for c in row)+"</tr>" for row in matrix)+"</table>"
    parts.append(html_table(rows))
    pair=csv("configuration-pairs").set_index("split")
    limits=" · ".join(f"{label(s)} {pair.loc[s,'lo_ms']:+.2f}~{pair.loc[s,'hi_ms']:+.2f} ms" for s in ("random","domain"))
    note="원저자 전처리 기준에서 관측 시간이 가장 짧았던 경계 재검사와 같은 모델의 기본 분기를 직접 짝지으면 시간 차이의 95% 구간은 "+limits+"였다. 두 구간 모두 0을 포함해 경계·기본 사이의 속도 우위는 유보했다."
    parts.append("<p>"+html.escape(note)+"</p>")
    parts.append("<p>미탐 허용치 판정 · 전체 시험의 추가 미탐/실제 악성 건수 · 같은 GRU 대비 재현율 손실(%p) · 공통 URL 1,000건 × 30회에서 절약한 시간(ms). 양수이면 단축.</p>")
    parts.append(html_table(policy_rows))
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = Document(out / "악성 URL 탐지 최종 실험 보고서 v1.docx")
    additions = ["<h2>분기 임계값과 재검사율별 결과</h2>",
                 "<p>원래 표기의 전체 시험에서 검증으로 정한 임계값을 고정했다. 비율별 표의 값은 재검사한 URL, 기본 분기에서 되찾은 악성, GRU 단독보다 더 놓친 악성, 기본 분기보다 새로 오탐한 정상 URL의 건수 순이다. 무작위 결과는 시드 0이다.</p>"]
    caption = ""
    selected_tables = []
    for element in document.element.body:
        if element.tag.endswith("}p"):
            caption = Paragraph(element, document).text
        elif element.tag.endswith("}tbl") and any(tag in caption for tag in (
                "재검사 예산을 정한 방법", "검증에서 확정한", "5% 경계 선정 점수",
                "재검사율별 결과 (건)", "탐지 평가와 시간 실측 범위")):
            matrix = [[cell.text for cell in row.cells] for row in Table(element, document).rows]
            additions.extend(["<p>"+html.escape(caption)+"</p>", html_table(matrix)])
            selected_tables.append({"caption":caption,"rows":matrix})
    assert len(selected_tables) == 11
    parts.extend(additions)
    (ROOT/"layout-qa"/"rate-tables.html").write_text('<meta charset="utf-8">'+"".join(additions), encoding="utf8")
    (out/"notion-body.html").write_text("".join(parts),encoding="utf8")
    (out/"notion-content.json").write_text(json.dumps(dict(paragraphs=texts,table=rows,policy_table=policy_rows,policy_comparison_note=note,recheck_tables=selected_tables),ensure_ascii=False,indent=2),encoding="utf8")
if __name__=="__main__":main()
