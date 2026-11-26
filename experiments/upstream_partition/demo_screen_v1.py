"""Produce four offline filming frames directly from the replayed case records."""
import csv
import html
import os
from itertools import groupby
from pathlib import Path

ROOT=Path(os.environ["URLCASCADE_RUN_DIR"])
RESULT=ROOT/"results"
STYLE="""
*{box-sizing:border-box}html{background:#eee;color:#141414;font-family:'Malgun Gothic',sans-serif}
body{max-width:1100px;margin:auto;padding:25px}header,nav,.meta{display:flex;align-items:center;justify-content:space-between;gap:16px}
header{border-bottom:2px solid #171717;margin-bottom:18px}header p{font-size:14px}h1{font-size:22px}nav{justify-content:flex-start;flex-wrap:wrap;margin:20px 0}
button{background:white;border:1px solid #888;padding:9px 15px;cursor:pointer;font:inherit}button[aria-current=true]{background:#181818;color:white}
article{background:white;padding:26px;border:1px solid #bbb}h2{font-size:29px;margin:0 0 10px}.address{font-size:25px;overflow-wrap:anywhere;font-family:Consolas,monospace}
table{margin:22px 0 10px;width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #ccc;overflow-wrap:anywhere}
thead{background:#eee}thead small{display:block;font-weight:normal}.note,footer{font-size:14px;color:#424242}footer{margin-top:18px}
@media print{nav{display:none}article[hidden]{display:block}article{break-inside:avoid;page-break-after:always}}"""

def main():
    with (RESULT/"시연 사례 v1.csv").open(encoding="utf-8-sig",newline="") as source:
        rows=list(csv.DictReader(source))
    cases=[(k,list(g)) for k,g in groupby(rows,key=lambda r:r["case"])]
    assert len(cases)==4
    cards=[]; buttons=[]
    for i,(case,group) in enumerate(cases):
        first=group[0]; assert all((r["truth"],r["source_id"])==(first["truth"],first["source_id"]) for r in group)
        esc=html.escape
        buttons.append(f'<button data-i="{i}" aria-current="{str(i==0).lower()}">{esc(case)}</button>')
        roles=["기준 구성","미탐 허용치 충족·처리 시간 최저","같은 조건에서 가장 오래 걸린 캐스케이드","재검사 사례"]
        head="".join(f'<th>{esc(r["configuration"])}<small>{roles[j]}</small></th>' for j,r in enumerate(group))
        def values(key):
            vals=[r[key] for r in group]
            if key in {"stage1","threshold","gru_malicious_score"}:vals=[f"{float(v):.6f}" if v else "—" for v in vals]
            return "".join("<td>"+esc(v)+"</td>" for v in vals)
        fields=[("정답","truth"),("1단계 악성 점수","stage1"),("분기 임계값","threshold"),("이동 경로","path"),("GRU 악성 점수","gru_malicious_score"),("최종 판정","verdict")]
        body="".join(f"<tr><th>{title}</th>{values(key)}</tr>" for title,key in fields)
        title_kind=("등록 도메인" if "domain" in first["kind"] else "URL 무작위")+" 분할 · "+("팀 정제 기준" if first["kind"].startswith("team-") else "원저자 전처리 기준")
        cards.append(f'<article id="case{i}"'+(" hidden" if i else "")+f'><h2>{esc(case)}</h2><div class="meta"><p class="address">{esc(first["url_display"])}</p></div><p>{title_kind} · 사례 경로를 정한 구성은 {esc(first["case_route_configuration"])}. 정답은 공개 CSV 라벨이다.</p><table><thead><tr><th>기록 항목</th>{head}</tr></thead><tbody>{body}</tbody></table><p class="note">전체 시험 배치에서 기록한 경로다. 재검사는 정상 종료 후보의 5%이며 무작위 재검사의 시드는 0이다. 대시는 해당 단계를 실행하지 않았음을 뜻한다.</p></article>')
    script="""const nav=[...document.querySelectorAll('nav button')];function show(i){nav.forEach((b,j)=>{b.setAttribute('aria-current',String(i===j));document.getElementById('case'+j).hidden=i!==j})}nav.forEach((b,i)=>b.onclick=()=>show(i));document.onkeydown=e=>{let i=nav.findIndex(b=>b.getAttribute('aria-current')==='true');if(e.key==='ArrowRight')show(Math.min(i+1,3));if(e.key==='ArrowLeft')show(Math.max(i-1,0))};"""
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>악성 URL 탐지 시연 v1</title><style>'+STYLE+'</style><header><h1>악성 URL 탐지 시연 v1</h1><p>최종 시험 기록 재생</p></header><nav aria-label="시연 사례">'+''.join(buttons)+'</nav>'+''.join(cards)+'<footer>좌우 방향키로 사례를 전환한다. 주소는 접속되지 않는 표기로 표시했다.</footer><script>'+script+'</script></html>'
    target=RESULT/"악성 URL 탐지 시연 v1.html"
    target.write_text(page,encoding="utf8")
    assert page.count("<article")==4 and page.count("<table>")==4 and "href=" not in page
    for r in rows:assert html.escape(r["url_display"]) in page and html.escape(r["path"]) in page
    print("Verified four case tables for",len(rows),"recorded routes")

if __name__=="__main__":main()
