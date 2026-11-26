"""Select four observed, offline examples after the decisions are sealed."""
import re
import hashlib
from inputs import ROOT, pd, np
from evaluate import route
from aggregate import load_test
NAMES={"gru":"GRU 단독","logistic_basic":"LR 기본","logistic_boundary":"LR 경계 5%","logistic_random":"LR 무작위 5%","boost_basic":"HGB 기본","boost_boundary":"HGB 경계 5%","boost_random":"HGB 무작위 5%"}


def main():
    dest=ROOT/"results"
    t=pd.read_csv(dest/"main-28.csv")
    cascade=t[t.family!="gru"]
    passed=cascade[cascade.decision=="충족·단축"]
    best=(passed if len(passed) else cascade).sort_values("total_ms").iloc[0]
    kind=best.kind
    local=cascade[cascade.kind==kind]
    worst=local[local.name!=best["name"]].sort_values(["total_ms"],ascending=False).iloc[0]
    options=[best["name"],worst["name"]]
    witness=local[local.policy=="boundary"].sort_values(["recovered_malicious","rechecks"],ascending=False).iloc[0]["name"]
    configs=list(dict.fromkeys(["gru",*options,witness]))
    d,s,taus=load_test(kind)
    y=s["y"].astype(bool); detected=s["gru"]>.5
    paths={}; masks={}; first={}
    for name in configs:
        if name=="gru":
            masks[name]=np.ones(len(y),bool); paths[name]=np.full(len(y),"전체 GRU"); continue
        fam,pol=name.split("_")
        first[name]=s[fam]
        direct=s[fam]>=taus[fam]
        masks[name]=route(s[fam],s["id"],taus[fam],0 if pol=="basic" else .05,pol)[0]
        paths[name]=np.where(direct,"즉시 GRU",np.where(masks[name],"재검사 → GRU","정상 조기 종료"))
    tests=[("정상 조기 종료", (~y)&(~masks[witness])),("즉시 GRU 전달", paths[witness]=="즉시 GRU"),("종료 후보 재검사", paths[witness]=="재검사 → GRU"),("재검사로 되찾은 악성", (paths[witness]=="재검사 → GRU")&y&detected)]
    if not tests[-1][1].any():
        tests[-1]=("조기 종료로 놓친 악성", (~masks[witness])&y&detected)
    rows=[]; used=set()
    for category,selected in tests:
        indices=[i for i in np.flatnonzero(selected) if i not in used]
        safe=[i for i in indices if re.fullmatch(r"[A-Za-z0-9./:_~-]{3,130}",d.url.iloc[i]) and "@" not in d.url.iloc[i] and not re.search(r"\.(exe|js|bat|sh|zip|apk|php)($|/)",d.url.iloc[i],re.I)]
        if not safe: raise ValueError("No displayable historical example: "+category)
        i=min(safe,key=lambda j:(len(d.url.iloc[j]),s["id"][j]))
        used.add(i)
        u=d.url.iloc[i].replace("https://","hxxps[:]//").replace("http://","hxxp[:]//").replace(".","[.]")
        for name in configs:
            rows.append(dict(case=category,kind=kind,source_id=s["id"][i],url_sha256=hashlib.sha256(d.url.iloc[i].encode()).hexdigest(),url_display=u,truth="악성" if y[i] else "정상",configuration=NAMES[name],stage1=None if name=="gru" else first[name][i],threshold=None if name=="gru" else taus[name.split("_")[0]],path=paths[name][i],gru_malicious_score=s["gru"][i] if masks[name][i] else None,verdict="악성" if detected[i] and masks[name][i] else "정상"))
    for row in rows:row["case_route_configuration"]=NAMES[witness]
    pd.DataFrame(rows).to_csv(dest/"시연 사례 v1.csv",index=False,encoding="utf-8-sig")
    lines=["# 악성 URL 탐지 시연 v1","",f"조건 {kind}. 같은 시험 파일의 네 주소를 오프라인 기록으로 재생한다. 주소는 클릭할 수 없게 표시했다. 악성·정상은 공개 자료 라벨이다.",
           f"시간·미탐 판정에서 선택한 구성은 {NAMES[options[0]]}({best.decision}), 같은 조건에서 처리 시간이 가장 길었던 비교 구성은 {NAMES[options[1]]}이다. 사례 제목은 {NAMES[witness]}의 경로를 가리킨다.",""]
    for case,df in pd.DataFrame(rows).groupby("case",sort=False):
        lines += ["## "+case,"",df.iloc[0].url_display+" · 정답 "+df.iloc[0].truth,"","| 구성 | 1단계 악성 점수 | 임계값 | 경로 | 최종 판정 |","|---|---:|---:|---|---|"]
        for r in df.itertuples():
            sc="—" if pd.isna(r.stage1) else f"{r.stage1:.6f}"
            tau="—" if pd.isna(r.threshold) else f"{r.threshold:.3f}"
            lines.append(f"| {r.configuration} | {sc} | {tau} | {r.path} | {r.verdict} |")
        lines.append("")
    (dest/"시연 자료 v1.md").write_text("\n".join(lines),encoding="utf8")


if __name__=="__main__":
    main()
