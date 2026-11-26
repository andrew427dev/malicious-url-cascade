"""Human-facing comparisons generated from the verified experiment tables."""
from pathlib import Path
import os
import json
import pandas as pd
ROOT=Path(os.environ["URLCASCADE_RUN_DIR"])
RESULT=ROOT/"results"
def readj(path): return json.loads((ROOT/path).read_text(encoding="utf8"))
def csv(name): return pd.read_csv(RESULT/(name+".csv"))
def num(v): return f"{int(v):,}"
def pct(v): return f"{100*v:.2f}"
def delta(v): return f"{v:+.2f}"
def label(split): return "URL 무작위" if split=="random" else "등록 도메인"
def proc(k): return "팀 정제 기준" if k.startswith("team-") else "원저자 전처리 기준"
def kinds(split):return (split+"-original","team-"+split+"-original")
NAMES={"gru":"GRU 단독","logistic_basic":"LR 기본","logistic_random":"LR 무작위","logistic_boundary":"LR 경계","boost_basic":"HGB 기본","boost_random":"HGB 무작위","boost_boundary":"HGB 경계"}
AB=("원저자 전처리 기준","팀 정제 기준")
class Evidence:
    def __init__(self):
        self.all=csv("full-and-paired")
        self.main=csv("main-28")
        self.pop=csv("cohorts").set_index("kind")
        self.time=csv("timing-summary")
        self.comp=[]
        self.summary=[]
        for split in ("random","domain"):
            a,b=kinds(split)
            ref,team=[self.row(k,"common","gru") for k in (a,b)]
            self.comp.append((label(split),num(ref.n),pct(ref.recall),pct(team.recall),delta(100*(team.recall-ref.recall))))
            self.summary.append(f"{label(split)} 분할에서 두 절차가 함께 남긴 {num(ref.n)}건 중 악성은 {num(ref.malicious)}건이었다. 팀 정제 기준을 적용하면 GRU의 악성 재현율이 {pct(ref.recall)}%에서 {pct(team.recall)}%로 {delta(100*(team.recall-ref.recall))}%p 변했고, 정확도는 {pct(ref.accuracy)}%에서 {pct(team.accuracy)}%로 변했다.")
            winners=self.main[(self.main.kind.isin((a,b)))&(self.main.decision=="충족·단축")]
            if len(winners):
                win=winners.sort_values("total_ms").iloc[0]
                self.summary.append(f"{label(split)} 분할에서 12개 캐스케이드 중 {len(winners)}개가 시간·미탐 기준을 함께 충족했다. 30회 총처리 시간 중앙값이 가장 작았던 구성은 {proc(win.kind)}의 {NAMES[win['name']]}였다. 1,000건당 {win.total_ms:.2f} ms였고 같은 GRU 단독보다 짝측정 중앙값 {win.saved_ms:.2f} ms를 절약했다. 전체 시험의 실제 악성 {num(win.malicious)}건 중 {num(win.additional_misses)}건을 더 놓쳤다.")
            else:
                group=self.main[(self.main.kind.isin((a,b)))&(self.main.family!="gru")]
                row=group.sort_values("total_ms").iloc[0]
                self.summary.append(f"{label(split)} 분할에서는 12개 캐스케이드 가운데 시간 절약과 미탐 허용치를 95% 구간으로 함께 확인한 구성이 없었다. 측정 시간이 가장 짧은 {proc(row.kind)}의 {NAMES[row['name']]}는 1,000건당 {row.total_ms:.2f} ms, GRU 대비 절약 {row.saved_ms:+.2f} ms였고 실제 악성 {num(row.malicious)}건 중 {num(row.additional_misses)}건을 더 놓쳤다.")
    def row(self,k,scope,name):
        z=self.all[(self.all.kind==k)&(self.all.scope==scope)&(self.all.name==name)&self.all.main]
        assert len(z)==1
        return z.iloc[0]
    def cleaning(self,split):
        a,b=kinds(split)
        ca,cb=[readj(k+"/cleaning.json") for k in (a,b)]
        left={r["rule"]:(r["removed"]["rows"],r["remaining"]) for r in ca["stages"]}
        right={r["rule"]:(r["removed"]["rows"],r["remaining"]) for r in cb["stages"] if "removed" in r}
        tokens=[r for r in cb["stages"] if "conflict" in r]
        before=right["character_numeric_repeat"][1]
        for reason,field in (("token_conflict","conflict"),("token_repeat","repeat")):
            c=sum(r[field] for r in tokens); before-=c; right[reason]=(c,before)
        text=ca["first_text"]
        before=left["identical_raw_label_repeat"][1]
        for reason,field in (("mixed_text","mixed_text_removed"),("same_text","same_label_text_removed")):
            c=sum(r[field] for r in text); before-=c; left[reason]=(c,before)
        mapping=[("invalid_url","주소 파싱·유효성"),("raw_label_conflict","원본 라벨 충돌"),("identical_raw_label_repeat","원본·라벨 반복"),("normalized_label_conflict","정규화 주소 라벨 충돌"),("normalized_repeat","정규화 주소 반복"),("mixed_text","전체 문자 동일·혼합 라벨 그룹"),("same_text","전체 문자 동일·단일 라벨 그룹"),("unusable_input","입력 생성 불가"),("character_numeric_label_conflict","앞 52문자·수치 라벨 충돌"),("character_numeric_repeat","앞 52문자·수치 반복"),("token_conflict","토큰·수치 라벨 충돌"),("token_repeat","토큰·수치 반복"),("unstable_normalization","재정규화 불일치")]
        def fmt(pair):return "—" if pair is None else num(pair[0])+" / "+num(pair[1])
        return [(title,fmt(left.get(key)),fmt(right.get(key)),num(right[key][0]-left[key][0]) if key in left and key in right else "—") for key,title in mapping]

    def metric_rows(self,split,scope,fields,names=None):
        rows=[]
        for n in names or NAMES:
            a,b=[self.row(k,scope,n) for k in kinds(split)]
            values=[[],[],[]]
            for field,scale,digits in fields:
                v,w=a[field]*scale,b[field]*scale
                values[0].append(f"{v:,.{digits}f}"); values[1].append(f"{w:,.{digits}f}"); values[2].append(f"{w-v:+,.{digits}f}")
            rows.append([NAMES[n],*[" / ".join(v) for v in values]])
        return rows
    def winning_names(self,split):
        a,b=kinds(split)
        rows=[]
        for k in (a,b):
            d=self.main[(self.main.kind==k)&(self.main.family!="gru")]
            selections={}
            for s in ("충족·단축","유보","시간 또는 미탐 기준 미충족"):
                selections[s]=", ".join(NAMES[r] for r in d.loc[d.decision==s,"name"]) or "없음"
            rows.append(selections)
        return [[s,rows[0][s],rows[1][s]] for s in rows[0]]

