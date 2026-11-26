"""Plot paired median time savings and full-test recall loss on a common call-savings axis."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT=Path(os.environ["URLCASCADE_RUN_DIR"])
font=font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10})
def main():
    all_rows=pd.read_csv(ROOT/"results/threshold-decisions.csv")
    dest=ROOT/"report"
    dest.mkdir(exist_ok=True)
    for split in ("random","domain"):
        fig,axes=plt.subplots(2,2,figsize=(9.1,6.4),sharex="col",layout="constrained")
        for j,(kind,title) in enumerate(((split+"-original","원저자 전처리 기준"),("team-"+split+"-original","팀 정제 기준"))):
            for fam,style,marker,label in (("logistic","#202020","o","LR"),("boost","#808080","s","HGB")):
                r=all_rows[(all_rows.kind==kind)&(all_rows.family==fam)&(all_rows.policy=="basic")].sort_values("calls",ascending=False)
                x=100*(1-r.calls.to_numpy()/1000)
                for ax,mid,lo,hi,scale in ((axes[0,j],"saved_ms","saved_lo","saved_hi",1),(axes[1,j],"recall_loss","loss_lo","loss_hi",100)):
                    y=r[mid].to_numpy()*scale
                    ax.plot(x,y,marker=marker,color=style,markersize=4,linewidth=1.2,label=label)
                    ax.vlines(x,r[lo]*scale,r[hi]*scale,color=style,alpha=.85,linewidth=1.4)
                    missing=r[hi].isna().to_numpy()
                    if missing.any():
                        ax.scatter(x[missing],y[missing],marker="x",s=72,color="black",zorder=5)
                    ax.grid(axis="y",color="#dddddd",lw=.6)
            axes[0,j].set_title(title)
            axes[0,j].axhline(0,linestyle="--",color="#222222",lw=1)
            axes[0,j].legend(loc="best",ncol=2)
            axes[1,j].axhline(1,linestyle="--",color="#222222",lw=1)
            axes[1,j].set_xlabel("1,000건 배치의 GRU 호출 감소율 (%)")
            axes[0,j].set_ylabel("절약 시간 (ms / 1,000건)")
            axes[1,j].set_ylabel("전체 시험의 악성 재현율 손실 (%p)")
        fig.savefig(dest/(split+"-break-even.png"),dpi=200)
        plt.close(fig)

if __name__=="__main__": main()
