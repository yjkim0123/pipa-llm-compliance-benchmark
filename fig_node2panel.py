"""노드 분석 2패널 합본: (A) 노드별 KO/EN 덤벨, (B) 노드→outcome slope."""
import json
import numpy as np
from figstyle import KO, EN, NEG
import matplotlib.pyplot as plt

nodes = json.load(open("node_result.json", encoding="utf-8"))
R = json.load(open("results_v1.json", encoding="utf-8"))
NK = ["lawful_basis","is_sensitive","consent_ok","purpose_specified","within_purpose"]
LB = ["lawful basis","sensitivity","consent adequacy","purpose spec.","within purpose"]
g9 = "gemma2:9b-instruct-q4_0"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios":[1.35,1]})

# Panel A: dumbbell
ko=[nodes["ko"][n] for n in NK]; en=[nodes["en"][n] for n in NK]; y=np.arange(len(NK))
for i,(k,e) in enumerate(zip(ko,en)):
    axA.plot([k,e],[i,i],color="#cccccc",lw=2.5,zorder=1)
    axA.text(min(k,e)-0.015,i,f"{abs(k-e)*100:.0f}pp",va="center",ha="right",fontsize=8,color="#777")
axA.scatter(en,y,s=120,color=EN,label="EN",zorder=3,edgecolor="white",lw=1)
axA.scatter(ko,y,s=120,color=KO,label="KO",zorder=3,edgecolor="white",lw=1)
axA.set_yticks(y); axA.set_yticklabels(LB); axA.set_xlim(0.55,1.02)
axA.set_xlabel("Node accuracy"); axA.set_title("(a) Per-node fact recognition: KO vs EN")
axA.legend(loc="lower left",frameon=True); axA.grid(axis="x",alpha=0.3); axA.invert_yaxis()

# Panel B: slope
navg={l:np.mean([nodes[l][n] for n in NK]) for l in ["ko","en"]}
yv=[(navg["ko"]-navg["en"])*100,(R[g9+"__ko"]["accuracy"]-R[g9+"__en"]["accuracy"])*100]
axB.plot([0,1],yv,"-o",color=NEG,lw=3,ms=14,zorder=3); axB.axhline(0,color="k",lw=0.8)
axB.annotate(f"{yv[0]:+.1f}pp",(0,yv[0]),textcoords="offset points",xytext=(0,12),ha="center",fontsize=12,fontweight="bold")
axB.annotate(f"{yv[1]:+.1f}pp",(1,yv[1]),textcoords="offset points",xytext=(0,-22),ha="center",fontsize=12,fontweight="bold",color=NEG)
axB.annotate(f"{abs(yv[1]-yv[0]):.0f}pp\ncollapse",xy=(0.5,sum(yv)/2),xytext=(0.6,sum(yv)/2),fontsize=11,color="#555",va="center")
axB.set_xticks([0,1]); axB.set_xticklabels(["Per-node\nfact recog.","Integrated\noutcome"])
axB.set_xlim(-0.35,1.5); axB.set_ylim(min(yv)-6,6)
axB.set_ylabel("KO $-$ EN gap (pp)"); axB.set_title("(b) Deficit emerges at integration")
axB.grid(axis="y",alpha=0.3)

plt.tight_layout(); plt.savefig("fig_node2panel.png", dpi=150)
print("saved fig_node2panel.png")
