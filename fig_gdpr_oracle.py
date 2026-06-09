"""fig8 교체: 공통요건 클래스별 PIPA-framed vs GDPR-framed 정확도 (GDPR oracle 대비)."""
import json
import numpy as np
from figstyle import KO, EN, NEG
import matplotlib.pyplot as plt

R=json.load(open("results_v1.json")); D={c["id"]:c for c in json.load(open("dataset_v1.json"))}
gp={int(k):v for k,v in json.load(open("gdpr_pred.json")).items()}
g9="gemma2:9b-instruct-q4_0"; pp={r["id"]:r["pred"] for r in R[f"{g9}__en"]["rows"]}
ids=[i for i in D if i in gp and i in pp]
PO={"STOP_PSEUDO_INSTITUTION","STOP_SAFETY"}
classes=[("STOP_SENSITIVE_CONSENT","sensitive\nconsent"),("STOP_PURPOSE","purpose"),
         ("STOP_LAWFUL_BASIS","lawful\nbasis"),("ALLOW","ALLOW")]
labels=[]; pa=[]; gd=[]
for cls,lab in classes:
    sub=[i for i in ids if D[i]["gold_class"]==cls and cls not in PO]
    labels.append(lab)
    pa.append(sum(1 for i in sub if pp[i]==cls)/len(sub)*100)
    gd.append(sum(1 for i in sub if gp[i]==cls)/len(sub)*100)
x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(7.5,4.4))
b1=ax.bar(x-w/2,pa,w,label="PIPA framing",color=EN)
b2=ax.bar(x+w/2,gd,w,label="GDPR framing",color=NEG)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.5,f"{b.get_height():.0f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy on regime-shared requirement (\\%)"); ax.set_ylim(0,105)
ax.set_title("Same facts, GDPR label: enforcement of shared requirements drops")
ax.legend(); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("fig8_jurisdiction.png",dpi=150)
print("saved fig8 (oracle version)")
