"""False-permit 비율 (STOP→ALLOW 오판) 모델별 KO vs EN. 기존 results에서 산출."""
import json
import numpy as np
from figstyle import KO, EN
import matplotlib.pyplot as plt

R = json.load(open("results_v1.json", encoding="utf-8"))
def short(m): return m.split(":")[0]+("-9b" if "9b" in m else "-7b" if "7b" in m else "-2b" if "2b" in m else "-3b" if "3b" in m else "")+("-q8" if "q8" in m else "-q4")
models = sorted({k.rsplit("__",1)[0] for k in R}, key=lambda m:-R[m+"__en"]["accuracy"])
def fp(m,l):
    stop=[r for r in R[f"{m}__{l}"]["rows"] if r["gold"]!="ALLOW"]
    return sum(1 for r in stop if r["pred"]=="ALLOW")/len(stop)
ko=[fp(m,"ko")*100 for m in models]; en=[fp(m,"en")*100 for m in models]
x=np.arange(len(models)); w=0.38
fig,ax=plt.subplots(figsize=(9,4.6))
ax.bar(x-w/2, ko, w, label="KO", color=KO)
ax.bar(x+w/2, en, w, label="EN", color=EN)
ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=20, ha="right")
ax.set_ylabel("False-permit rate (\\%)")
ax.set_title("False permits (prohibited cases judged ALLOW) by model and language")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig9_falsepermit.png", dpi=150)
print("saved fig9_falsepermit.png")
