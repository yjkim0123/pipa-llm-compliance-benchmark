"""
Graphical Abstract (IEEE Access 선택 항목).
핵심 메시지 2패널: (A) 성능-안전 역상관, (B) 사실인식→통합 단계 갭 붕괴.
실데이터(results_v1.json, node_result.json)에서 직접 산출.
"""
import json
import numpy as np
from figstyle import KO, EN, NEG
import matplotlib.pyplot as plt

R = json.load(open("results_v1.json", encoding="utf-8"))
nodes = json.load(open("node_result.json", encoding="utf-8"))
NODE_KEYS = ["lawful_basis", "is_sensitive", "consent_ok", "purpose_specified", "within_purpose"]

def short(m):
    return m.split(":")[0] + ("-9b" if "9b" in m else "-7b" if "7b" in m else "-3b" if "3b" in m else "-3.8b" if "3.8b" in m else "")

models = sorted({r["model"] for r in R.values()})
acc = {(r["model"], r["lang"]): r["accuracy"] for r in R.values()}
en_acc = [acc[(m, "en")] * 100 for m in models]
gaps = [(acc[(m, "ko")] - acc[(m, "en")]) * 100 for m in models]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# ── Panel A: capability–safety inversion ──
axA.scatter(en_acc, gaps, s=140, c=NEG, zorder=3, edgecolor="white", lw=1)
for m, x_, y_ in zip(models, en_acc, gaps):
    axA.annotate(short(m), (x_, y_), textcoords="offset points", xytext=(6, 5), fontsize=8)
z = np.polyfit(en_acc, gaps, 1)
xs = np.linspace(min(en_acc), max(en_acc), 50)
r = np.corrcoef(en_acc, gaps)[0, 1]
axA.plot(xs, np.polyval(z, xs), "--", color="gray", lw=2, label=f"$r={r:.2f}$")
axA.axhline(0, color="k", lw=0.6)
axA.set_xlabel("English accuracy (capability)")
axA.set_ylabel("Korean$-$English gap (pp)")
axA.set_title("Stronger in English $\\Rightarrow$ weaker in Korean")
axA.legend(loc="lower left")

# ── Panel B: node → outcome collapse ──
node_avg = {l: np.mean([nodes[l][n] for n in NODE_KEYS]) for l in ["ko", "en"]}
g9 = "gemma2:9b-instruct-q4_0"
yv = [(node_avg["ko"] - node_avg["en"]) * 100,
      (acc[(g9, "ko")] - acc[(g9, "en")]) * 100]
axB.plot([0, 1], yv, "-o", color=NEG, lw=3, ms=16, zorder=3)
axB.axhline(0, color="k", lw=0.8)
axB.annotate(f"{yv[0]:+.1f}pp", (0, yv[0]), textcoords="offset points", xytext=(0, 14),
             ha="center", fontsize=13, fontweight="bold")
axB.annotate(f"{yv[1]:+.1f}pp", (1, yv[1]), textcoords="offset points", xytext=(0, -26),
             ha="center", fontsize=13, fontweight="bold", color=NEG)
axB.annotate(f"{abs(yv[1]-yv[0]):.0f}pp\ncollapse", xy=(0.5, sum(yv)/2),
             xytext=(0.60, sum(yv)/2), fontsize=12, color="#555", va="center")
axB.set_xticks([0, 1]); axB.set_xticklabels(["Per-node\nfact recognition", "Integrated\noutcome"])
axB.set_xlim(-0.35, 1.5); axB.set_ylim(min(yv) - 6, 6)
axB.set_ylabel("Korean$-$English gap (pp)")
axB.set_title("Facts intact, but reasoning collapses")

fig.suptitle("Auditing LLMs as Korean PIPA Compliance Reasoners: a Capability$-$Safety Inversion",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("graphical_abstract.png", dpi=200)
print("→ graphical_abstract.png")
