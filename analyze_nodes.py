"""
노드 평가 결과 시각화 + '사실인식 vs 추론통합' 진단.
node_result.json + results_v1.json(최종 outcome) 비교.
"""
import json
import numpy as np
from figstyle import KO, EN, NEG  # 공통 팔레트/폰트
import matplotlib.pyplot as plt

nodes = json.load(open("node_result.json", encoding="utf-8"))
NODE_KEYS = ["lawful_basis", "is_sensitive", "consent_ok", "purpose_specified", "within_purpose"]

# ── Fig6: 노드별 KO vs EN (덤벨 차트) ─────────────────────────────
LABELS = ["lawful basis", "sensitivity", "consent adequacy", "purpose spec.", "within purpose"]
ko_vals = [nodes["ko"][n] for n in NODE_KEYS]
en_vals = [nodes["en"][n] for n in NODE_KEYS]
ypos = np.arange(len(NODE_KEYS))
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, (k, e) in enumerate(zip(ko_vals, en_vals)):
    ax.plot([k, e], [i, i], color="#cccccc", lw=2.5, zorder=1)
    ax.text(min(k, e) - 0.015, i, f"{abs(k-e)*100:.0f}pp", va="center", ha="right",
            fontsize=8, color="#777")
ax.scatter(en_vals, ypos, s=130, color=EN, label="EN", zorder=3, edgecolor="white", lw=1)
ax.scatter(ko_vals, ypos, s=130, color=KO, label="KO", zorder=3, edgecolor="white", lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(LABELS)
ax.set_xlim(0.55, 1.02); ax.set_xlabel("Node accuracy")
ax.set_title("Per-node fact recognition: KO vs EN are nearly identical")
ax.legend(loc="lower left", frameon=True); ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
plt.tight_layout(); plt.savefig("fig6_node_accuracy.png", dpi=150)
print("→ fig6_node_accuracy.png")

# ── 진단: 노드평균 vs 최종 outcome ────────────────────────────────
node_avg = {l: np.mean([nodes[l][n] for n in NODE_KEYS]) for l in ["ko", "en"]}
allnode = {l: nodes[l]["_all_nodes"] for l in ["ko", "en"]}

# 최종 outcome 정확도 (results_v1.json, gemma2-9b)
R = json.load(open("results_v1.json", encoding="utf-8"))
def outcome_acc(model_key):
    return R[model_key]["accuracy"] if model_key in R else None
ko_out = outcome_acc("gemma2:9b-instruct-q4_0__ko")
en_out = outcome_acc("gemma2:9b-instruct-q4_0__en")

print("\n=== 진단: 단계별 정확도 (gemma2-9b) ===")
print(f"{'':18s}  KO       EN      gap")
print(f"{'node-avg':18s}  {node_avg['ko']:.1%}  {node_avg['en']:.1%}  {(node_avg['ko']-node_avg['en'])*100:+.1f}p")
print(f"{'all-5-nodes':18s}  {allnode['ko']:.1%}  {allnode['en']:.1%}  {(allnode['ko']-allnode['en'])*100:+.1f}p")
if ko_out:
    print(f"{'final-outcome':18s}  {ko_out:.1%}  {en_out:.1%}  {(ko_out-en_out)*100:+.1f}p")

print("\n해석 가이드:")
print("- node-avg 갭 ≈ 0 인데 final-outcome 갭 큼 → '추론 통합' 단계에서 KO 붕괴 (언어이해 아닌 법추론)")
print("- node-avg 갭부터 큼 → 언어이해/사실인식 단계부터 붕괴")

# ── Fig7: 사실인식 → 통합 단계 갭 급락 (slope chart) ──────────────
xlab = ["Per-node\nfact recognition", "Integrated\noutcome"]
yv = [(node_avg["ko"]-node_avg["en"])*100, (ko_out-en_out)*100 if ko_out else 0.0]
fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.plot([0, 1], yv, "-o", color=NEG, lw=3, ms=15, zorder=3)
ax.axhline(0, color="k", lw=0.8)
ax.annotate(f"{yv[0]:+.1f}pp", (0, yv[0]), textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=13, fontweight="bold")
ax.annotate(f"{yv[1]:+.1f}pp", (1, yv[1]), textcoords="offset points", xytext=(0, -24),
            ha="center", fontsize=13, fontweight="bold", color=NEG)
# 급락 강조
ax.annotate(f"{abs(yv[1]-yv[0]):.0f}pp collapse", xy=(0.5, (yv[0]+yv[1])/2),
            xytext=(0.62, (yv[0]+yv[1])/2), fontsize=11, color="#555", va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(xlab, fontsize=11)
ax.set_xlim(-0.35, 1.45)
ax.set_ylim(min(yv) - 6, 6)
ax.set_ylabel("KO − EN accuracy gap (pp)")
ax.set_title("The Korean deficit emerges at reasoning integration")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig7_stage_gap.png", dpi=150)
print("→ fig7_stage_gap.png")
