"""
노드 평가 결과 시각화 + '사실인식 vs 추론통합' 진단.
node_result.json + results_v1.json(최종 outcome) 비교.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nodes = json.load(open("node_result.json", encoding="utf-8"))
NODE_KEYS = ["lawful_basis", "is_sensitive", "consent_ok", "purpose_specified", "within_purpose"]

# ── Fig6: 노드별 KO vs EN ─────────────────────────────────────────
x = np.arange(len(NODE_KEYS)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
for i, lang in enumerate(["ko", "en"]):
    vals = [nodes[lang][n] for n in NODE_KEYS]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lang.upper())
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.0%}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(NODE_KEYS, rotation=15)
ax.set_ylabel("Node accuracy"); ax.set_ylim(0, 1)
ax.set_title("Per-node (fact-recognition) accuracy: KO vs EN")
ax.legend(); ax.grid(axis="y", alpha=0.3)
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

# ── Fig7: 단계별 갭 비교 막대 ─────────────────────────────────────
stages, gaps = [], []
stages.append("node-avg"); gaps.append((node_avg["ko"]-node_avg["en"])*100)
stages.append("all-5-nodes"); gaps.append((allnode["ko"]-allnode["en"])*100)
if ko_out:
    stages.append("final-outcome"); gaps.append((ko_out-en_out)*100)
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#d62728" if g < 0 else "#2ca02c" for g in gaps]
ax.bar(stages, gaps, color=colors)
ax.axhline(0, color="k", lw=0.8)
for i, g in enumerate(gaps):
    ax.text(i, g, f"{g:+.1f}p", ha="center", va="bottom" if g >= 0 else "top", fontsize=10)
ax.set_ylabel("KO − EN accuracy gap (pp)")
ax.set_title("Where does Korean break down? Fact-recognition vs reasoning integration")
plt.tight_layout(); plt.savefig("fig7_stage_gap.png", dpi=150)
print("→ fig7_stage_gap.png")
