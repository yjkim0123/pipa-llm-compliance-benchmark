"""
RQ3 분석: PIPA 판단 vs GDPR 판단 (같은 모델, 같은 EN 시나리오).
- 관할권 민감도: 두 판단이 갈리는 비율
- PIPA-고유 케이스(PSEUDO/SAFETY): GDPR에서 ALLOW로 전환 = 관할권 구분 / STOP 유지 = 혼동
- PIPA-무관 케이스: 일관돼야 정상 (안 갈려야)
"""
import json
from collections import defaultdict, Counter
import numpy as np
from figstyle import KO, NEG  # 공통 팔레트/폰트
import matplotlib.pyplot as plt

cases = {c["id"]: c for c in json.load(open("dataset_v1.json", encoding="utf-8"))}
gdpr = {int(k): v for k, v in json.load(open("gdpr_pred.json", encoding="utf-8")).items()}
R = json.load(open("results_v1.json", encoding="utf-8"))
pipa = {r["id"]: r["pred"] for r in R["gemma2:9b-instruct-q4_0__en"]["rows"]}

PIPA_ONLY = {"STOP_PSEUDO_INSTITUTION", "STOP_SAFETY"}

n = 0; diverge = 0
by_gold = defaultdict(lambda: [0, 0])  # gold -> [diverge, total]
pseudo_to_allow = 0; pseudo_total = 0; pseudo_stayed = 0
for cid, c in cases.items():
    if cid not in gdpr or cid not in pipa:
        continue
    n += 1
    g = c["gold_class"]
    d = (pipa[cid] != gdpr[cid])
    diverge += d
    by_gold[g][1] += 1
    by_gold[g][0] += d
    if g in PIPA_ONLY:
        pseudo_total += 1
        if gdpr[cid] == "ALLOW":
            pseudo_to_allow += 1
        elif "PSEUDO" in gdpr[cid] or "SAFETY" in gdpr[cid]:
            pseudo_stayed += 1

print(f"총 {n}개, 전체 관할권 divergence(PIPA판단≠GDPR판단): {diverge}/{n} = {diverge/n:.0%}\n")
print("=== gold_class별 divergence (PIPA-고유는 높아야 정상, 무관은 낮아야 정상) ===")
for g, (dv, tot) in sorted(by_gold.items(), key=lambda x: -x[1][0]/x[1][1]):
    tag = "  [PIPA-고유]" if g in PIPA_ONLY else ""
    print(f"  {g:26s} {dv}/{tot} = {dv/tot:.0%}{tag}")
print(f"\n=== PIPA-고유 STOP 케이스 (n={pseudo_total}) GDPR 판단 ===")
print(f"  ALLOW로 전환(관할권 구분): {pseudo_to_allow}/{pseudo_total} = {pseudo_to_allow/max(pseudo_total,1):.0%}")
print(f"  PIPA식 STOP 유지(혼동):    {pseudo_stayed}/{pseudo_total}")

# Fig8: gold_class별 divergence 막대
golds = sorted(by_gold, key=lambda g: by_gold[g][0]/by_gold[g][1])
vals = [by_gold[g][0]/by_gold[g][1] for g in golds]
colors = [NEG if g in PIPA_ONLY else KO for g in golds]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([g.replace("STOP_", "S:") for g in golds], vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.0%}", va="center", fontsize=9)
ax.set_xlabel("PIPA vs GDPR judgment divergence")
ax.set_title("Jurisdiction sensitivity by case type (red = PIPA-specific)")
ax.set_xlim(0, 1.05)
plt.tight_layout(); plt.savefig("fig8_jurisdiction.png", dpi=150)
print("\n→ fig8_jurisdiction.png")

print("\n해석:")
print("- PIPA-고유(빨강)만 divergence 높고 무관(파랑) 낮으면 → 모델이 관할권을 올바르게 구분")
print("- 무관 케이스도 divergence 높으면 → 관할권 프레이밍에 불안정하게 흔들림(신뢰성 문제)")
