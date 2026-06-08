"""
결과 분석 + 시각화. results_v1.json → figures.
라벨은 영어(논문용). 실험 완료 후 실행.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

R = json.load(open("results_v1.json", encoding="utf-8"))

def short(m):
    return m.split(":")[0] + ("-9b" if "9b" in m else "-7b" if "7b" in m else "-3b" if "3b" in m else "-3.8b" if "3.8b" in m else "")

# ── Fig1: 모델 × 언어 정확도 ──────────────────────────────────────
models = sorted({r["model"] for r in R.values()})
langs = ["ko", "en"]
acc = {(r["model"], r["lang"]): r["accuracy"] for r in R.values()}

x = np.arange(len(models)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
for i, lang in enumerate(langs):
    vals = [acc.get((m, lang), 0) for m in models]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lang.upper())
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=15)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("LLM PIPA Compliance Accuracy by Model and Language")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig1_model_lang.png", dpi=150)
print("→ fig1_model_lang.png")

# ── Fig2: 언어간 차이 (KO - EN) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
diffs = [acc.get((m, "ko"), 0) - acc.get((m, "en"), 0) for m in models]
colors = ["#d62728" if d < 0 else "#2ca02c" for d in diffs]
ax.barh([short(m) for m in models], diffs, color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Accuracy gap (KO − EN)")
ax.set_title("Cross-lingual consistency gap per model")
for i, d in enumerate(diffs):
    ax.text(d, i, f" {d:+.1%}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig("fig2_lang_gap.png", dpi=150)
print("→ fig2_lang_gap.png")

# ── Fig3: 민감유형별 정확도 (모든 결과 합산) ─────────────────────
by_stype = defaultdict(lambda: [0, 0])  # type -> [correct, total]
for r in R.values():
    for row in r["rows"]:
        st = row["sensitive_type"]
        by_stype[st][1] += 1
        by_stype[st][0] += int(row["ok"])
types = sorted(by_stype, key=lambda t: by_stype[t][0]/by_stype[t][1])
vals = [by_stype[t][0]/by_stype[t][1] for t in types]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(types, vals, color="#1f77b4")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("Accuracy by sensitive data type (all models/langs pooled)")
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig("fig3_sensitive_type.png", dpi=150)
print("→ fig3_sensitive_type.png")

# ── Fig4: 혼동 — gold별 예측 분포 (pooled) ───────────────────────
CLASSES = ["ALLOW", "STOP_LAWFUL_BASIS", "STOP_SENSITIVE_CONSENT",
           "STOP_PURPOSE", "STOP_PSEUDO_INSTITUTION", "STOP_SAFETY"]
cm = np.zeros((len(CLASSES), len(CLASSES)))
idx = {c: i for i, c in enumerate(CLASSES)}
for r in R.values():
    for row in r["rows"]:
        g, p = row["gold"], row["pred"]
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels([c.replace("STOP_", "S:") for c in CLASSES], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([c.replace("STOP_", "S:") for c in CLASSES], fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
ax.set_title("Confusion matrix (row-normalized, pooled)")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if cm[i, j] > 0:
            ax.text(j, i, f"{cm_norm[i,j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
plt.colorbar(im, fraction=0.046)
plt.tight_layout(); plt.savefig("fig4_confusion.png", dpi=150)
print("→ fig4_confusion.png")

# ── Fig5: 핵심 — 영어 성능 vs 언어 갭 (성능-안전 역상관) ──────────
fig, ax = plt.subplots(figsize=(8, 6))
en_acc = [acc.get((m, "en"), 0) for m in models]
gaps = [acc.get((m, "ko"), 0) - acc.get((m, "en"), 0) for m in models]
ax.scatter(en_acc, gaps, s=90, c="#d62728", zorder=3)
for m, x_, y_ in zip(models, en_acc, gaps):
    ax.annotate(short(m), (x_, y_), textcoords="offset points", xytext=(6, 4), fontsize=8)
# 추세선
if len(models) >= 3:
    z = np.polyfit(en_acc, gaps, 1)
    xs = np.linspace(min(en_acc), max(en_acc), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="gray",
            label=f"slope={z[0]:.2f}, r={np.corrcoef(en_acc, gaps)[0,1]:.2f}")
    ax.legend()
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("English accuracy (capability proxy)")
ax.set_ylabel("Cross-lingual gap (KO − EN)")
ax.set_title("Capability–Safety inversion: stronger models lose more in Korean")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig5_capability_safety.png", dpi=150)
print("→ fig5_capability_safety.png")

# ── 부트스트랩: 각 모델의 KO−EN 갭 95% CI ────────────────────────
def bootstrap_gap_ci(model, n_boot=2000):
    ko_rows = {r["id"]: r["ok"] for r in R[f"{model}__ko"]["rows"]}
    en_rows = {r["id"]: r["ok"] for r in R[f"{model}__en"]["rows"]}
    ids = [i for i in ko_rows if i in en_rows]
    paired = np.array([[ko_rows[i], en_rows[i]] for i in ids], dtype=float)
    n = len(paired)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = paired[idx]
        boots.append(s[:, 0].mean() - s[:, 1].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (paired[:, 0].mean() - paired[:, 1].mean()), lo, hi

print("\n=== Cross-lingual gap with 95% bootstrap CI ===")
gap_stats = {}
for m in models:
    try:
        g, lo, hi = bootstrap_gap_ci(m)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        gap_stats[m] = (g, lo, hi, sig)
        print(f"{short(m):16s} gap={g:+.1%} CI[{lo:+.1%}, {hi:+.1%}] {sig}")
    except Exception as e:
        print(f"{short(m)}: {e}")

# ── 텍스트 요약 ──────────────────────────────────────────────────
print("\n=== SUMMARY ===")
for m in models:
    ko, en = acc.get((m, "ko"), 0), acc.get((m, "en"), 0)
    print(f"{short(m):14s} KO={ko:.1%} EN={en:.1%} gap={ko-en:+.1%}")
print("\nSensitive-type accuracy:")
for t in types:
    print(f"  {t:12s} {by_stype[t][0]/by_stype[t][1]:.1%} (n={by_stype[t][1]})")
