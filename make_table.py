"""
Table 3 자동 생성: 모델별 KO/EN 정확도, 갭, bootstrap 95% CI, McNemar p.
LaTeX(booktabs) + Markdown 동시 출력.
"""
import json
import numpy as np

R = json.load(open("results_v1.json", encoding="utf-8"))
MC = {m["model"]: m for m in json.load(open("mcnemar_result.json", encoding="utf-8"))}

def short(m):
    base = m.split(":")[0]
    q = "q8" if "q8" in m else "q4"
    sz = "9b" if "9b" in m else "7b" if "7b" in m else "2b" if "2b" in m else "3b" if "3b" in m else "3.8b"
    return f"{base}-{sz}-{q}"

def boot_ci(model, n_boot=2000):
    ko = {r["id"]: r["ok"] for r in R[f"{model}__ko"]["rows"]}
    en = {r["id"]: r["ok"] for r in R[f"{model}__en"]["rows"]}
    ids = [i for i in ko if i in en]
    p = np.array([[ko[i], en[i]] for i in ids], float)
    rng = np.random.default_rng(0)
    b = [ (s:=p[rng.integers(0,len(p),len(p))])[:,0].mean()-s[:,1].mean() for _ in range(n_boot)]
    return np.percentile(b, [2.5, 97.5])

models = sorted({k.rsplit("__",1)[0] for k in R}, key=lambda m: -R[f"{m}__en"]["accuracy"])

def pfmt(p):
    return "<1e-8" if p < 1e-8 else f"{p:.1e}" if p < 0.01 else f"{p:.3f}"

rows = []
for m in models:
    ko = R[f"{m}__ko"]["accuracy"]; en = R[f"{m}__en"]["accuracy"]
    lo, hi = boot_ci(m)
    mc = MC.get(m, {})
    rows.append((short(m), ko, en, ko-en, lo, hi, mc.get("p", float("nan")), mc.get("sig","")))

# Markdown
print("### Table 3 (Markdown)\n")
print("| Model | KO | EN | Gap(KO−EN) | 95% CI | McNemar p | sig |")
print("|---|---|---|---|---|---|---|")
for s,ko,en,g,lo,hi,p,sig in rows:
    print(f"| {s} | {ko:.1%} | {en:.1%} | {g:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {pfmt(p)} | {sig} |")

# LaTeX
print("\n\n### Table 3 (LaTeX)\n")
print(r"\begin{table}[t]\centering")
print(r"\caption{Per-model Korean/English compliance accuracy, cross-lingual gap with 95\% bootstrap CI, and paired McNemar significance.}")
print(r"\label{tab:main}")
print(r"\begin{tabular}{lccccc}")
print(r"\toprule")
print(r"Model & KO & EN & Gap & 95\% CI & McNemar $p$ \\")
print(r"\midrule")
for s,ko,en,g,lo,hi,p,sig in rows:
    star = sig if sig and sig!="ns" else ""
    print(f"{s} & {ko*100:.1f} & {en*100:.1f} & {g*100:+.1f} & [{lo*100:+.1f}, {hi*100:+.1f}] & {pfmt(p)}{star} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
