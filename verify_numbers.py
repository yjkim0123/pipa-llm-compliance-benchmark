"""
수치 검증: 논문 draft에 등장하는 모든 핵심 수치를 결과 json에서 재계산/추출.
출력값을 draft(abstract/RESULTS_draft/main.tex)와 1:1 대조.
"""
import json
import numpy as np

R = json.load(open("results_v1.json", encoding="utf-8"))
def acc(m, l): return R[f"{m}__{l}"]["accuracy"]

print("="*60)
print("[1] 모델별 KO/EN 정확도 + 갭 (draft Table 3 / 4.x 대조)")
models = sorted({k.rsplit("__",1)[0] for k in R}, key=lambda m: -acc(m,"en"))
allacc = []
for m in models:
    ko, en = acc(m,"ko"), acc(m,"en")
    allacc += [ko, en]
    print(f"  {m.split(':')[0]:8s} {('9b' if '9b' in m else '7b' if '7b' in m else '2b' if '2b' in m else '3b' if '3b' in m else '?'):>3s} {'q8' if 'q8' in m else 'q4'}: KO={ko:.1%} EN={en:.1%} gap={ko-en:+.1%}")
print(f"  >> 정확도 범위: {min(allacc):.1%} ~ {max(allacc):.1%}  (draft '38%~83%' 확인)")

print("\n[2] 성능-안전 상관 r (draft 'r=-0.66')")
en = [acc(m,"en") for m in models]; gap = [acc(m,"ko")-acc(m,"en") for m in models]
r = np.corrcoef(en, gap)[0,1]
print(f"  >> Pearson r = {r:.3f}")

print("\n[3] McNemar (draft p값들)")
for x in json.load(open("mcnemar_result.json", encoding="utf-8")):
    print(f"  {x['model'].split(':')[0]:8s}: b={x['b_en_only']} c={x['c_ko_only']} p={x['p']:.2e} {x['sig']} ({x['direction']})")

print("\n[4] 노드 진단 (draft '91.1 vs 92.5, -1.4pp / outcome -23.1pp')")
nd = json.load(open("node_result.json", encoding="utf-8"))
NK = ["lawful_basis","is_sensitive","consent_ok","purpose_specified","within_purpose"]
for l in ["ko","en"]:
    navg = np.mean([nd[l][n] for n in NK])
    print(f"  node-avg {l.upper()}={navg:.1%}", end="  ")
print()
g9 = "gemma2:9b-instruct-q4_0"
print(f"  outcome gap (gemma2-9b) = {acc(g9,'ko')-acc(g9,'en'):+.1%}")
print(f"  consent_ok: KO={nd['ko']['consent_ok']:.1%} EN={nd['en']['consent_ok']:.1%}")

print("\n[5] RQ3 관할권 (draft '35%, 민감동의 65%, 목적 43%')")
cases = {c["id"]: c for c in json.load(open("dataset_v1.json", encoding="utf-8"))}
gdpr = {int(k):v for k,v in json.load(open("gdpr_pred.json", encoding="utf-8")).items()}
pipa = {r["id"]: r["pred"] for r in R[f"{g9}__en"]["rows"]}
from collections import defaultdict
bg = defaultdict(lambda:[0,0]); tot=0; dv=0
for cid,c in cases.items():
    if cid in gdpr and cid in pipa:
        tot+=1; d=(pipa[cid]!=gdpr[cid]); dv+=d
        bg[c["gold_class"]][1]+=1; bg[c["gold_class"]][0]+=d
print(f"  전체 divergence = {dv/tot:.0%}")
for g in ["STOP_PSEUDO_INSTITUTION","STOP_SAFETY","STOP_SENSITIVE_CONSENT","STOP_PURPOSE"]:
    if g in bg: print(f"  {g}: {bg[g][0]/bg[g][1]:.0%}")

print("\n[6] 재현성 (draft 'temp0 sd0.00, temp0.7 sd<=0.79, gap -17.5~-18.0')")
st = json.load(open("stability_result.json", encoding="utf-8"))
for k in ["temp0.0_ko","temp0.0_en","temp0.7_ko","temp0.7_en"]:
    print(f"  {k}: mean={st[k]['mean']:.1%} sd={st[k]['sd']:.2%}")
print(f"  gap temp0.0={st['gap_temp0.0']:+.1%}  gap temp0.7={st['gap_temp0.7']:+.1%}")
print("="*60)
