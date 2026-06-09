"""
GDPR oracle: 구조적 차이만 인코딩한 GDPR 정답 라벨 + 모델의 GDPR 판단 정확도.
GDPR gold = PIPA gold, 단 PIPA-고유 게이트(PSEUDO_INSTITUTION/SAFETY)는
GDPR엔 없으므로 → STOP_PURPOSE(목적제한/Art.6(4) 비호환 재이용)로 매핑.
[⚠️ 선교수님 확인 포인트: 재이용 매핑 = STOP_PURPOSE]
모델의 GDPR판단(gdpr_pred.json) vs GDPR gold → 공통요건 shift가 정답에서 멀어지는지 판정.
"""
import json
from collections import Counter, defaultdict

R = json.load(open("results_v1.json", encoding="utf-8"))
D = {c["id"]: c for c in json.load(open("dataset_v1.json", encoding="utf-8"))}
gdpr_pred = {int(k): v for k, v in json.load(open("gdpr_pred.json", encoding="utf-8")).items()}
g9 = "gemma2:9b-instruct-q4_0"
pipa_pred = {r["id"]: r["pred"] for r in R[f"{g9}__en"]["rows"]}

PIPA_ONLY = {"STOP_PSEUDO_INSTITUTION", "STOP_SAFETY"}
def gdpr_gold(pipa_gold):
    return "STOP_PURPOSE" if pipa_gold in PIPA_ONLY else pipa_gold

ids = [i for i in D if i in gdpr_pred and i in pipa_pred]
# 1) 모델 GDPR 정확도
gd_correct = sum(1 for i in ids if gdpr_pred[i] == gdpr_gold(D[i]["gold_class"]))
pi_correct = sum(1 for i in ids if pipa_pred[i] == D[i]["gold_class"])
print(f"N={len(ids)}")
print(f"모델 PIPA-framed 정확도(vs PIPA gold) = {pi_correct/len(ids):.1%}")
print(f"모델 GDPR-framed 정확도(vs GDPR gold) = {gd_correct/len(ids):.1%}")

# 2) 공통요건(=GDPR gold가 PIPA gold와 동일한 케이스)에서 shift 방향
shared = [i for i in ids if gdpr_gold(D[i]["gold_class"]) == D[i]["gold_class"]]
print(f"\n공통요건 시나리오 (정답 regime-invariant): N={len(shared)}")
shifted = [i for i in shared if pipa_pred[i] != gdpr_pred[i]]
print(f"  그 중 PIPA→GDPR로 판단 바뀐 것: {len(shifted)} ({len(shifted)/len(shared):.0%})")
c2w = sum(1 for i in shifted if pipa_pred[i]==D[i]["gold_class"] and gdpr_pred[i]!=D[i]["gold_class"])
w2c = sum(1 for i in shifted if pipa_pred[i]!=D[i]["gold_class"] and gdpr_pred[i]==D[i]["gold_class"])
w2w = sum(1 for i in shifted if pipa_pred[i]!=D[i]["gold_class"] and gdpr_pred[i]!=D[i]["gold_class"])
print(f"  정답→오답(유해): {c2w}  오답→정답(개선): {w2c}  오답→오답: {w2w}")
if shifted: print(f"  >> 유해 shift 비율 = {c2w/len(shifted):.0%}")

# 3) 민감동의·목적 클래스별
print("\n공통요건 클래스별 (GDPR gold 대비 정확도, PIPA-framed vs GDPR-framed):")
for cls in ["STOP_SENSITIVE_CONSENT", "STOP_PURPOSE", "STOP_LAWFUL_BASIS", "ALLOW"]:
    sub=[i for i in shared if D[i]["gold_class"]==cls]
    if not sub: continue
    pa=sum(1 for i in sub if pipa_pred[i]==cls)/len(sub)
    gd=sum(1 for i in sub if gdpr_pred[i]==cls)/len(sub)
    print(f"  {cls:24s} n={len(sub):3d}  PIPA-framed={pa:.0%}  GDPR-framed={gd:.0%}  Δ={(gd-pa)*100:+.0f}pp")
