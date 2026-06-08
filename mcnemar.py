"""
McNemar 검정: 같은 시나리오에 대한 KO vs EN 정답/오답의 paired 비교.
부트스트랩 CI에 더해, 언어 간 차이가 우연이 아님을 표준 검정으로 입증.
results_v1.json 사용 (각 모델의 ko/en rows, id 정렬).
"""
import json, math
from collections import defaultdict

R = json.load(open("results_v1.json", encoding="utf-8"))

def short(m):
    return m.split(":")[0] + ("-9b" if "9b" in m else "-7b" if "7b" in m else
            "-2b-q8" if "2b-instruct-q8" in m else "-2b-q4" if "2b-instruct-q4" in m else
            "-3b-q8" if "3b-instruct-q8" in m else "-3b-q4" if "3b-instruct-q4" in m else
            "-q8" if "q8" in m else "-q4")

def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# 모델 목록 (ko/en 쌍)
models = sorted({k.rsplit("__", 1)[0] for k in R})

print(f"{'model':16s} {'b(EN✓KO✗)':>10s} {'c(KO✓EN✗)':>10s} {'chi2':>7s} {'p':>9s}  sig")
rows_out = []
for m in models:
    ko = {r["id"]: r["ok"] for r in R[f"{m}__ko"]["rows"]}
    en = {r["id"]: r["ok"] for r in R[f"{m}__en"]["rows"]}
    ids = [i for i in ko if i in en]
    b = sum(1 for i in ids if en[i] and not ko[i])   # EN 맞고 KO 틀림
    c = sum(1 for i in ids if ko[i] and not en[i])   # KO 맞고 EN 틀림
    n = b + c
    if n == 0:
        continue
    # 연속성 보정 McNemar
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    z = math.sqrt(chi2)
    p = 2 * (1 - norm_cdf(z))  # chi2(1df) ~ z^2
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    direction = "EN>KO" if b > c else "KO>EN" if c > b else "="
    print(f"{short(m):16s} {b:10d} {c:10d} {chi2:7.1f} {p:9.4g}  {sig} ({direction})")
    rows_out.append({"model": m, "b_en_only": b, "c_ko_only": c, "chi2": round(chi2, 2),
                     "p": p, "sig": sig, "direction": direction})

json.dump(rows_out, open("mcnemar_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n→ mcnemar_result.json 저장")
print("해석: b=EN만 맞힘, c=KO만 맞힘. b≫c이고 p<0.05면 'EN이 유의하게 우세'")
