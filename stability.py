"""
재현성/안정성 테스트.
핵심 모델(gemma2-9b)을 KO/EN 각각 N회 반복하여 정확도 분산 측정.
- temp=0: 결정론성 확인 (분산 ~0 기대)
- temp=0.7: 샘플링 robustness (갭이 반복해도 유지되는지)
서브셋(앞 80케이스)으로 빠르게.
"""
import json, statistics, urllib.request, re

OLLAMA_URL = "http://localhost:11434/api/generate"
from evaluate import PROMPT, parse_decision

MODEL = "gemma2:9b-instruct-q4_0"
N_REPEATS = 5
SUBSET = 80

def query(model, prompt, temp, seed):
    data = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temp, "seed": seed}, "format": "json",
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]

def run_once(cases, lang, temp, seed):
    correct = 0
    for c in cases:
        scen = c["scenario_ko"] if lang == "ko" else c["scenario_en"]
        try:
            pred = parse_decision(query(MODEL, PROMPT.format(scenario=scen), temp, seed))
        except Exception:
            pred = "ERR"
        correct += (pred == c.get("gold_class"))
    return correct / len(cases)

def main():
    cases = json.load(open("dataset_v1.json", encoding="utf-8"))[:SUBSET]
    out = {}
    for temp in [0.0, 0.7]:
        for lang in ["ko", "en"]:
            accs = []
            for rep in range(N_REPEATS):
                a = run_once(cases, lang, temp, seed=rep + 1)
                accs.append(a)
                print(f"temp={temp} {lang} rep{rep+1}: {a:.1%}", flush=True)
            mean = statistics.mean(accs)
            sd = statistics.pstdev(accs)
            out[f"temp{temp}_{lang}"] = {"accs": accs, "mean": mean, "sd": sd}
            print(f"  => temp={temp} {lang}: mean={mean:.1%} sd={sd:.2%}\n", flush=True)
    # 갭 안정성
    for temp in [0.0, 0.7]:
        ko = out[f"temp{temp}_ko"]["mean"]
        en = out[f"temp{temp}_en"]["mean"]
        out[f"gap_temp{temp}"] = ko - en
        print(f"GAP (KO-EN) @temp{temp}: {ko-en:+.1%}")
    json.dump(out, open("stability_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n→ stability_result.json 저장")

if __name__ == "__main__":
    main()
