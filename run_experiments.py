"""
본 실험 러너: 모델 × 언어 매트릭스.
각 (model, lang)별 정확도 + 케이스별 예측을 저장. 진행상황은 progress.log에 기록.
"""
import json, time, sys
from collections import Counter
from evaluate import query_ollama, parse_decision, PROMPT

MODELS = [
    "gemma2:9b-instruct-q4_0",
    "qwen2.5:7b-instruct-q4_0",
    "gemma2:2b-instruct-q8_0",
    "gemma2:2b-instruct-q4_0",
    "llama3.2:3b-instruct-q8_0",
    "llama3.2:3b-instruct-q4_0",
    "phi3:3.8b-mini-4k-instruct-q8_0",
    "phi3:3.8b-mini-4k-instruct-q4_0",
]
LANGS = ["ko", "en"]
DATASET = "dataset_v1.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run():
    cases = json.load(open(DATASET, encoding="utf-8"))
    results = {}
    for model in MODELS:
        for lang in LANGS:
            key = f"{model}__{lang}"
            log(f"START {key} ({len(cases)} cases)")
            rows, correct = [], 0
            t0 = time.time()
            for i, c in enumerate(cases):
                scenario = c["scenario_ko"] if lang == "ko" else c["scenario_en"]
                gold = c.get("gold_class")
                try:
                    resp = query_ollama(model, PROMPT.format(scenario=scenario))
                    pred = parse_decision(resp)
                except Exception as e:
                    pred = f"ERROR:{e}"
                ok = (pred == gold)
                correct += ok
                rows.append({"id": c["id"], "gold": gold, "pred": pred, "ok": ok,
                             "sensitive_type": c["sensitive_type"]})
                if (i + 1) % 30 == 0:
                    log(f"  {key} {i+1}/{len(cases)} (acc so far {correct/(i+1):.1%})")
            acc = correct / len(cases)
            dt = time.time() - t0
            results[key] = {"model": model, "lang": lang, "accuracy": acc,
                            "correct": correct, "n": len(cases), "secs": round(dt), "rows": rows}
            log(f"DONE {key}: acc={acc:.1%} ({correct}/{len(cases)}) in {dt:.0f}s")
            json.dump(results, open("results_v1.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
    # 요약
    log("=== SUMMARY ===")
    for key, r in results.items():
        log(f"{r['model']:34s} {r['lang']}  acc={r['accuracy']:.1%}")
    log("ALL DONE")

if __name__ == "__main__":
    run()
