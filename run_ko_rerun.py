"""
받침/조사 수정(자연스러운 한국어) 후 재실험.
- 한국어(ko)만 전 모델 재실행 → results_v1.json의 __ko 키 갱신.
- 영어(en)는 입력 텍스트가 바이트 동일 + greedy(temp=0, 재현성 sd=0.00%)라 불변 → 기존값 유지.
- 가정 검증: gemma2:9b EN 1개만 재실행해 기존 __en과 1:1 일치 확인(SANITY).
"""
import json, time
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
DATASET = "dataset_v1.json"
SANITY_MODEL = "gemma2:9b-instruct-q4_0"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("rerun_ko.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def eval_one(model, lang, cases):
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
        if (i + 1) % 50 == 0:
            log(f"  {model}__{lang} {i+1}/{len(cases)} (acc {correct/(i+1):.1%})")
    acc = correct / len(cases)
    dt = time.time() - t0
    return {"model": model, "lang": lang, "accuracy": acc, "correct": correct,
            "n": len(cases), "secs": round(dt), "rows": rows}, dt

def run():
    cases = json.load(open(DATASET, encoding="utf-8"))
    results = json.load(open("results_v1.json", encoding="utf-8"))  # 기존(영어 보존)

    # --- SANITY: 영어 1개 재실행 → 기존값과 일치 확인 ---
    log(f"SANITY: re-running {SANITY_MODEL} EN to verify greedy determinism")
    new_en, _ = eval_one(SANITY_MODEL, "en", cases)
    old_en = results[f"{SANITY_MODEL}__en"]
    old_pred = {r["id"]: r["pred"] for r in old_en["rows"]}
    mism = sum(1 for r in new_en["rows"] if old_pred.get(r["id"]) != r["pred"])
    log(f"SANITY result: EN acc old={old_en['accuracy']:.1%} new={new_en['accuracy']:.1%}; mismatched preds={mism}/{len(cases)}")
    if mism == 0:
        log("SANITY PASS: EN bit-identical → keeping all existing EN results.")
    else:
        log(f"SANITY WARN: {mism} EN preds differ → consider full EN re-run.")

    # --- 한국어 전 모델 재실행 ---
    for model in MODELS:
        key = f"{model}__ko"
        log(f"START {key} ({len(cases)} cases)")
        res, dt = eval_one(model, "ko", cases)
        results[key] = res
        log(f"DONE {key}: acc={res['accuracy']:.1%} ({res['correct']}/{len(cases)}) in {dt:.0f}s")
        json.dump(results, open("results_v1.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    log("=== KO SUMMARY (post-josa) ===")
    for model in MODELS:
        ko = results[f"{model}__ko"]["accuracy"]
        en = results[f"{model}__en"]["accuracy"]
        log(f"{model:34s} KO={ko:.1%} EN={en:.1%} gap={ko-en:+.1%}")
    log("ALL DONE")

if __name__ == "__main__":
    run()
