"""
CoT 개입 실험: '추론통합 결손'이 chain-of-thought로 완화되는가?
- baseline(JSON 강제, 설명금지) gemma2-9b 갭 = -23.1pp.
- 여기서는 단계별 추론을 허용한 뒤 최종 라벨만 파싱.
- KO/EN 둘 다 재실행 → CoT 갭과 baseline 갭 비교.
출력: cot_result.json
"""
import json, re, time, urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:9b-instruct-q4_0"
DATASET = "dataset_v1.json"
LABELS = ["STOP_LAWFUL_BASIS", "STOP_SENSITIVE_CONSENT", "STOP_PURPOSE",
          "STOP_PSEUDO_INSTITUTION", "STOP_SAFETY", "ALLOW"]

PROMPT_KO = """당신은 한국 개인정보보호법(PIPA) 전문가입니다.
아래 시나리오를 6개 기준에 따라 순서대로 단계별로 검토하세요.

판단 기준(순서대로):
1) 적법근거(공개정보 or 기타 법적근거)가 없으면 → STOP_LAWFUL_BASIS
2) 민감정보(건강/인종/정치성향/생체/사상)인데 별도 동의가 없으면 → STOP_SENSITIVE_CONSENT
3) 수집 목적이 특정되지 않으면 → STOP_PURPOSE
4) 목적 외 이용이면서 목적외 적법근거도 없고, 지정기관을 통하지 않은 가명처리면 → STOP_PSEUDO_INSTITUTION
5) 가명처리 안전성 검토를 통과 못하면 → STOP_SAFETY
6) 위 모두 통과하면 → ALLOW

시나리오:
{scenario}

각 단계를 한 줄씩 짚어가며 추론한 뒤, 마지막 줄에 정확히 이렇게 결론지으세요:
최종판단: <위 6개 라벨 중 하나>
"""

PROMPT_EN = """You are an expert on Korea's Personal Information Protection Act (PIPA).
Analyze the scenario below by checking the six criteria in order, step by step.

Criteria (in order):
1) No lawful basis (publicly available or other legal ground) -> STOP_LAWFUL_BASIS
2) Sensitive data (health/race/political/biometric/belief) without separate consent -> STOP_SENSITIVE_CONSENT
3) Purpose not specified -> STOP_PURPOSE
4) Out-of-purpose use, no other legal ground, pseudonymized without a designated institution -> STOP_PSEUDO_INSTITUTION
5) Pseudonymization safety review not passed -> STOP_SAFETY
6) If all pass -> ALLOW

Scenario:
{scenario}

Reason through each step one line at a time, then on the final line conclude exactly:
FINAL: <one of the six labels above>
"""

def gold_to_class(o):
    if o.startswith("STOP:no_lawful_basis"): return "STOP_LAWFUL_BASIS"
    if o.startswith("STOP:sensitive"): return "STOP_SENSITIVE_CONSENT"
    if o.startswith("STOP:purpose"): return "STOP_PURPOSE"
    if o.startswith("STOP:no_designated"): return "STOP_PSEUDO_INSTITUTION"
    if o.startswith("STOP:safety"): return "STOP_SAFETY"
    return "ALLOW"

def query(prompt):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 512}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]

def parse_final(resp):
    # 마지막 등장하는 라벨을 최종 판단으로 (결론 줄 우선)
    tail = resp.split("최종판단")[-1].split("FINAL")[-1]
    for lab in LABELS:  # STOP_*를 ALLOW보다 먼저 매칭
        if lab in tail:
            return lab
    for lab in LABELS:
        if lab in resp:
            return lab
    return "PARSE_ERROR"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open("cot.log", "a").write(line + "\n")

def run():
    cases = json.load(open(DATASET, encoding="utf-8"))
    out = {}
    for lang in ["ko", "en"]:
        tmpl = PROMPT_KO if lang == "ko" else PROMPT_EN
        rows, correct = [], 0
        t0 = time.time()
        for i, c in enumerate(cases):
            scen = c["scenario_ko"] if lang == "ko" else c["scenario_en"]
            gold = c["gold_class"]
            try:
                pred = parse_final(query(tmpl.format(scenario=scen)))
            except Exception as e:
                pred = f"ERROR:{e}"
            ok = (pred == gold)
            correct += ok
            rows.append({"id": c["id"], "gold": gold, "pred": pred, "ok": ok})
            if (i + 1) % 50 == 0:
                log(f"  {lang} {i+1}/{len(cases)} (acc {correct/(i+1):.1%})")
        acc = correct / len(cases)
        out[lang] = {"accuracy": acc, "correct": correct, "n": len(cases),
                     "secs": round(time.time() - t0), "rows": rows}
        log(f"DONE cot {lang}: acc={acc:.1%} in {out[lang]['secs']}s")
        json.dump(out, open("cot_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    gap = out["ko"]["accuracy"] - out["en"]["accuracy"]
    log(f"=== CoT gemma2-9b: KO={out['ko']['accuracy']:.1%} EN={out['en']['accuracy']:.1%} gap={gap:+.1%} (baseline gap -23.1pp) ===")

if __name__ == "__main__":
    run()
