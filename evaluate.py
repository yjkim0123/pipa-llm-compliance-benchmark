"""
LLM Compliance Evaluation — Pilot
시나리오를 LLM에 주고 PIPA 처리 결과(outcome)를 분류시켜 ground truth와 비교.
ollama 로컬 모델 사용.
"""
import json, re, sys, urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

# 결과 클래스 (rule_engine의 final_outcome을 7개 클래스로 정규화)
OUTCOME_CLASSES = {
    "ALLOW":  "처리 허용 (이용/제공/가명처리 가능, 정상 완료)",
    "STOP_LAWFUL_BASIS": "중단: 적법근거 없음",
    "STOP_SENSITIVE_CONSENT": "중단: 민감정보인데 별도동의 없음",
    "STOP_PURPOSE": "중단: 수집목적 미특정",
    "STOP_PSEUDO_INSTITUTION": "중단: 지정기관 없이 가명처리",
    "STOP_SAFETY": "중단: 가명처리 안전성 검토 실패",
}

def gold_to_class(outcome: str) -> str:
    if outcome.startswith("STOP:no_lawful_basis"): return "STOP_LAWFUL_BASIS"
    if outcome.startswith("STOP:sensitive"): return "STOP_SENSITIVE_CONSENT"
    if outcome.startswith("STOP:purpose"): return "STOP_PURPOSE"
    if outcome.startswith("STOP:no_designated"): return "STOP_PSEUDO_INSTITUTION"
    if outcome.startswith("STOP:safety"): return "STOP_SAFETY"
    return "ALLOW"  # COMPLETE / PRESERVE / DISPOSAL 계열

PROMPT = """당신은 한국 개인정보보호법(PIPA) 전문가입니다.
아래 개인정보 처리 시나리오를 분석하여, 처리가 적법하게 가능한지 또는 어느 단계에서 중단되어야 하는지 판단하세요.

판단 기준(순서대로 검토):
1) 적법근거(공개정보 or 기타 법적근거)가 없으면 → STOP_LAWFUL_BASIS
2) 민감정보(건강/인종/정치성향/생체/사상/성생활/범죄)인데 별도 동의가 없으면 → STOP_SENSITIVE_CONSENT
3) 수집 목적이 특정되지 않으면 → STOP_PURPOSE
4) 목적 외 이용이면서 목적외 적법근거도 없고, 지정기관을 통하지 않은 가명처리면 → STOP_PSEUDO_INSTITUTION
5) 가명처리 안전성 검토를 통과 못하면 → STOP_SAFETY
6) 위 모두 통과하면 → ALLOW

시나리오:
{scenario}

다음 6개 중 정확히 하나만 골라 JSON으로만 답하세요. 설명 금지.
{{"decision": "ALLOW|STOP_LAWFUL_BASIS|STOP_SENSITIVE_CONSENT|STOP_PURPOSE|STOP_PSEUDO_INSTITUTION|STOP_SAFETY"}}
"""

def query_ollama(model: str, prompt: str) -> str:
    data = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0}, "format": "json",
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]

def parse_decision(resp: str) -> str:
    try:
        return json.loads(resp).get("decision", "PARSE_ERROR")
    except Exception:
        m = re.search(r"(STOP_\w+|ALLOW)", resp)
        return m.group(1) if m else "PARSE_ERROR"

def main(model="gemma2:9b-instruct-q4_0", lang="ko", dataset="pilot_dataset.json"):
    cases = json.load(open(dataset, encoding="utf-8"))
    correct = 0
    rows = []
    for c in cases:
        scenario = c["scenario_ko"] if lang == "ko" else c["scenario_en"]
        gold = c.get("gold_class") or gold_to_class(c["outcome"])
        resp = query_ollama(model, PROMPT.format(scenario=scenario))
        pred = parse_decision(resp)
        ok = (pred == gold)
        correct += ok
        rows.append({"id": c["id"], "gold": gold, "pred": pred, "ok": ok})
        print(f"Case {c['id']:2d} [{ '✓' if ok else '✗'}] gold={gold:24s} pred={pred}")
    acc = correct / len(cases)
    print(f"\n=== {model} ({lang}) Accuracy: {correct}/{len(cases)} = {acc:.1%} ===")
    return rows, acc

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:9b-instruct-q4_0"
    lang = sys.argv[2] if len(sys.argv) > 2 else "ko"
    main(model, lang)
