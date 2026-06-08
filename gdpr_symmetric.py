"""
RQ3 본실험 (대칭 설계): 같은 350 EN 시나리오를 GDPR 프롬프트로 판단.
PIPA 판단(results_v1.json의 gemma2-9b__en)과 1:1 비교 → 관할권 민감도/혼동.
GDPR 프롬프트는 PIPA와 동일한 6개 선택지를 주되, 지정기관 가명처리는
"GDPR엔 해당 요건 없음"을 명시 → 모델이 GDPR에서 STOP_PSEUDO를 고르면 '혼동'.
"""
import json, urllib.request
from evaluate import parse_decision

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:9b-instruct-q4_0"

GDPR_PROMPT = """You are an EU GDPR compliance expert. Decide whether the following personal-data processing is permissible under the EU GDPR, or at which step it must STOP. Choose exactly one of six labels.

Criteria:
1) No lawful basis (Art.6) -> STOP_LAWFUL_BASIS
2) Special-category data without explicit consent / Art.9 exception -> STOP_SENSITIVE_CONSENT
3) Purpose not specified (Art.5 purpose limitation) -> STOP_PURPOSE
4) STOP_PSEUDO_INSTITUTION and STOP_SAFETY are NOT GDPR requirements: the GDPR does NOT require pseudonymization to be done only through a government-designated institution, nor a separate safety review as a gatekeeping step. Do NOT choose these unless GDPR truly requires it.
5) Otherwise -> ALLOW

Scenario:
{scenario}

JSON only:
{{"decision": "ALLOW|STOP_LAWFUL_BASIS|STOP_SENSITIVE_CONSENT|STOP_PURPOSE|STOP_PSEUDO_INSTITUTION|STOP_SAFETY"}}
"""

def query(prompt):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0}, "format": "json"}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]

def main():
    cases = json.load(open("dataset_v1.json", encoding="utf-8"))
    gdpr_pred = {}
    for i, c in enumerate(cases):
        gdpr_pred[c["id"]] = parse_decision(query(GDPR_PROMPT.format(scenario=c["scenario_en"])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}", flush=True)
    json.dump(gdpr_pred, open("gdpr_pred.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("→ gdpr_pred.json 저장")

if __name__ == "__main__":
    main()
