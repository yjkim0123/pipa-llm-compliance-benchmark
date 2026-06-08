"""
노드 단위 평가: LLM이 각 법적 요건(노드)을 올바르게 판단하는가.
최종 outcome(법적 추론 통합)과 분리하여, KO 약점이
'사실 인식(노드)' 단계인지 '추론 통합' 단계인지 진단.
항상 텍스트에 단서가 있는 5개 핵심 노드만 채점.
"""
import json, urllib.request, sys
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api/generate"

NODES = ["lawful_basis", "is_sensitive", "consent_ok", "purpose_specified", "within_purpose"]

def gold_nodes(v):
    return {
        "lawful_basis": v["publicly_available"] or v["other_lawful_ground"],
        "is_sensitive": v["is_sensitive"],
        "consent_ok": (v["separate_consent"] if v["is_sensitive"] else v["consent_notified"]),
        "purpose_specified": v["purpose_specified"],
        "within_purpose": v["within_purpose"],
    }

PROMPT = """당신은 한국 개인정보보호법(PIPA) 전문가입니다.
아래 시나리오를 읽고, 각 항목이 참인지 거짓인지 판단하세요. 시나리오에 서술된 사실에 근거하세요.

항목:
- lawful_basis: 수집의 적법근거(공개정보 또는 법령상 근거)가 있는가
- is_sensitive: 수집 대상이 민감정보(건강/인종/정치성향/생체/사상/성생활/범죄)인가
- consent_ok: 필요한 동의가 적절히 확보되었는가 (민감정보면 별도동의, 일반정보면 수집·이용 동의)
- purpose_specified: 수집 목적이 구체적으로 특정되었는가
- within_purpose: 당초 수집 목적 범위 내의 이용인가

시나리오:
{scenario}

JSON으로만 답하세요. 설명 금지.
{{"lawful_basis": true/false, "is_sensitive": true/false, "consent_ok": true/false, "purpose_specified": true/false, "within_purpose": true/false}}
"""

def query(model, prompt):
    data = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0}, "format": "json"}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]

def main(model="gemma2:9b-instruct-q4_0", lang="ko", dataset="dataset_v1.json"):
    cases = json.load(open(dataset, encoding="utf-8"))
    node_correct = defaultdict(int)
    node_total = defaultdict(int)
    all_nodes_correct = 0  # 5개 노드 전부 맞은 케이스 수
    for c in cases:
        scen = c["scenario_ko"] if lang == "ko" else c["scenario_en"]
        g = gold_nodes(c["vars"])
        try:
            pred = json.loads(query(model, PROMPT.format(scenario=scen)))
        except Exception:
            pred = {}
        all_ok = True
        for n in NODES:
            node_total[n] += 1
            if bool(pred.get(n)) == bool(g[n]):
                node_correct[n] += 1
            else:
                all_ok = False
        all_nodes_correct += all_ok
    print(f"\n=== {model} ({lang}) node-level accuracy ===")
    for n in NODES:
        print(f"  {n:18s} {node_correct[n]/node_total[n]:.1%}")
    print(f"  [ALL 5 nodes correct] {all_nodes_correct/len(cases):.1%}")
    result = {n: node_correct[n]/node_total[n] for n in NODES}
    result["_all_nodes"] = all_nodes_correct / len(cases)
    return result

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:9b-instruct-q4_0"
    out = {}
    for lang in ["ko", "en"]:
        out[lang] = main(model, lang)
    json.dump(out, open("node_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n→ node_result.json 저장")
    # KO vs EN 노드별 갭
    print("\n=== KO−EN node gap ===")
    for n in NODES + ["_all_nodes"]:
        print(f"  {n:18s} {(out['ko'][n]-out['en'][n])*100:+.1f}%p")
