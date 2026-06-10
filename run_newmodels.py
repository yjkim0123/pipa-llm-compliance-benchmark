"""새 모델(mistral/exaone/qwen3b) 본실험 → results_v1.json에 append. timeout 60+재시도."""
import json, time, urllib.request
from evaluate import PROMPT, parse_decision

NEW = ["mistral:7b", "exaone3.5:7.8b", "qwen2.5:3b-instruct"]
OLLAMA="http://localhost:11434/api/generate"

def q(model, prompt):
    for a in range(3):
        try:
            data=json.dumps({"model":model,"prompt":prompt,"stream":False,
                             "options":{"temperature":0.0},"format":"json"}).encode()
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=60) as r:
                return json.loads(r.read())["response"]
        except Exception:
            if a==2: return "{}"
            time.sleep(1)

def log(m): 
    open("newmodels.log","a").write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

cases=json.load(open("dataset_v1.json",encoding="utf-8"))
R=json.load(open("results_v1.json",encoding="utf-8"))
for model in NEW:
    for lang in ["ko","en"]:
        key=f"{model}__{lang}"
        rows,correct=[],0; t0=time.time()
        for i,c in enumerate(cases):
            scen=c["scenario_ko"] if lang=="ko" else c["scenario_en"]
            gold=c["gold_class"]
            pred=parse_decision(q(model,PROMPT.format(scenario=scen)))
            ok=(pred==gold); correct+=ok
            rows.append({"id":c["id"],"gold":gold,"pred":pred,"ok":ok,"sensitive_type":c["sensitive_type"]})
            if (i+1)%100==0: log(f"  {key} {i+1}/{len(cases)} acc{correct/(i+1):.0%}")
        R[key]={"model":model,"lang":lang,"accuracy":correct/len(cases),"correct":correct,
                "n":len(cases),"secs":round(time.time()-t0),"rows":rows}
        log(f"DONE {key}: {correct/len(cases):.1%} in {R[key]['secs']}s")
        json.dump(R,open("results_v1.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
log("ALL NEWMODELS DONE")
