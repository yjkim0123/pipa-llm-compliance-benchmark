"""
Dataset Generator — 변수 조합 + 도메인 템플릿으로 PIPA 시나리오 자동 생성.
ground truth는 rule_engine.decide()로 결정론적 산출.
KO/EN 병렬.
"""
import json, random
from rule_engine import Case, SensitiveType, decide

# gold_to_class는 evaluate.py에 있으므로 여기서 재정의
def gold_to_class(outcome: str) -> str:
    if outcome.startswith("STOP:no_lawful_basis"): return "STOP_LAWFUL_BASIS"
    if outcome.startswith("STOP:sensitive"): return "STOP_SENSITIVE_CONSENT"
    if outcome.startswith("STOP:purpose"): return "STOP_PURPOSE"
    if outcome.startswith("STOP:no_designated"): return "STOP_PSEUDO_INSTITUTION"
    if outcome.startswith("STOP:safety"): return "STOP_SAFETY"
    return "ALLOW"

random.seed(42)  # 재현성 (Date/random 제약은 스크립트 직접 실행이라 무관)

# 도메인: (주체, 데이터, 민감유형)
DOMAINS = [
    ("병원", "환자 진료 기록", SensitiveType.HEALTH, "a hospital", "patient medical records"),
    ("건강검진센터", "건강검진 결과", SensitiveType.HEALTH, "a health checkup center", "health screening results"),
    ("정당", "당원의 정치적 성향", SensitiveType.POLITICAL, "a political party", "members' political affiliation"),
    ("연구기관", "유전자·생체 정보", SensitiveType.BIOMETRIC, "a research institute", "genetic and biometric data"),
    ("종교단체", "신도의 종교·사상 정보", SensitiveType.BELIEF, "a religious organization", "members' religious beliefs"),
    ("인사컨설팅사", "지원자의 인종·출신 정보", SensitiveType.RACE, "an HR consulting firm", "applicants' racial origin data"),
    ("쇼핑몰", "구매 이력", SensitiveType.NONE, "an online shopping mall", "purchase history"),
    ("통신사", "통화 내역", SensitiveType.NONE, "a telecom carrier", "call records"),
    ("은행", "계좌 거래 내역", SensitiveType.NONE, "a bank", "account transaction records"),
    ("SNS플랫폼", "공개 게시물", SensitiveType.NONE, "a social media platform", "public posts"),
    ("학교", "학생 성적 정보", SensitiveType.NONE, "a school", "student grade records"),
    ("보험사", "고객 가입 정보", SensitiveType.NONE, "an insurance company", "customer enrollment data"),
]

def frag_lawful(c, en=False):
    if c.publicly_available:
        return "공개된 정보를 바탕으로" if not en else "based on publicly available information"
    if c.other_lawful_ground:
        return "법령상 근거에 따라" if not en else "under a statutory legal basis"
    return "별도의 적법 근거 없이" if not en else "without any legal basis"

def frag_consent(c, sensitive, en=False):
    if sensitive:
        if c.separate_consent:
            return "민감정보 처리에 대한 별도 동의를 받아" if not en else "with separate consent for sensitive data"
        return "일반 동의만 받고 별도 동의 없이" if not en else "with only general consent and no separate consent"
    else:
        if c.consent_notified:
            return "수집·이용 목적을 고지하고 동의를 받아" if not en else "after notifying and obtaining consent"
        return "동의 고지 절차 없이" if not en else "without any consent notification"

def frag_purpose(c, en=False):
    if not c.purpose_specified:
        return "구체적 수집 목적을 특정하지 않고" if not en else "without specifying a concrete purpose"
    return "수집 목적을 명확히 특정하여" if not en else "with a clearly specified purpose"

def frag_use(c, en=False):
    if c.within_purpose:
        return "당초 수집 목적 범위 내에서 이용한다" if not en else "and uses it within the original purpose"
    base = "당초 목적과 다른 목적으로" if not en else "for a purpose different from the original"
    if c.other_purpose_lawful:
        tail = " 법령상 허용 근거에 따라 이용한다" if not en else ", permitted under a separate legal ground"
    else:
        if c.via_designated_institution:
            sc = "안전성 검토를 거쳐" if c.safety_check_passed else "안전성 검토 없이"
            sc_en = "after a safety review" if c.safety_check_passed else "without a safety review"
            tail = f" 데이터전문기관(지정기관)을 통해 가명처리하여 {sc} 결합·이용한다" if not en else f", pseudonymized through a designated institution {sc_en}"
        else:
            tail = " 지정기관을 거치지 않고 자체 가명처리하여 제3자에게 제공한다" if not en else ", pseudonymized internally without a designated institution and shared with a third party"
    return base + tail

def frag_preserve(c, en=False):
    if c.preservation_required:
        return " 이후 법정 보존의무에 따라 일정 기간 보관한다." if not en else " It is then retained for a legally mandated period."
    return "" if not en else ""

def build_text(subj, data, c, en=False):
    if not en:
        s = f"{subj}가 {frag_lawful(c)} {data}을(를) {frag_consent(c, c.sensitive_type!=SensitiveType.NONE)} 수집하고, {frag_purpose(c)} {frag_use(c)}.{frag_preserve(c)}"
    else:
        s = f"{subj} collects {data} {frag_lawful(c, True)}, {frag_consent(c, c.sensitive_type!=SensitiveType.NONE, True)}, {frag_purpose(c, True)}, {frag_use(c, True)}.{frag_preserve(c, True)}"
    return s.strip()

def random_case(domain):
    subj_ko, data_ko, stype, subj_en, data_en = domain
    # 민감 도메인은 sensitive 유지, 일반 도메인은 NONE
    publicly = random.random() < 0.25
    other_ground = random.random() < 0.7 if not publicly else False
    c = Case(
        publicly_available=publicly,
        other_lawful_ground=other_ground,
        sensitive_type=stype,
        consent_notified=random.random() < 0.8,
        separate_consent=random.random() < 0.6,
        purpose_specified=random.random() < 0.8,
        within_purpose=random.random() < 0.55,
        other_purpose_lawful=random.random() < 0.5,
        via_designated_institution=random.random() < 0.5,
        safety_check_passed=random.random() < 0.6,
        preservation_required=random.random() < 0.3,
    )
    c.scenario_text = build_text(subj_ko, data_ko, c, en=False)
    c.scenario_text_en = build_text(subj_en, data_en, c, en=True)
    return c

def main(n=150):
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        dom = random.choice(DOMAINS)
        c = random_case(dom)
        d = decide(c)
        cls = gold_to_class(d.final_outcome)
        key = (c.scenario_text,)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "id": len(out) + 1,
            "scenario_ko": c.scenario_text,
            "scenario_en": c.scenario_text_en,
            "sensitive_type": c.sensitive_type.value,
            "outcome": d.final_outcome,
            "gold_class": cls,
            "path": d.path,
            "vars": {
                "publicly_available": c.publicly_available,
                "other_lawful_ground": c.other_lawful_ground,
                "is_sensitive": c.sensitive_type != SensitiveType.NONE,
                "separate_consent": c.separate_consent,
                "consent_notified": c.consent_notified,
                "purpose_specified": c.purpose_specified,
                "within_purpose": c.within_purpose,
                "other_purpose_lawful": c.other_purpose_lawful,
                "via_designated_institution": c.via_designated_institution,
                "safety_check_passed": c.safety_check_passed,
                "preservation_required": c.preservation_required,
            },
        })
    json.dump(out, open("dataset_v1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 분포 출력
    from collections import Counter
    dist = Counter(r["gold_class"] for r in out)
    stype = Counter(r["sensitive_type"] for r in out)
    print(f"생성: {len(out)}개 → dataset_v1.json\n")
    print("== gold_class 분포 ==")
    for k, v in dist.most_common():
        print(f"  {k:26s} {v}")
    print("\n== sensitive_type 분포 ==")
    for k, v in stype.most_common():
        print(f"  {k:12s} {v}")
    print("\n== 샘플 3개 ==")
    for r in out[:3]:
        print(f"[{r['gold_class']}] {r['scenario_ko']}")

if __name__ == "__main__":
    main(150)
