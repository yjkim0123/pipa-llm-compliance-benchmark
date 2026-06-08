"""
Personal Data Compliance Decision Engine (PIPA-based)
Ground truth generator for the benchmark dataset.
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class SensitiveType(Enum):
    NONE = "none"
    HEALTH = "health"
    RACE = "race"
    POLITICAL = "political"
    BIOMETRIC = "biometric"
    BELIEF = "belief"
    SEXUAL = "sexual"
    CRIMINAL = "criminal"

class Jurisdiction(Enum):
    PIPA = "PIPA"   # 한국 개인정보보호법
    GDPR = "GDPR"   # EU GDPR

@dataclass
class Case:
    # N1: Lawful basis
    publicly_available: bool
    other_lawful_ground: bool

    # N2: Sensitivity
    sensitive_type: SensitiveType  # NONE = 일반정보

    # N3: Consent
    consent_notified: bool        # 일반: 수집·이용 고지
    separate_consent: bool        # 민감: 별도 동의

    # N4: Purpose
    purpose_specified: bool

    # N5: Within purpose
    within_purpose: bool

    # N6: Other purpose lawful ground
    other_purpose_lawful: bool

    # N8: Pseudonymization path
    via_designated_institution: bool
    safety_check_passed: bool

    # N9: Preservation
    preservation_required: bool

    # Metadata
    jurisdiction: Jurisdiction = Jurisdiction.PIPA
    language: str = "KO"
    scenario_text: str = ""
    scenario_text_en: str = ""

@dataclass
class Decision:
    node_decisions: dict      # node_id -> chosen branch
    final_outcome: str        # COMPLETE / STOP:{reason}
    path: list                # ordered list of nodes visited

def decide(case: Case) -> Decision:
    path = []
    nodes = {}

    # N0: Content → Is this personal data? (assumed YES for all cases in dataset)

    # N1: Lawful Basis
    path.append("N1_lawful_basis")
    if case.publicly_available:
        nodes["N1"] = "publicly_available=YES"
    elif case.other_lawful_ground:
        nodes["N1"] = "other_lawful_ground=YES"
    else:
        nodes["N1"] = "NO_lawful_basis"
        return Decision(nodes, "STOP:no_lawful_basis", path)

    # N2: Sensitivity
    path.append("N2_sensitivity")
    is_sensitive = case.sensitive_type != SensitiveType.NONE
    nodes["N2"] = f"sensitive={is_sensitive} type={case.sensitive_type.value}"

    # N3: Consent
    path.append("N3_consent")
    if is_sensitive:
        if not case.separate_consent:
            nodes["N3"] = "STOP:sensitive_no_separate_consent"
            return Decision(nodes, "STOP:sensitive_no_separate_consent", path)
        nodes["N3"] = "separate_consent=YES"
    else:
        if not case.consent_notified:
            nodes["N3"] = "STOP:general_no_consent_notification"
            return Decision(nodes, "STOP:no_consent_notification", path)
        nodes["N3"] = "consent_notified=YES"

    # N4: Purpose specified
    path.append("N4_purpose_specified")
    if not case.purpose_specified:
        nodes["N4"] = "STOP:purpose_not_specified"
        return Decision(nodes, "STOP:purpose_not_specified", path)
    nodes["N4"] = "purpose_specified=YES"

    # N5: Within purpose
    path.append("N5_within_purpose")
    if case.within_purpose:
        nodes["N5"] = "within_purpose=YES"
        path.append("N7_use_provision")
        nodes["N7"] = "POSSIBLE_USE_PROVISION"
    else:
        nodes["N5"] = "within_purpose=NO"
        # N6: Other purpose lawful ground
        path.append("N6_other_purpose_lawful")
        if case.other_purpose_lawful:
            nodes["N6"] = "other_purpose_lawful=YES"
            path.append("N7_use_provision")
            nodes["N7"] = "POSSIBLE_USE_PROVISION(other_purpose)"
        else:
            nodes["N6"] = "other_purpose_lawful=NO"
            # N8: Pseudonymization
            path.append("N8_pseudonymization")
            if case.via_designated_institution:
                nodes["N8"] = "via_designated_institution=YES"
                path.append("N8b_safety_check")
                if case.safety_check_passed:
                    nodes["N8b"] = "safety_check=YES"
                    nodes["N7"] = "PSEUDONYMIZATION_USE"
                else:
                    nodes["N8b"] = "STOP:safety_check_failed"
                    return Decision(nodes, "STOP:safety_check_failed", path)
            else:
                nodes["N8"] = "STOP:no_designated_institution(external_aid_needed)"
                return Decision(nodes, "STOP:no_designated_institution", path)

    # N9: Preservation requirement
    path.append("N9_preservation")
    if case.preservation_required:
        nodes["N9"] = "preservation_required=YES"
        return Decision(nodes, "PRESERVE_THEN_COMPLETE", path)
    else:
        nodes["N9"] = "preservation_required=NO"
        path.append("N10_secure_disposal")
        nodes["N10"] = "SECURELY_DISPOSAL"
        return Decision(nodes, "SECURE_DISPOSAL_COMPLETE", path)


# ── Pilot Dataset (10 cases) ──────────────────────────────────────────────────

PILOT_CASES = [
    # 1. 공개정보, 일반, 목적내 이용, 보존필요없음 → COMPLETE (정상 경로)
    Case(
        publicly_available=True, other_lawful_ground=False,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="온라인 커뮤니티에 공개된 사용자 게시글을 마케팅 분석에 활용한다.",
        scenario_text_en="A company uses publicly posted community content for marketing analysis.",
    ),
    # 2. 적법근거 없음 → STOP:no_lawful_basis
    Case(
        publicly_available=False, other_lawful_ground=False,
        sensitive_type=SensitiveType.NONE,
        consent_notified=False, separate_consent=False,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="고객의 동의 없이, 별도 법적 근거도 없이 구매이력을 수집한다.",
        scenario_text_en="A company collects purchase history without consent or any legal basis.",
    ),
    # 3. 민감정보(건강), 별도동의 없음 → STOP:sensitive_no_separate_consent
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.HEALTH,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="병원이 환자 건강정보를 수집할 때 일반동의만 받고 별도 동의를 받지 않았다.",
        scenario_text_en="A hospital collects patient health data with only general consent, no separate consent.",
    ),
    # 4. 민감정보(인종), 별도동의 있음, 목적내 이용 → COMPLETE
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.RACE,
        consent_notified=True, separate_consent=True,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="연구기관이 민족별 의약품 반응 연구를 위해 별도 동의를 받고 인종정보를 수집한다.",
        scenario_text_en="A research institute collects racial data for drug response research with separate consent.",
    ),
    # 5. 수집목적 미특정 → STOP:purpose_not_specified
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=False, within_purpose=False,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="기업이 '향후 활용 가능성'만 명시하고 이메일 주소를 수집한다.",
        scenario_text_en="A company collects email addresses stating only 'possible future use' as the purpose.",
    ),
    # 6. 목적 외 이용, 목적외 적법근거 있음 → COMPLETE (other purpose)
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=False,
        other_purpose_lawful=True,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="수집 목적 외 이용이지만 법원 명령에 따라 수사기관에 정보를 제공한다.",
        scenario_text_en="Data is shared with authorities outside original purpose under a court order.",
    ),
    # 7. 목적 외, 적법근거 없음, 지정기관 통해 가명처리, 안전성 검토 통과 → COMPLETE
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=False,
        other_purpose_lawful=False,
        via_designated_institution=True, safety_check_passed=True,
        preservation_required=False,
        scenario_text="기업이 목적 외 통계분석을 위해 데이터전문기관을 통해 가명처리 후 결합한다.",
        scenario_text_en="A company pseudonymizes data via a designated institution for statistical analysis outside original purpose.",
    ),
    # 8. 가명처리, 지정기관 없음 → STOP:no_designated_institution
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=False,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="기업이 지정기관을 거치지 않고 자체적으로 가명처리 후 제3자에게 제공한다.",
        scenario_text_en="A company pseudonymizes data internally without a designated institution and shares with a third party.",
    ),
    # 9. 정상 경로, 법정보존의무 있음 → PRESERVE_THEN_COMPLETE
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.NONE,
        consent_notified=True, separate_consent=False,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=True,
        scenario_text="전자상거래 업체가 거래기록을 5년간 법정 보존 의무에 따라 보관한다.",
        scenario_text_en="An e-commerce company retains transaction records for 5 years as required by law.",
    ),
    # 10. 민감정보(정치성향), 별도동의, 목적내, 보존불필요 → COMPLETE
    Case(
        publicly_available=False, other_lawful_ground=True,
        sensitive_type=SensitiveType.POLITICAL,
        consent_notified=True, separate_consent=True,
        purpose_specified=True, within_purpose=True,
        other_purpose_lawful=False,
        via_designated_institution=False, safety_check_passed=False,
        preservation_required=False,
        scenario_text="정당이 당원의 정치적 성향 정보를 별도 동의를 받아 내부 활동 목적으로 수집한다.",
        scenario_text_en="A political party collects members' political orientation data with separate consent for internal activities.",
    ),
]


if __name__ == "__main__":
    import json
    print("=== Pilot Dataset: Ground Truth Labels ===\n")
    results = []
    for i, case in enumerate(PILOT_CASES, 1):
        d = decide(case)
        results.append({
            "id": i,
            "scenario_ko": case.scenario_text,
            "scenario_en": case.scenario_text_en,
            "sensitive_type": case.sensitive_type.value,
            "outcome": d.final_outcome,
            "path": d.path,
            "node_decisions": d.node_decisions,
        })
        print(f"Case {i:2d}: {d.final_outcome}")
        print(f"  Path: {' → '.join(d.path)}")
        print(f"  KO: {case.scenario_text}")
        print()
    with open("pilot_dataset.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("→ pilot_dataset.json 저장 완료")
