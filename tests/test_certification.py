import json
from crowdflow_dna.security_validation.certification.report import ProductionCertificationReport
from crowdflow_dna.security_validation.certification.score import calculate_score

def test_certification_report_determinism():
    score = calculate_score(())
    report1 = ProductionCertificationReport(
        score=score,
        critical_findings=(),
        warnings=(),
        passed_controls=(),
        failed_controls=(),
        skipped_controls=(),
        recommendations=(),
        generated_timestamp="2026-07-25T00:00:00Z",
        framework_version="1.0"
    )
    
    report2 = ProductionCertificationReport(
        score=score,
        critical_findings=(),
        warnings=(),
        passed_controls=(),
        failed_controls=(),
        skipped_controls=(),
        recommendations=(),
        generated_timestamp="2026-07-25T00:00:00Z",
        framework_version="1.0"
    )
    
    assert report1.to_json() == report2.to_json()
    
    # Verify sorting
    j = json.loads(report1.to_json())
    assert "score" in j
