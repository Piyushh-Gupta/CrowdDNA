from crowdflow_dna.security_validation.compliance.manager import ComplianceManager
from crowdflow_dna.security_validation.metadata import Control

def test_compliance_manager():
    mgr = ComplianceManager()
    mgr.register_control(Control(id="C1", description="Control 1"))
    
    results = mgr.evaluate_controls(findings=())
    assert len(results) == 1
    assert results[0].control_id == "C1"
