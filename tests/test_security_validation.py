from crowdflow_dna.security_validation.registry import SecurityRegistry
from crowdflow_dna.security_validation.audit.authentication import AuthenticationAudit
from crowdflow_dna.security_validation.metadata import Severity
import threading

def test_registry_ordering():
    reg = SecurityRegistry()
    def fake_auth(): return []
    def fake_authz(): return []
    reg.register("auth", fake_auth, 10)
    reg.register("authz", fake_authz, 20)
    
    validators = reg.get_ordered_validators()
    assert validators[0] == fake_auth
    assert validators[1] == fake_authz

def test_registry_thread_safety():
    reg = SecurityRegistry()
    def worker():
        for i in range(100):
            reg.register(f"w_{threading.get_ident()}_{i}", lambda: [], priority=i)
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(reg.get_ordered_validators()) == 1000

def test_authentication_audit():
    audit = AuthenticationAudit()
    findings = audit.audit()
    assert len(findings) > 0
    assert findings[0].severity == Severity.HIGH
    assert findings[0].passed is True
