import sys
from .engine import SecurityValidationEngine
from .registry import SecurityRegistry
from .audit.authentication import AuthenticationAudit
from .audit.authorization import AuthorizationAudit
from .audit.api import APISecurityAudit
from .audit.frontend import FrontendSecurityAudit
from .audit.deployment import DeploymentSecurityAudit
from .certification.gates import CertificationGate
from .certification.manager import CertificationManager

def main(dry_run: bool = False):
    registry = SecurityRegistry()
    registry.register("auth", AuthenticationAudit().audit, priority=10)
    registry.register("authz", AuthorizationAudit().audit, priority=20, depends_on=("auth",))
    registry.register("api", APISecurityAudit().audit, priority=30)
    registry.register("frontend", FrontendSecurityAudit().audit, priority=40)
    registry.register("deployment", DeploymentSecurityAudit().audit, priority=50)

    engine = SecurityValidationEngine(registry)
    findings = engine.execute()
    
    cert_manager = CertificationManager(CertificationGate())
    report = cert_manager.generate_report(findings)

    if dry_run:
        print("Dry run completed successfully.")
        print(report.to_json())
    else:
        print(report.to_json())

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run)
