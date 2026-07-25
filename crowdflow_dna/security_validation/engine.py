from typing import Tuple
from .registry import SecurityRegistry
from .metadata import SecurityFinding

class SecurityValidationEngine:
    def __init__(self, registry: SecurityRegistry):
        self.registry = registry

    def execute(self) -> Tuple[SecurityFinding, ...]:
        validators = self.registry.get_ordered_validators()
        findings = []
        for validator in validators:
            findings.extend(validator())
        return tuple(findings)
