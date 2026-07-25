from typing import Tuple
from .metadata import PolicyRequirement, ControlResult

class SecurityPolicyEngine:
    def evaluate(self, policy: PolicyRequirement, results: Tuple[ControlResult, ...]) -> bool:
        result_map = {r.control_id: r.passed for r in results}
        return all(result_map.get(c, False) for c in policy.required_controls)
