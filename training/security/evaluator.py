from typing import Dict, Any
from training.security.metadata import SecurityDecision, Principal, Resource, Severity
from training.security.compiler import CompiledPolicyTree

class PolicyEvaluator:
    @staticmethod
    def evaluate(principal: Principal, action: str, resource: Resource, compiled_tree: CompiledPolicyTree, environment: Dict[str, Any]) -> SecurityDecision:
        trace = []
        allow = False
        reason = "Implicit Deny"
        matched_policy = None
        
        for policy in compiled_tree.policies:
            # Very simplistic condition evaluation
            match = True
            for k, v in policy.conditions.items():
                env_val = environment.get(k)
                res_val = resource.attributes.attributes.get(k)
                if env_val != v and res_val != v:
                    match = False
                    break
            
            if match:
                trace.append(f"Matched {policy.name} ({policy.effect})")
                if policy.effect == 'deny':
                    return SecurityDecision(
                        allow=False,
                        reason=f"Explicit Deny by {policy.name}",
                        matched_policy=policy.name,
                        evaluation_trace=tuple(trace),
                        severity=Severity.WARNING
                    )
                else:
                    allow = True
                    reason = f"Allowed by {policy.name}"
                    matched_policy = policy.name
                    # Note: we continue evaluating because a subsequent deny might override this

        return SecurityDecision(
            allow=allow,
            reason=reason,
            matched_policy=matched_policy,
            evaluation_trace=tuple(trace),
            severity=Severity.INFO if allow else Severity.WARNING
        )
