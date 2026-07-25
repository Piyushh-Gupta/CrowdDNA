from training.security.metadata import Principal, Resource, SecurityDecision, Severity
from training.security.permissions import PermissionRegistry

class AuthorizationEngine:
    @staticmethod
    def evaluate(principal: Principal, action: str, resource: Resource) -> SecurityDecision:
        trace = []
        allow = False
        reason = "Implicit Deny"
        matched_role = None
        matched_permission = None

        # Resolve capability
        try:
            capability = PermissionRegistry.get_capability(action)
            trace.append(f"Action '{action}' mapped to capability '{capability.name}'")
        except KeyError:
            return SecurityDecision(allow=False, reason=f"Unknown action {action}", evaluation_trace=tuple(trace), severity=Severity.WARNING)

        # In a real system, roles would map to permissions. For now we assume a simple role -> permission string mapping logic
        # For this implementation, we will assume Principal holds permissions directly in 'roles' just for simplicity, 
        # or we check if the required permission is in the principal's roles.
        
        for required_perm in capability.required_permissions:
            if required_perm in principal.roles:
                allow = True
                reason = "Granted by RBAC"
                matched_role = required_perm # simplificaton
                matched_permission = required_perm
                trace.append(f"Principal has required permission: {required_perm}")
                break
        
        if not allow:
            trace.append(f"Principal lacks required permissions: {capability.required_permissions}")

        return SecurityDecision(
            allow=allow,
            reason=reason,
            matched_role=matched_role,
            matched_permission=matched_permission,
            evaluation_trace=tuple(trace),
            severity=Severity.INFO if allow else Severity.WARNING
        )
