from training.security.metadata import (
    Identity, SecurityDecision, Principal, Resource, 
    ResourceType, ResourceIdentifier, ResourceAttributes
)
from training.security.context import SecurityContext
from training.security.registry import SecurityRegistry
from training.security.audit import AuditLogger
from training.security.engine import SecurityEngine
from training.security.permissions import PermissionRegistry, Permission, Capability
from training.security.authentication import LocalAuthProvider

class SecurityPipeline:
    """Security integration layer."""
    
    def __init__(self):
        self.registry = SecurityRegistry()
        self.audit_logger = AuditLogger(strict=False)
        self.engine = SecurityEngine(self.registry, self.audit_logger)
        
    def bootstrap(self):
        # Setup dummy auth provider for tests
        self.registry.register_auth_provider("local", LocalAuthProvider({"admin": Identity("admin", "local")}))
        
        # Setup dummy capabilities
        PermissionRegistry.register_permission(Permission("execute_node", "Can execute node"))
        PermissionRegistry.register_capability(Capability("run_workflow", ("execute_node",)))
        
        # Mapping
        self.engine.set_role_mapping({"admin": ("execute_node",)})
        
    def authorize_plugin(self, plugin_name: str, principal: Principal, context: SecurityContext) -> SecurityDecision:
        resource = Resource(
            identifier=ResourceIdentifier(ResourceType("plugin"), plugin_name),
            attributes=ResourceAttributes()
        )
        # Assuming loading a plugin requires "load_plugin" action, which we haven't mapped, we mock an action
        # For tests, we use the capability we registered
        return self.engine.authorize(principal, "run_workflow", resource, context)
        
    def authorize_workflow_node(self, node_id: str, principal: Principal, context: SecurityContext) -> SecurityDecision:
        resource = Resource(
            identifier=ResourceIdentifier(ResourceType("node"), node_id),
            attributes=ResourceAttributes()
        )
        return self.engine.authorize(principal, "run_workflow", resource, context)
