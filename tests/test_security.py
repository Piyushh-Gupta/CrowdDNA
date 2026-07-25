from training.security.metadata import (
    Identity, Principal, Resource, ResourceIdentifier, ResourceType,
    ResourceAttributes, Policy
)
from training.security.context import SecurityContext
from training.security.pipeline import SecurityPipeline
from training.security.secrets import SecretReference, SecretResolver, EnvSecretProvider
from training.security.registry import SecurityRegistry

def test_authentication():
    pipeline = SecurityPipeline()
    pipeline.bootstrap()
    identity = pipeline.engine.authenticate("local", {"username": "admin", "password": "secret"})
    assert identity is not None
    assert identity.identity_id == "admin"
    
    bad_identity = pipeline.engine.authenticate("local", {"username": "admin", "password": "wrong"})
    assert bad_identity is None

def test_authorization():
    pipeline = SecurityPipeline()
    pipeline.bootstrap()
    
    identity = Identity("admin", "local")
    principal = Principal(identity, ("execute_node",))
    context = SecurityContext("wf-1", "exec-1", "corr-1")
    
    decision = pipeline.authorize_workflow_node("node-1", principal, context)
    assert decision.allow is True
    assert decision.matched_role == "execute_node"

def test_policy_deny_precedence():
    pipeline = SecurityPipeline()
    pipeline.bootstrap()
    
    # Add a policy that explicitly denies
    policy1 = Policy("AllowAll", "allow", {})
    policy2 = Policy("DenyAdminNode", "deny", {"node_id": "restricted"})
    
    pipeline.engine.load_policies([policy1, policy2])
    
    identity = Identity("admin", "local")
    principal = Principal(identity, ("execute_node",))
    context = SecurityContext("wf-1", "exec-1", "corr-1", environment_data={})
    
    resource = Resource(
        identifier=ResourceIdentifier(ResourceType("node"), "restricted"),
        attributes=ResourceAttributes({"node_id": "restricted"})
    )
    
    decision = pipeline.engine.authorize(principal, "run_workflow", resource, context)
    assert decision.allow is False
    assert decision.reason == "Explicit Deny by DenyAdminNode"

def test_secret_resolution():
    registry = SecurityRegistry()
    registry.register_secret_provider("env", EnvSecretProvider({"MY_SECRET": "topsecret"}))
    
    resolver = SecretResolver()
    ref = SecretReference("MY_SECRET", "env")
    
    val = resolver.resolve(ref)
    assert val == "topsecret"

def test_audit_generation():
    pipeline = SecurityPipeline()
    pipeline.bootstrap()
    
    identity = Identity("admin", "local")
    principal = Principal(identity, ("execute_node",))
    context = SecurityContext("wf-1", "exec-1", "corr-1")
    
    pipeline.authorize_workflow_node("node-1", principal, context)
    
    records = pipeline.audit_logger.get_records()
    assert len(records) == 1
    assert records[0].action == "run_workflow"
    assert records[0].decision.allow is True
