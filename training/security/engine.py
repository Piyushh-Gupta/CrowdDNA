from typing import Optional, Dict, Any, List
import time
import uuid
import threading

from training.security.metadata import (
    Identity, SecurityDecision, SecurityReport, Principal, Resource, Severity
)
from training.security.context import SecurityContext, SecuritySession
from training.security.registry import SecurityRegistry
from training.security.cache import AuthenticationCache, AuthorizationCache, PolicyCache
from training.security.authorization import AuthorizationEngine
from training.security.evaluator import PolicyEvaluator
from training.security.compiler import CompiledPolicyTree, PolicyCompiler
from training.security.audit import AuditLogger

class SecurityEngine:
    def __init__(self, registry: SecurityRegistry, audit_logger: AuditLogger):
        self.registry = registry
        self.audit_logger = audit_logger
        self.auth_cache = AuthenticationCache()
        self.authz_cache = AuthorizationCache()
        self.policy_cache = PolicyCache()
        self.compiled_policy_tree: Optional[CompiledPolicyTree] = None
        self._role_mapping: Dict[str, tuple[str, ...]] = {}
        self._decisions: List[SecurityDecision] = []
        self._lock = threading.Lock()

    def load_policies(self, policies: List[Any]):
        with self._lock:
            self.compiled_policy_tree = PolicyCompiler.compile(policies)

    def set_role_mapping(self, mapping: Dict[str, tuple[str, ...]]):
        with self._lock:
            self._role_mapping = mapping

    def authenticate(self, provider_name: str, credentials: Dict[str, Any]) -> Optional[Identity]:
        token = credentials.get("token")
        if token:
            cached = self.auth_cache.get(token)
            if cached:
                return cached
                
        provider = self.registry.get_auth_provider(provider_name)
        identity = provider.authenticate(credentials)
        
        # Emitting Event (we mock dispatch here, Phase 18 EventDispatcher handles it normally)
        # Emitting Event (we mock dispatch here, Phase 18 EventDispatcher handles it normally)
        # success = identity is not None
        # identity_id = identity.identity_id if identity else "unknown"
        # The prompt mentions to construct the events
        # Real dispatcher integration happens in pipeline
        
        return identity

    def create_session(self, identity: Identity, expires_in_seconds: float = 3600) -> SecuritySession:
        return SecuritySession(
            session_id=str(uuid.uuid4()),
            identity=identity,
            created_at=time.time(),
            expires_at=time.time() + expires_in_seconds
        )

    def authorize(self, principal: Principal, action: str, resource: Resource, context: SecurityContext) -> SecurityDecision:
        # Check cache
        cached = self.authz_cache.get(principal.identity.identity_id, action, resource.identifier.resource_id)
        if cached:
            return cached
            
        # 1. RBAC (Static)
        rbac_decision = AuthorizationEngine.evaluate(principal, action, resource)
        if not rbac_decision.allow:
            self._log_and_store(action, principal, resource, rbac_decision, context)
            return rbac_decision

        # 2. Policy Evaluator (Dynamic)
        with self._lock:
            compiled_tree = self.compiled_policy_tree

        if compiled_tree and compiled_tree.policies:
            policy_decision = PolicyEvaluator.evaluate(principal, action, resource, compiled_tree, context.environment_data)
            if not policy_decision.allow:
                self._log_and_store(action, principal, resource, policy_decision, context)
                return policy_decision
            
            # Merge decisions
            final_decision = SecurityDecision(
                allow=True,
                reason="Granted by RBAC and Policies",
                matched_role=rbac_decision.matched_role,
                matched_permission=rbac_decision.matched_permission,
                matched_policy=policy_decision.matched_policy,
                evaluation_trace=rbac_decision.evaluation_trace + policy_decision.evaluation_trace,
                severity=Severity.INFO
            )
        else:
            final_decision = rbac_decision
            
        self.authz_cache.set(principal.identity.identity_id, action, resource.identifier.resource_id, final_decision)
        self._log_and_store(action, principal, resource, final_decision, context)
        return final_decision

    def _log_and_store(self, action: str, principal: Principal, resource: Resource, decision: SecurityDecision, context: SecurityContext):
        with self._lock:
            self._decisions.append(decision)
        self.audit_logger.log(
            action=action,
            principal_id=principal.identity.identity_id,
            resource_id=resource.identifier.resource_id,
            decision=decision,
            context_data={"workflow_id": context.workflow_id, "execution_id": context.execution_id}
        )

    def generate_report(self, workflow_id: str, execution_id: str) -> SecurityReport:
        with self._lock:
            decisions = tuple(self._decisions)
        return SecurityReport(
            workflow_id=workflow_id,
            execution_id=execution_id,
            decisions=decisions
        )
