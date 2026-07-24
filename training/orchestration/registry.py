from typing import Dict, Type
from training.orchestration.nodes.base import BaseNode
from training.orchestration.policies import RetryPolicy, TimeoutPolicy, FailurePolicy, ExecutionPolicy
from training.orchestration.metadata import WorkflowDefinition

class NodeRegistry:
    _nodes: Dict[str, Type[BaseNode]] = {}

    @classmethod
    def register(cls, name: str, node_cls: Type[BaseNode]):
        if name in cls._nodes:
            raise ValueError(f"Node '{name}' is already registered.")
        cls._nodes[name] = node_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseNode]:
        if name not in cls._nodes:
            raise KeyError(f"Node '{name}' not found.")
        return cls._nodes[name]

    @classmethod
    def clear(cls):
        cls._nodes.clear()


class PolicyRegistry:
    _retry_policies: Dict[str, RetryPolicy] = {}
    _timeout_policies: Dict[str, TimeoutPolicy] = {}
    _failure_policies: Dict[str, FailurePolicy] = {}
    _execution_policies: Dict[str, ExecutionPolicy] = {}

    @classmethod
    def register_retry_policy(cls, name: str, policy: RetryPolicy):
        if name in cls._retry_policies:
            raise ValueError(f"RetryPolicy '{name}' is already registered.")
        cls._retry_policies[name] = policy

    @classmethod
    def get_retry_policy(cls, name: str) -> RetryPolicy:
        return cls._retry_policies.get(name, RetryPolicy())

    @classmethod
    def register_timeout_policy(cls, name: str, policy: TimeoutPolicy):
        if name in cls._timeout_policies:
            raise ValueError(f"TimeoutPolicy '{name}' is already registered.")
        cls._timeout_policies[name] = policy

    @classmethod
    def get_timeout_policy(cls, name: str) -> TimeoutPolicy:
        return cls._timeout_policies.get(name, TimeoutPolicy())

    @classmethod
    def register_failure_policy(cls, name: str, policy: FailurePolicy):
        if name in cls._failure_policies:
            raise ValueError(f"FailurePolicy '{name}' is already registered.")
        cls._failure_policies[name] = policy

    @classmethod
    def get_failure_policy(cls, name: str) -> FailurePolicy:
        return cls._failure_policies.get(name, FailurePolicy())
        
    @classmethod
    def register_execution_policy(cls, name: str, policy: ExecutionPolicy):
        if name in cls._execution_policies:
            raise ValueError(f"ExecutionPolicy '{name}' is already registered.")
        cls._execution_policies[name] = policy

    @classmethod
    def get_execution_policy(cls, name: str) -> ExecutionPolicy:
        return cls._execution_policies.get(name, ExecutionPolicy())

    @classmethod
    def clear(cls):
        cls._retry_policies.clear()
        cls._timeout_policies.clear()
        cls._failure_policies.clear()
        cls._execution_policies.clear()


class WorkflowRegistry:
    _workflows: Dict[str, WorkflowDefinition] = {}

    @classmethod
    def register(cls, name: str, workflow: WorkflowDefinition):
        if name in cls._workflows:
            raise ValueError(f"Workflow '{name}' is already registered.")
        cls._workflows[name] = workflow

    @classmethod
    def get(cls, name: str) -> WorkflowDefinition:
        if name not in cls._workflows:
            raise KeyError(f"Workflow '{name}' not found.")
        return cls._workflows[name]

    @classmethod
    def clear(cls):
        cls._workflows.clear()
