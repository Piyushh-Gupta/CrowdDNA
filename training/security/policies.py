from typing import List
from training.security.metadata import Policy

class PolicyProvider:
    def get_policies(self) -> List[Policy]:
        return []

class InMemoryPolicyProvider(PolicyProvider):
    def __init__(self, policies: List[Policy]):
        self._policies = policies

    def get_policies(self) -> List[Policy]:
        return self._policies
