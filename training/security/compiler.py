from typing import List
from training.security.metadata import Policy

class CompiledPolicyTree:
    def __init__(self, policies: List[Policy]):
        self.policies = policies

class PolicyCompiler:
    @staticmethod
    def compile(policies: List[Policy]) -> CompiledPolicyTree:
        # Sort policies to ensure DENY takes precedence
        sorted_policies = sorted(policies, key=lambda p: 0 if p.effect == 'deny' else 1)
        return CompiledPolicyTree(sorted_policies)
