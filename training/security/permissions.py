from typing import Dict
from training.security.metadata import Permission, Capability

class PermissionRegistry:
    _permissions: Dict[str, Permission] = {}
    _capabilities: Dict[str, Capability] = {}

    @classmethod
    def register_permission(cls, perm: Permission):
        cls._permissions[perm.name] = perm

    @classmethod
    def register_capability(cls, cap: Capability):
        cls._capabilities[cap.name] = cap

    @classmethod
    def get_permission(cls, name: str) -> Permission:
        return cls._permissions[name]

    @classmethod
    def get_capability(cls, name: str) -> Capability:
        return cls._capabilities[name]

    @classmethod
    def clear(cls):
        cls._permissions.clear()
        cls._capabilities.clear()
