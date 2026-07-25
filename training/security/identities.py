from typing import Dict
from training.security.metadata import Principal, Identity

class IdentityManager:
    @staticmethod
    def resolve_principal(identity: Identity, role_mapping: Dict[str, tuple[str, ...]]) -> Principal:
        roles = role_mapping.get(identity.identity_id, tuple())
        return Principal(identity=identity, roles=roles)
