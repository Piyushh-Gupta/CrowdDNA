from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class SecretReference:
    secret_id: str
    provider_name: str

class SecretProvider:
    def fetch(self, secret_id: str) -> Optional[str]:
        raise NotImplementedError

class SecretResolver:
    def __init__(self):
        from training.security.registry import SecurityRegistry
        self.registry = SecurityRegistry

    def resolve(self, reference: SecretReference) -> Optional[str]:
        provider = self.registry.get_secret_provider(reference.provider_name)
        return provider.fetch(reference.secret_id)

class EnvSecretProvider(SecretProvider):
    def __init__(self, env: Dict[str, str]):
        self.env = env

    def fetch(self, secret_id: str) -> Optional[str]:
        return self.env.get(secret_id)
