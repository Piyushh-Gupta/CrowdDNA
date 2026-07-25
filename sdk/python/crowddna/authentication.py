from abc import ABC, abstractmethod
from typing import Dict

class AuthenticationProvider(ABC):
    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        pass

class APIKeyProvider(AuthenticationProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_auth_headers(self):
        return {"X-API-Key": self.api_key}
        
class JWTProvider(AuthenticationProvider):
    def __init__(self, token: str):
        self.token = token
        
    def get_auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"}
