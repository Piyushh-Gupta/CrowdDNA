from typing import Optional
from training.security.metadata import Identity

class AuthProvider:
    def authenticate(self, credentials: dict) -> Optional[Identity]:
        raise NotImplementedError

class LocalAuthProvider(AuthProvider):
    def __init__(self, users: dict):
        self._users = users # dict of username: Identity

    def authenticate(self, credentials: dict) -> Optional[Identity]:
        username = credentials.get("username")
        password = credentials.get("password")
        # In a real scenario we check password hash. Here we just mock it.
        if username in self._users and password == "secret":
            return self._users[username]
        return None
