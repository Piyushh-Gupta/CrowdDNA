import hashlib
import hmac

class EncryptionUtils:
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return hmac.compare_digest(EncryptionUtils.hash_password(password), hashed)
