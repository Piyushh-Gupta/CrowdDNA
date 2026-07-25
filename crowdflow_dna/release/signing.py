import hashlib
import os
from .exceptions import ChecksumError

class SigningManager:
    def __init__(self, chunk_size=8192):
        self.chunk_size = chunk_size
        
    def generate_checksums(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise ChecksumError(f"File not found: {filepath}")
            
        sha256 = hashlib.sha256()
        sha512 = hashlib.sha512()
        
        try:
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    sha256.update(chunk)
                    sha512.update(chunk)
        except IOError as e:
            raise ChecksumError(f"Failed to read file: {e}")
            
        return {
            "sha256": sha256.hexdigest(),
            "sha512": sha512.hexdigest()
        }
