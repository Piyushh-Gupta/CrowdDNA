import hashlib

def generate_cache_key(*args: str) -> str:
    """Generates a deterministic cache key from immutable inputs."""
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(arg.encode("utf-8"))
    return hasher.hexdigest()
