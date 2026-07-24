import os
import hashlib

def _hash_file_strict(filepath: str) -> str:
    """Computes SHA256 of the entire file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _hash_file_fast(filepath: str) -> str:
    """Computes a fast hash based on file size, modification time, and name."""
    stat = os.stat(filepath)
    hasher = hashlib.sha256()
    # Adding filename to hash ensures different files with same size/time hash differently
    fast_str = f"{os.path.basename(filepath)}_{stat.st_size}_{stat.st_mtime}"
    hasher.update(fast_str.encode('utf-8'))
    return hasher.hexdigest()

def compute_dataset_fingerprint(dataset_path: str, strict: bool = False) -> str:
    """
    Computes a deterministic dataset fingerprint.
    Fast mode: hashes structure, names, and sizes.
    Strict mode: hashes entire file contents.
    """
    if not os.path.exists(dataset_path):
        return "invalid_path"
        
    file_hashes = {}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            filepath = os.path.join(root, file)
            # Normalize paths to use forward slashes for cross-platform determinism
            relpath = os.path.relpath(filepath, dataset_path).replace("\\", "/")
            
            if strict:
                file_hash = _hash_file_strict(filepath)
            else:
                file_hash = _hash_file_fast(filepath)
                
            file_hashes[relpath] = file_hash

    # Sort to guarantee determinism
    sorted_hashes = sorted(file_hashes.items())
    
    # Hash the aggregate
    aggregate_hasher = hashlib.sha256()
    for relpath, f_hash in sorted_hashes:
        aggregate_hasher.update(f"{relpath}:{f_hash}".encode('utf-8'))
        
    return aggregate_hasher.hexdigest()
