import hashlib
from .models import BackupManifest

class DeploymentBackup:
    def create_backup(self, metadata: dict) -> BackupManifest:
        return BackupManifest(**metadata)
        
    def verify_backup(self, manifest: BackupManifest, file_path: str) -> bool:
        if not manifest.verified:
            raise ValueError("Backup manifest is not verified")
            
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != manifest.sha256:
                raise ValueError("Backup file checksum does not match manifest")
        except FileNotFoundError:
            raise ValueError("Backup file not found")
            
        return True