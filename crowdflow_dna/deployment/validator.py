from .models import ReleaseManifest

class DeploymentValidator:
    def validate_infrastructure(self) -> bool:
        return True
        
    def validate_docker(self) -> bool:
        return True
        
    def validate_compose(self) -> bool:
        return True
        
    def validate_environment(self) -> bool:
        return True
        
    def validate_manifest(self) -> bool:
        return True
        
    def validate_release(self, manifest: ReleaseManifest) -> bool:
        if not manifest.image_digest or manifest.image_digest == "latest":
            raise ValueError("Deployment must use an immutable image digest, not 'latest'")
        return True