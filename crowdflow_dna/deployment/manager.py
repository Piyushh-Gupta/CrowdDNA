import os

try:
    import fcntl
except ImportError:
    class MockFcntl:
        LOCK_EX = 1
        LOCK_NB = 2
        LOCK_UN = 8
        _locked_files = set()
        def flock(self, fd, op):
            name = fd.name if hasattr(fd, 'name') else str(fd)
            if op & self.LOCK_UN:
                if name in self._locked_files:
                    self._locked_files.remove(name)
            elif op & self.LOCK_EX:
                if name in self._locked_files:
                    raise BlockingIOError("Resource temporarily unavailable")
                self._locked_files.add(name)
    fcntl = MockFcntl()

from .state_machine import DeploymentLifecycle, DeploymentState
from .validator import DeploymentValidator
from .health import DeploymentHealth
from .models import ReleaseManifest

class DeploymentManager:
    def __init__(self, lock_file: str = "deployment.lock"):
        self.lifecycle = DeploymentLifecycle()
        self.validator = DeploymentValidator()
        self.health = DeploymentHealth()
        self.lock_file = lock_file
        self.lock_fd = None

    def acquire_lock(self):
        self.lock_fd = open(self.lock_file, 'w')
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_fd.close()
            raise RuntimeError("Another deployment is currently running (lock acquired)")

    def release_lock(self):
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                self.lock_fd.close()
                if os.path.exists(self.lock_file):
                    try:
                        os.remove(self.lock_file)
                    except OSError:
                        pass

    def execute_deployment(self, manifest: ReleaseManifest):
        self.acquire_lock()
        try:
            self.lifecycle.transition(DeploymentState.VALIDATING)
            self.validator.validate_infrastructure()
            self.validator.validate_docker()
            self.validator.validate_compose()
            self.validator.validate_environment()
            self.validator.validate_manifest()
            self.validator.validate_release(manifest)
            
            self.lifecycle.transition(DeploymentState.BUILDING)
            self.lifecycle.transition(DeploymentState.DEPLOYING)
            
            self.lifecycle.transition(DeploymentState.VERIFYING)
            if not self.health.verify_startup():
                self.lifecycle.transition(DeploymentState.VERIFY_FAILED)
                return False
            if not self.health.verify_readiness():
                self.lifecycle.transition(DeploymentState.VERIFY_FAILED)
                return False
            if not self.health.verify_liveness():
                self.lifecycle.transition(DeploymentState.VERIFY_FAILED)
                return False
            if not self.health.verify_synthetic_service():
                self.lifecycle.transition(DeploymentState.VERIFY_FAILED)
                return False
                
            self.lifecycle.transition(DeploymentState.ACTIVE)
            return True
        except Exception as e:
            if self.lifecycle.state in [DeploymentState.DEPLOYING, DeploymentState.VERIFYING]:
                self.lifecycle.transition(
                    DeploymentState.DEPLOY_FAILED if self.lifecycle.state == DeploymentState.DEPLOYING else DeploymentState.VERIFY_FAILED
                )
            elif self.lifecycle.state != DeploymentState.FAILED:
                self.lifecycle.transition(DeploymentState.FAILED)
            raise e
        finally:
            self.release_lock()