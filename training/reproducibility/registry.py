from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, Callable, Any

class Severity(Enum):
    CRITICAL = 4
    MAJOR = 3
    MINOR = 2
    INFO = 1

@dataclass(frozen=True)
class ValidatorMetadata:
    name: str
    version: str
    severity: Severity

class ValidatorRegistry:
    """Registry for managing reproducibility validation rules."""
    
    _registry: Dict[str, Type[Any]] = {}
    _metadata: Dict[str, ValidatorMetadata] = {}

    @classmethod
    def register(cls, metadata: ValidatorMetadata) -> Callable:
        """Decorator to register a validation rule."""
        def wrapper(validator_cls: Type[Any]) -> Type[Any]:
            if metadata.name in cls._registry:
                raise ValueError(f"Validator '{metadata.name}' is already registered.")
            cls._registry[metadata.name] = validator_cls
            cls._metadata[metadata.name] = metadata
            return validator_cls
        return wrapper

    @classmethod
    def get_validator(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise ValueError(f"Validator '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def get_metadata(cls, name: str) -> ValidatorMetadata:
        if name not in cls._metadata:
            raise ValueError(f"Metadata for validator '{name}' not found.")
        return cls._metadata[name]

    @classmethod
    def list_validators(cls) -> Dict[str, ValidatorMetadata]:
        return dict(cls._metadata)

# Define some built-in validators
from training.reproducibility.metadata import ExperimentManifest, EnvironmentSnapshot

@ValidatorRegistry.register(ValidatorMetadata(name="git_clean", version="1.0", severity=Severity.MAJOR))
class GitCleanValidator:
    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> bool:
        return not current_env.git_is_dirty

@ValidatorRegistry.register(ValidatorMetadata(name="git_commit_match", version="1.0", severity=Severity.CRITICAL))
class GitCommitValidator:
    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> bool:
        return manifest.environment.git_commit == current_env.git_commit

@ValidatorRegistry.register(ValidatorMetadata(name="python_version", version="1.0", severity=Severity.INFO))
class PythonVersionValidator:
    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> bool:
        return manifest.environment.python_version == current_env.python_version

@ValidatorRegistry.register(ValidatorMetadata(name="cuda_version", version="1.0", severity=Severity.MINOR))
class CudaVersionValidator:
    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> bool:
        return manifest.environment.cuda_version == current_env.cuda_version
