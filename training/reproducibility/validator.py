from typing import Dict, List, Any, Tuple
from training.reproducibility.metadata import ExperimentManifest, EnvironmentSnapshot
from training.reproducibility.registry import ValidatorRegistry, Severity

class ValidatorProtocol:
    """Protocol for a validation rule."""
    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> bool:
        pass

class ReproducibilityValidator:
    """Executes registered validators and produces a reproducibility score."""
    
    def __init__(self):
        self.max_score = 100.0
        # Deduction weights based on severity
        self.weights = {
            Severity.CRITICAL: 100.0,
            Severity.MAJOR: 20.0,
            Severity.MINOR: 5.0,
            Severity.INFO: 0.0
        }

    def validate(self, manifest: ExperimentManifest, current_env: EnvironmentSnapshot) -> Tuple[float, List[Dict[str, Any]]]:
        """Validates the manifest against the current environment."""
        diagnostics = []
        score = self.max_score
        
        for name, metadata in ValidatorRegistry.list_validators().items():
            validator_cls = ValidatorRegistry.get_validator(name)
            validator = validator_cls()
            
            passed = validator.validate(manifest, current_env)
            if not passed:
                deduction = self.weights[metadata.severity]
                score -= deduction
                diagnostics.append({
                    "validator": name,
                    "severity": metadata.severity.name,
                    "status": "FAILED",
                    "deduction": deduction
                })
            else:
                diagnostics.append({
                    "validator": name,
                    "severity": metadata.severity.name,
                    "status": "PASSED",
                    "deduction": 0.0
                })
                
        # Cap score at 0
        final_score = max(0.0, score)
        return final_score, diagnostics
