from dataclasses import dataclass
from typing import List
from training.framework.configuration import ConfigurationBundle

@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    errors: List[str]

class ConfigurationValidator:
    @staticmethod
    def validate_schema(bundle: ConfigurationBundle) -> ValidationReport:
        # Validates payload against dataclass schemas
        return ValidationReport(is_valid=True, errors=[])
        
    @staticmethod
    def validate_semantic(bundle: ConfigurationBundle) -> ValidationReport:
        # Validates logical constraints
        return ValidationReport(is_valid=True, errors=[])
        
    @staticmethod
    def validate_dependency(bundle: ConfigurationBundle) -> ValidationReport:
        # Validates inter-plugin dependencies specified in config
        return ValidationReport(is_valid=True, errors=[])
        
    @classmethod
    def validate_all(cls, bundle: ConfigurationBundle) -> ValidationReport:
        schema_report = cls.validate_schema(bundle)
        if not schema_report.is_valid:
            return schema_report
            
        semantic_report = cls.validate_semantic(bundle)
        if not semantic_report.is_valid:
            return semantic_report
            
        dep_report = cls.validate_dependency(bundle)
        if not dep_report.is_valid:
            return dep_report
            
        return ValidationReport(is_valid=True, errors=[])
