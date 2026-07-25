from typing import Any, Callable

class OperationsRegistry:
    def __init__(self):
        self.health_providers = {}
        self.dependency_providers = {}
        self.alert_rules = {}
        self.report_exporters = {}
        self.maintenance_handlers = {}

    def register_health_provider(self, name: str, provider: Callable):
        self.health_providers[name] = provider
        
    def register_dependency_provider(self, name: str, provider: Callable):
        self.dependency_providers[name] = provider

    def register_alert_rule(self, name: str, rule: Any):
        self.alert_rules[name] = rule
        
    def register_report_exporter(self, name: str, exporter: Any):
        self.report_exporters[name] = exporter
        
    def register_maintenance_handler(self, name: str, handler: Any):
        self.maintenance_handlers[name] = handler
