import pytest
from crowdflow_dna.operations.engine import OperationsEngine, InvalidStateTransitionError
from crowdflow_dna.operations.registry import OperationsRegistry
from crowdflow_dna.operations.context import OperationsState
from crowdflow_dna.operations.maintenance.manager import MaintenanceManager
from crowdflow_dna.operations.alerts.throttling import AlertThrottler
from crowdflow_dna.operations.alerts.manager import AlertManager
from crowdflow_dna.operations.incidents.manager import IncidentManager
from crowdflow_dna.operations.administration.manager import AdministrationManager
from crowdflow_dna.operations.administration.permissions import AdminPermissions
from crowdflow_dna.operations.reports.generator import ReportGenerator
from crowdflow_dna.operations.reports.exporters import JSONExporter, MarkdownExporter

def test_state_transitions():
    engine = OperationsEngine(OperationsRegistry())
    assert engine.state == OperationsState.NORMAL
    
    # Idempotency
    engine.transition_state(OperationsState.NORMAL)
    
    engine.transition_state(OperationsState.DEGRADED)
    assert engine.state == OperationsState.DEGRADED
    
    with pytest.raises(InvalidStateTransitionError):
        engine.transition_state(OperationsState.MAINTENANCE)

def test_maintenance_mode():
    engine = OperationsEngine(OperationsRegistry())
    mgr = MaintenanceManager(engine)
    mgr.request_maintenance()
    assert engine.state == OperationsState.MAINTENANCE_REQUESTED
    mgr.start_drain()
    assert engine.state == OperationsState.DRAINING
    mgr.enter_maintenance()
    assert engine.state == OperationsState.MAINTENANCE
    mgr.recover()
    assert engine.state == OperationsState.NORMAL

def test_alert_throttling_and_dedup():
    throttler = AlertThrottler()
    mgr = AlertManager(throttler, None)
    
    # First alert triggers successfully
    assert mgr.trigger_alert("test_alert") is True
    # Second identical alert is deduped immediately (returns False)
    assert mgr.trigger_alert("test_alert") is False

def test_incident_lifecycle():
    mgr = IncidentManager()
    assert mgr.create_incident({"issue": "test"}) == "INC-0001"
    assert mgr.create_incident({"issue": "test2"}) == "INC-0002"

def test_administrator_override():
    perms = AdminPermissions()
    mgr = AdministrationManager(perms)
    
    assert mgr.execute_override("admin", "override_health") is True
    with pytest.raises(PermissionError):
        mgr.execute_override("user", "override_health")

def test_report_generation():
    exporters = {"json": JSONExporter(), "md": MarkdownExporter()}
    gen = ReportGenerator(exporters)
    
    # Test JSON determinism
    res = gen.generate({"b": 2, "a": 1}, "json")
    assert res == '{"a": 1, "b": 2}'
    
def test_registry_behavior():
    reg = OperationsRegistry()
    def mock_health(): pass
    reg.register_health_provider("mock", mock_health)
    assert "mock" in reg.health_providers
