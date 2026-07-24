import pytest
from training.framework.metadata import PluginMetadata, VersionConstraint, HealthReport
from training.framework.dependency import DependencyGraph, DependencyCycleError
from training.framework.lifecycle import StateMachine, LifecycleState, IllegalStateTransitionError
from training.framework.resolver import ConfigurationMerger

def test_dependency_graph_topological_sort():
    graph = DependencyGraph()
    # A depends on B and C. B depends on C.
    graph.add_plugin(PluginMetadata(name="A", version="1", schema_version="1", api_version="1", category="t", capabilities=frozenset(), entrypoint="A", dependencies={"B": VersionConstraint("1"), "C": VersionConstraint("1")}))
    graph.add_plugin(PluginMetadata(name="B", version="1", schema_version="1", api_version="1", category="t", capabilities=frozenset(), entrypoint="B", dependencies={"C": VersionConstraint("1")}))
    graph.add_plugin(PluginMetadata(name="C", version="1", schema_version="1", api_version="1", category="t", capabilities=frozenset(), entrypoint="C", dependencies={}))
    
    order = graph.resolve_topological_order()
    names = [m.name for m in order]
    assert names == ["C", "B", "A"]

def test_dependency_cycle_detection():
    graph = DependencyGraph()
    graph.add_plugin(PluginMetadata(name="A", version="1", schema_version="1", api_version="1", category="t", capabilities=frozenset(), entrypoint="A", dependencies={"B": VersionConstraint("1")}))
    graph.add_plugin(PluginMetadata(name="B", version="1", schema_version="1", api_version="1", category="t", capabilities=frozenset(), entrypoint="B", dependencies={"A": VersionConstraint("1")}))
    
    with pytest.raises(DependencyCycleError):
        graph.resolve_topological_order()

def test_lifecycle_state_machine():
    StateMachine.validate_transition(LifecycleState.UNINITIALIZED, LifecycleState.INITIALIZED)
    with pytest.raises(IllegalStateTransitionError):
        StateMachine.validate_transition(LifecycleState.UNINITIALIZED, LifecycleState.RUNNING)

def test_configuration_merger():
    merger = ConfigurationMerger()
    merger.add_layer('default', {"a": 1})
    merger.add_layer('cli', {"a": 2})
    bundle = merger.merge()
    assert bundle.get("a") == 2
    assert bundle.values["a"].provenance == "cli"

def test_health_report():
    assert HealthReport.HEALTHY.name == "HEALTHY"
