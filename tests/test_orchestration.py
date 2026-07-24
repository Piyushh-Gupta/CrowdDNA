from training.orchestration.dispatcher import EventDispatcher
from training.orchestration.executor import NodeExecutor
from training.orchestration.exceptions import NodeExecutionError
import pytest
import os
import logging
from training.orchestration.metadata import (
    WorkflowDefinition, WorkflowExecution, WorkflowManifest, NodeMetadata
)
from training.framework.metadata import FrameworkMetadata
from training.orchestration.validation import WorkflowValidator
from training.orchestration.exceptions import WorkflowCycleError
from training.orchestration.planner import WorkflowPlanner
from training.orchestration.registry import NodeRegistry
from training.orchestration.nodes.base import BaseNode
from training.orchestration.context import NodeContext
from training.orchestration.metadata import NodeResult
from training.orchestration.state import NodeState
from training.orchestration.engine import WorkflowEngine

# Dummy node for tests
class DummyNode(BaseNode):
    category = "Utility"
    def execute(self, context: NodeContext) -> NodeResult:
        return NodeResult(node_id=context.node_id, status=NodeState.SUCCESS)

class FailingNode(BaseNode):
    category = "Utility"
    def execute(self, context: NodeContext) -> NodeResult:
        return NodeResult(node_id=context.node_id, status=NodeState.FAILED, error_message="Intended failure")

class ExceptionNode(BaseNode):
    category = "Utility"
    def execute(self, context: NodeContext) -> NodeResult:
        raise ValueError("Boom")

@pytest.fixture(autouse=True)
def setup_registry():
    NodeRegistry.clear()
    NodeRegistry.register("DummyNode", DummyNode)
    NodeRegistry.register("FailingNode", FailingNode)
    NodeRegistry.register("ExceptionNode", ExceptionNode)
    yield
    NodeRegistry.clear()

def test_validation_success():
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="DummyNode"),
        "B": NodeMetadata(node_id="B", node_type="DummyNode", dependencies=["A"])
    }
    definition = WorkflowDefinition(
        workflow_name="test",
        workflow_schema_version="1.0",
        nodes=def_nodes
    )
    WorkflowValidator.validate(definition)

def test_validation_cycle():
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="DummyNode", dependencies=["B"]),
        "B": NodeMetadata(node_id="B", node_type="DummyNode", dependencies=["A"])
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    with pytest.raises(WorkflowCycleError):
        WorkflowValidator.validate(definition)

def test_planner_topological_sort():
    def_nodes = {
        "C": NodeMetadata(node_id="C", node_type="DummyNode", dependencies=["B"]),
        "A": NodeMetadata(node_id="A", node_type="DummyNode"),
        "B": NodeMetadata(node_id="B", node_type="DummyNode", dependencies=["A"])
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    plan, cursor = WorkflowPlanner.plan(definition)
    
    # Expected tiers: [A], [B], [C]
    assert plan.tiers == (("A",), ("B",), ("C",))

def test_engine_successful_execution(tmp_path):
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="DummyNode"),
        "B": NodeMetadata(node_id="B", node_type="DummyNode", dependencies=["A"])
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    execution = WorkflowExecution("exec-1", "sess-1", str(tmp_path), FrameworkMetadata(framework_version="1.0", crowddna_version="prod"))
    manifest = WorkflowManifest(definition, execution)
    
    logger = logging.getLogger("test")
    engine = WorkflowEngine(logger)
    report = engine.run(manifest)
    
    assert report.status == "SUCCESS"
    assert len(report.node_reports) == 2
    for nr in report.node_reports:
        assert nr.status == "SUCCESS"

def test_engine_failure_handling(tmp_path):
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="FailingNode"),
        "B": NodeMetadata(node_id="B", node_type="DummyNode", dependencies=["A"])
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    execution = WorkflowExecution("exec-1", "sess-1", str(tmp_path), FrameworkMetadata(framework_version="1.0", crowddna_version="prod"))
    manifest = WorkflowManifest(definition, execution)
    
    logger = logging.getLogger("test")
    engine = WorkflowEngine(logger)
    report = engine.run(manifest)
    
    assert report.status == "FAILED"
    assert any(nr.node_id == "A" and nr.status == "FAILED" for nr in report.node_reports)
    # Since cascading failure default? Wait, FailureIsolationLevel.CRITICAL aborts.
    assert any(nr.node_id == "B" and nr.status == "PENDING" for nr in report.node_reports)

def test_resume(tmp_path):
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="DummyNode")
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    execution = WorkflowExecution("exec-1", "sess-1", str(tmp_path), FrameworkMetadata(framework_version="1.0", crowddna_version="prod"))
    manifest = WorkflowManifest(definition, execution)
    
    logger = logging.getLogger("test")
    engine = WorkflowEngine(logger)
    report = engine.run(manifest)
    
    assert report.status == "SUCCESS"
    
    # Resume
    report_resume = engine.run(manifest, resume_dir=os.path.join(str(tmp_path), "checkpoints"))
    assert report_resume.status == "SUCCESS"


def test_executor_exception_wrapping():
    context = NodeContext('wf-1', 'ex-1', 'A', 'corr-1', './artifacts', None, logging.getLogger('test'))
    node = ExceptionNode()
    try:
        NodeExecutor.run(node, context)
        assert False, "Expected NodeExecutionError"
    except NodeExecutionError as e:
        assert isinstance(e.__cause__, ValueError)
        assert str(e.__cause__) == "Boom"

def test_event_emission(tmp_path):
    events = []
    def listener(event):
        events.append(event)
    
    EventDispatcher.subscribe(listener)
    
    def_nodes = {
        "A": NodeMetadata(node_id="A", node_type="DummyNode")
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    execution = WorkflowExecution("exec-1", "sess-1", str(tmp_path), FrameworkMetadata(framework_version="1.0", crowddna_version="prod"))
    manifest = WorkflowManifest(definition, execution)
    
    engine = WorkflowEngine(logging.getLogger("test"))
    engine.run(manifest)
    
    assert len(events) == 2
    assert events[0].payload['action'] == 'WORKFLOW_STARTED'
    assert events[1].payload['action'] == 'WORKFLOW_COMPLETED'
    assert events[0].workflow_id == 'test'
    assert events[0].execution_id == 'exec-1'
    assert events[0].correlation_id == 'exec-1'
    
    EventDispatcher.clear()

def test_deterministic_report_ordering(tmp_path):
    def_nodes = {
        "Z": NodeMetadata(node_id="Z", node_type="DummyNode"),
        "M": NodeMetadata(node_id="M", node_type="DummyNode"),
        "A": NodeMetadata(node_id="A", node_type="DummyNode")
    }
    definition = WorkflowDefinition("test", "1.0", def_nodes)
    execution = WorkflowExecution("exec-1", "sess-1", str(tmp_path), FrameworkMetadata(framework_version="1.0", crowddna_version="prod"))
    manifest = WorkflowManifest(definition, execution)
    
    engine = WorkflowEngine(logging.getLogger("test"))
    report = engine.run(manifest)
    
    ordered_ids = [nr.node_id for nr in report.node_reports]
    assert ordered_ids == ["Z", "M", "A"]

