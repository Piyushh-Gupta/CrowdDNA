import argparse
import sys
import logging
from training.orchestration.engine import WorkflowEngine
from training.orchestration.reporting.exporters import ReportExporter
# For CLI we'll assume a dummy manifest factory for now since it's just scaffolding.
from training.orchestration.pipeline import PipelineIntegrator
from training.orchestration.validation import WorkflowValidator
from training.orchestration.planner import WorkflowPlanner
from training.orchestration.metadata import NodeMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WorkflowCLI")

def _get_dummy_manifest():
    return PipelineIntegrator.create_manifest(
        workflow_name="cli_dummy_workflow",
        execution_id="exec-001",
        session_id="sess-001",
        artifact_root="./artifacts/cli_test",
        nodes={
            "node_a": NodeMetadata(node_id="node_a", node_type="DummyNode"),
            "node_b": NodeMetadata(node_id="node_b", node_type="DummyNode", dependencies=["node_a"])
        }
    )

def cmd_validate():
    manifest = _get_dummy_manifest()
    try:
        WorkflowValidator.validate(manifest.definition)
        logger.info("Validation successful.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

def cmd_plan():
    manifest = _get_dummy_manifest()
    try:
        WorkflowValidator.validate(manifest.definition)
        plan, _ = WorkflowPlanner.plan(manifest.definition)
        logger.info(f"Plan successful. Tiers: {plan.tiers}")
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        sys.exit(1)

def cmd_run(dry_run: bool):
    manifest = _get_dummy_manifest()
    if dry_run:
        logger.info("Dry-run requested. Doing plan only.")
        cmd_plan()
        return
        
    engine = WorkflowEngine(logger)
    report = engine.run(manifest)
    
    import os
    os.makedirs(manifest.execution.artifact_tree_root, exist_ok=True)
    report_path = os.path.join(manifest.execution.artifact_tree_root, "report.json")
    ReportExporter.export_json(report, report_path)
    logger.info(f"Execution complete. Status: {report.status}. Report saved to {report_path}")

def cmd_resume(resume_dir: str):
    manifest = _get_dummy_manifest()
    engine = WorkflowEngine(logger)
    report = engine.run(manifest, resume_dir=resume_dir)
    logger.info(f"Resume complete. Status: {report.status}.")

def cmd_status():
    logger.info("Status command not fully implemented for external queries yet.")

def main():
    parser = argparse.ArgumentParser(description="Workflow Orchestration CLI")
    parser.add_argument("action", choices=["run", "validate", "plan", "resume", "status"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-dir", type=str, help="Path to checkpoint directory for resume")
    args = parser.parse_args()

    if args.action == "validate":
        cmd_validate()
    elif args.action == "plan":
        cmd_plan()
    elif args.action == "run":
        cmd_run(args.dry_run)
    elif args.action == "resume":
        if not args.resume_dir:
            logger.error("--resume-dir is required for resume action")
            sys.exit(1)
        cmd_resume(args.resume_dir)
    elif args.action == "status":
        cmd_status()

if __name__ == "__main__":
    main()
