import time
from typing import Optional
from training.orchestration.metadata import NodeResult
from training.orchestration.context import NodeContext
from training.orchestration.nodes.base import BaseNode
from training.orchestration.state import NodeState
from training.orchestration.exceptions import NodeExecutionError, NodeCompensationError

class NodeExecutor:
    @staticmethod
    def run(node: BaseNode, context: NodeContext) -> NodeResult:
        start_time = time.time()
        metrics = {}
        error_msg: Optional[str] = None
        status = NodeState.FAILED
        artifacts_produced = []

        try:
            # 1. Prepare
            node.prepare(context)
            
            # 2. Execute
            result = node.execute(context)
            
            status = result.status
            artifacts_produced = list(result.artifacts_produced)
            metrics = dict(result.metrics)
            if result.error_message:
                error_msg = result.error_message
                
            duration = time.time() - start_time
            metrics["duration_seconds"] = duration
            
            return NodeResult(
                node_id=context.node_id,
                status=status,
                artifacts_produced=artifacts_produced,
                metrics=metrics,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = str(e)
            context.logger.error(f"Node {context.node_id} failed: {e}")
            
            # 3. Compensate
            wrapped_error = NodeExecutionError(str(e))
            wrapped_error.__cause__ = e
            
            try:
                node.compensate(context, wrapped_error)
            except Exception as comp_e:
                context.logger.error(f"Node {context.node_id} compensation failed: {comp_e}")
                comp_error = NodeCompensationError(str(comp_e))
                raise comp_error from comp_e
                
            raise wrapped_error from e
