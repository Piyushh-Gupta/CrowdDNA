from typing import List
from training.orchestration.metadata import ExecutionPlan, ExecutionCursor, NodeResult
from training.orchestration.state import NodeState
from training.orchestration.policies import FailureIsolationLevel

class WorkflowScheduler:
    @staticmethod
    def get_ready_nodes(plan: ExecutionPlan, cursor: ExecutionCursor) -> List[str]:
        """Returns the next batch of ready nodes based on topological tiers and cursor state."""
        ready = []
        for tier in plan.tiers:
            tier_complete = True
            for node in tier:
                state = cursor.get_state(node)
                
                if state in (NodeState.SUCCESS, NodeState.SKIPPED):
                    continue
                    
                tier_complete = False
                
                if state == NodeState.PENDING:
                    # Check if parents are satisfied
                    parents = plan.node_dependencies.get(node, [])
                    parents_ready = True
                    for p in parents:
                        p_state = cursor.get_state(p)
                        if p_state != NodeState.SUCCESS:
                            parents_ready = False
                            # If a parent failed/skipped and we are in cascading mode, we might skip this node.
                            # We handle cascading skip in handle_result, so by the time we check here, 
                            # if it's supposed to be skipped, its state would be SKIPPED.
                            break
                    if parents_ready:
                        ready.append(node)
                        
            # If we found ready nodes in this tier, or the tier is incomplete, we don't proceed to next tier.
            if not tier_complete:
                break
                
        return ready

    @staticmethod
    def transition(cursor: ExecutionCursor, node_id: str, to_state: NodeState) -> None:
        current = cursor.get_state(node_id)
        # Simplified validation mapping
        valid_transitions = {
            NodeState.PENDING: {NodeState.PREPARING, NodeState.SKIPPED, NodeState.CANCELLED, NodeState.RUNNING},
            NodeState.PREPARING: {NodeState.RUNNING, NodeState.FAILED},
            NodeState.RUNNING: {NodeState.SUCCESS, NodeState.COMPENSATING, NodeState.FAILED},
            NodeState.COMPENSATING: {NodeState.FAILED},
            NodeState.FAILED: {NodeState.PENDING},  # Retry
            NodeState.SUCCESS: set(),
            NodeState.SKIPPED: set(),
            NodeState.CANCELLED: set()
        }
        
        if to_state not in valid_transitions.get(current, set()):
            # To allow flexibility where PREPARING wasn't emitted because Executor abstracts it,
            # we allow PENDING -> RUNNING.
            pass # We rely on Executor logic, but could strict enforce.
            
        cursor.update_state(node_id, to_state)

    @staticmethod
    def handle_result(plan: ExecutionPlan, cursor: ExecutionCursor, result: NodeResult, isolation_level: FailureIsolationLevel, max_retries: int) -> None:
        """Processes a node result, updates the cursor, applies retry/failure policies."""
        if result.status == NodeState.SUCCESS:
            WorkflowScheduler.transition(cursor, result.node_id, NodeState.SUCCESS)
        else:
            # Handle failure
            retries = cursor.retry_counts.get(result.node_id, 0)
            if retries < max_retries:
                cursor.retry_counts[result.node_id] = retries + 1
                WorkflowScheduler.transition(cursor, result.node_id, NodeState.PENDING) # Requeue
            else:
                WorkflowScheduler.transition(cursor, result.node_id, NodeState.FAILED)
                
                if isolation_level == FailureIsolationLevel.CASCADING:
                    WorkflowScheduler._cascade_skip(plan, cursor, result.node_id)

    @staticmethod
    def _cascade_skip(plan: ExecutionPlan, cursor: ExecutionCursor, failed_node: str):
        # BFS to skip all descendants
        from collections import deque
        
        # Build children map
        children_map = {n: [] for n in plan.node_dependencies.keys()}
        for child, parents in plan.node_dependencies.items():
            for p in parents:
                if p in children_map:
                    children_map[p].append(child)
                    
        queue = deque(children_map.get(failed_node, []))
        while queue:
            curr = queue.popleft()
            if cursor.get_state(curr) == NodeState.PENDING:
                WorkflowScheduler.transition(cursor, curr, NodeState.SKIPPED)
                queue.extend(children_map.get(curr, []))
