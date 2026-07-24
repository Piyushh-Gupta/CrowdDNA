from typing import Dict, List, Set, Tuple
from training.orchestration.metadata import WorkflowDefinition, ExecutionPlan, ExecutionCursor
from training.orchestration.exceptions import WorkflowValidationError

class WorkflowPlanner:
    @staticmethod
    def plan(definition: WorkflowDefinition) -> Tuple[ExecutionPlan, ExecutionCursor]:
        # Build dependency graph
        deps: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}
        
        for node_id, meta in definition.nodes.items():
            if node_id not in in_degree:
                in_degree[node_id] = 0
            if node_id not in deps:
                deps[node_id] = []
                
            for dep_id in meta.dependencies:
                if dep_id not in deps:
                    deps[dep_id] = []
                if dep_id not in in_degree:
                    in_degree[dep_id] = 0
                    
                deps[dep_id].append(node_id)
                in_degree[node_id] += 1
                
        # Generate topological tiers
        tiers: List[List[str]] = []
        queue: List[str] = sorted([n for n, deg in in_degree.items() if deg == 0])
        
        while queue:
            current_tier = list(queue)
            tiers.append(current_tier)
            next_queue: Set[str] = set()
            
            for node in current_tier:
                for child in deps[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.add(child)
                        
            queue = sorted(list(next_queue))
            
        if sum(len(t) for t in tiers) != len(definition.nodes):
            raise WorkflowValidationError("Graph contains cycles or disconnected components not handled properly.")
            
        # Reconstruct node dependencies mapping (node -> [parents])
        node_parents = {node_id: list(meta.dependencies) for node_id, meta in definition.nodes.items()}
            
        plan = ExecutionPlan(
            tiers=tiers,
            node_dependencies=node_parents
        )
        
        cursor = ExecutionCursor()
        return plan, cursor
