from typing import List, Dict, Set
from training.framework.metadata import PluginMetadata

class DependencyCycleError(Exception):
    pass

class DependencyGraph:
    def __init__(self):
        self._nodes: Dict[str, PluginMetadata] = {}
        
    def add_plugin(self, metadata: PluginMetadata) -> None:
        self._nodes[metadata.name] = metadata
        
    def resolve_topological_order(self) -> List[PluginMetadata]:
        # Perform topological sort with cycle detection
        # Deterministic sorting: if multiple plugins can be initialized, sort alphabetically
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        order: List[PluginMetadata] = []
        
        # Sort nodes alphabetically to ensure deterministic priority resolution
        node_names = sorted(list(self._nodes.keys()))
        
        def visit(name: str):
            if name in temp_mark:
                raise DependencyCycleError(f"Dependency cycle detected involving '{name}'")
            if name not in visited:
                if name not in self._nodes:
                    # Depending on strictness, missing required dependencies should fail before this.
                    # Assuming we only visit registered nodes.
                    return
                temp_mark.add(name)
                
                meta = self._nodes[name]
                deps = sorted(list(meta.dependencies.keys()))
                for dep in deps:
                    constraint = meta.dependencies[dep]
                    if dep not in self._nodes:
                        raise ValueError(f"Missing required dependency '{dep}' for '{name}'")
                    dep_meta = self._nodes[dep]
                    if constraint.exact_version and dep_meta.version != constraint.exact_version:
                        raise ValueError(f"Version mismatch for '{dep}'. Expected {constraint.exact_version}, got {dep_meta.version}")
                    visit(dep)
                    
                temp_mark.remove(name)
                visited.add(name)
                order.append(meta)
                
        for name in node_names:
            visit(name)
            
        return order
