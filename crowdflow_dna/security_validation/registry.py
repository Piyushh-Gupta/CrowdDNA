import threading
from typing import Callable, Tuple
from .exceptions import RegistryConfigurationError

class SecurityRegistry:
    """Thread-safe registry for security validators with dependency ordering."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._validators = {}

    def register(self, name: str, validator: Callable, priority: int = 100, depends_on: Tuple[str, ...] = ()):
        with self._lock:
            self._validators[name] = {
                "validator": validator,
                "priority": priority,
                "depends_on": depends_on,
                "enabled": True
            }

    def disable(self, name: str):
        with self._lock:
            if name in self._validators:
                self._validators[name]["enabled"] = False

    def enable(self, name: str):
        with self._lock:
            if name in self._validators:
                self._validators[name]["enabled"] = True

    def get_ordered_validators(self) -> Tuple[Callable, ...]:
        with self._lock:
            enabled = {k: v for k, v in self._validators.items() if v["enabled"]}
            
            # Sort primarily by priority, then resolve dependencies (topological sort)
            # A simple implementation: order by priority first
            sorted_keys = sorted(enabled.keys(), key=lambda k: enabled[k]["priority"])
            
            # Topological sort
            result = []
            visited = set()
            temp_mark = set()
            
            def visit(node: str):
                if node in temp_mark:
                    raise RegistryConfigurationError(f"Circular dependency detected involving {node}")
                if node not in visited:
                    temp_mark.add(node)
                    for dep in enabled[node]["depends_on"]:
                        if dep in enabled:
                            visit(dep)
                    temp_mark.remove(node)
                    visited.add(node)
                    result.append(node)
                    
            for key in sorted_keys:
                if key not in visited:
                    visit(key)
                    
            return tuple(enabled[k]["validator"] for k in result)
