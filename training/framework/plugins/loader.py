from typing import List, Dict
from training.framework.metadata import PluginMetadata
from training.framework.plugins.registry import PluginManifestRegistry
from training.framework.plugins.base import BasePlugin
from training.framework.lifecycle import LifecycleState, StateMachine, LifecycleEvent
from training.framework.clock import Clock

class PluginLoader:
    def __init__(self, context_provider):
        self.context_provider = context_provider
        self.active_plugins: Dict[str, BasePlugin] = {}
        self.plugin_states: Dict[str, LifecycleState] = {}
        self.events: List[LifecycleEvent] = []
        
    def _emit(self, name: str, state: LifecycleState, duration: float, error: str = None) -> None:
        self.events.append(LifecycleEvent(
            timestamp=Clock.now(),
            plugin_name=name,
            state=state,
            duration=duration,
            error=error
        ))
        
    def _transition(self, name: str, target_state: LifecycleState, func) -> None:
        current = self.plugin_states.get(name, LifecycleState.UNINITIALIZED)
        StateMachine.validate_transition(current, target_state)
        
        start_t = Clock.perf_counter()
        error = None
        try:
            func()
            self.plugin_states[name] = target_state
        except Exception as e:
            error = str(e)
            raise e
        finally:
            duration = Clock.perf_counter() - start_t
            self._emit(name, target_state, duration, error)
            
    def load_and_initialize(self, order: List[PluginMetadata]) -> None:
        initialized: List[str] = []
        try:
            for meta in order:
                # Constructor injection
                plugin_cls = PluginManifestRegistry.get_class(meta.name)
                ctx = self.context_provider(meta.name)
                plugin = plugin_cls(ctx)
                self.active_plugins[meta.name] = plugin
                
                try:
                    self._transition(meta.name, LifecycleState.INITIALIZED, plugin.initialize)
                    self._transition(meta.name, LifecycleState.CONFIGURED, plugin.configure)
                    self._transition(meta.name, LifecycleState.RUNNING, plugin.start)
                    initialized.append(meta.name)
                except Exception as e:
                    if meta.category == 'exporter':
                        import logging
                        logging.warning(f"Exporter plugin '{meta.name}' failed to start: {e}. Disabling.")
                        self.active_plugins.pop(meta.name)
                    else:
                        raise e
        except Exception as e:
            # Rollback initialized plugins in reverse order
            for name in reversed(initialized):
                plugin = self.active_plugins[name]
                try:
                    # Transition through STOPPED then SHUTDOWN
                    current_state = self.plugin_states.get(name, LifecycleState.UNINITIALIZED)
                    if current_state == LifecycleState.RUNNING:
                        self._transition(name, LifecycleState.STOPPING, plugin.stop)
                        self.plugin_states[name] = LifecycleState.STOPPED
                    if self.plugin_states.get(name) in (LifecycleState.STOPPED, LifecycleState.CONFIGURED, LifecycleState.INITIALIZED):
                        self._transition(name, LifecycleState.SHUTDOWN, plugin.shutdown)
                except Exception:
                    pass
            raise RuntimeError(f"Startup failed. Rolled back successfully. Cause: {e}")
