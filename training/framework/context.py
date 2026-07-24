from dataclasses import dataclass
from typing import Any
from training.framework.metadata import FrameworkRuntime

@dataclass(frozen=True)
class FrameworkContext:
    runtime: FrameworkRuntime
    environment: str

@dataclass(frozen=True)
class PluginContext:
    configuration: Any # ResolvedConfiguration
    logger: Any # Centralized logger
    clock: Any # Immutable clock
    framework_context: FrameworkContext
    metrics: Any # Metrics interface
    cancellation_token: Any
