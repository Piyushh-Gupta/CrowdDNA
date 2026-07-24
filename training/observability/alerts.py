from dataclasses import dataclass
from typing import Callable, List
from training.observability.metadata import MetricObservation, AlertObservation, EventPriority
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

@dataclass(frozen=True)
class AlertRule:
    name: str
    metric_name: str
    threshold: float
    severity: EventPriority
    comparator: Callable[[float, float], bool]  # e.g. lambda val, thresh: val > thresh
    message_template: str
    cooldown_sec: float = 60.0

class AlertEngine:
    """Evaluates rules against incoming MetricObservations to emit AlertObservations."""
    
    def __init__(self, buffer: ObservationBuffer):
        self._buffer = buffer
        self._rules: List[AlertRule] = []
        self._last_triggered: dict = {}
        
    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)
        
    def evaluate(self, metric: MetricObservation) -> None:
        """Evaluates a single metric against all applicable rules."""
        now = Clock.now()
        for rule in self._rules:
            if rule.metric_name == metric.name:
                if rule.comparator(metric.value, rule.threshold):
                    last = self._last_triggered.get(rule.name, 0.0)
                    if now - last >= rule.cooldown_sec:
                        alert = AlertObservation(
                            timestamp=now,
                            alert_name=rule.name,
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=rule.message_template.format(value=metric.value, threshold=rule.threshold),
                            trigger_value=metric.value
                        )
                        self._buffer.add(alert)
                        self._last_triggered[rule.name] = now
