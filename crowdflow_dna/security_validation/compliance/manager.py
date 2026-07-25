import threading
from typing import Tuple
from ..metadata import Control, ControlResult

class ComplianceManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._controls = {}

    def register_control(self, control: Control):
        with self._lock:
            self._controls[control.id] = control

    def evaluate_controls(self, findings: Tuple) -> Tuple[ControlResult, ...]:
        with self._lock:
            return tuple(
                ControlResult(control_id=cid, passed=True, findings=findings)
                for cid in self._controls.keys()
            )
