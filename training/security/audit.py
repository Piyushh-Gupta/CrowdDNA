import time
import uuid
import threading
from typing import List, Dict, Any, Optional
from training.security.metadata import AuditRecord, SecurityDecision
from types import MappingProxyType

class AuditLogger:
    def __init__(self, strict: bool = False):
        self._strict = strict
        self._records: List[AuditRecord] = []
        self._lock = threading.Lock()

    def log(self, action: str, principal_id: str, resource_id: Optional[str], decision: SecurityDecision, context_data: Dict[str, Any] = None):
        try:
            record = AuditRecord(
                audit_id=str(uuid.uuid4()),
                timestamp=time.time(),
                action=action,
                principal_id=principal_id,
                resource_id=resource_id,
                decision=decision,
                context_data=MappingProxyType(context_data or {})
            )
            # In a real scenario, this would write to an append-only storage backend.
            with self._lock:
                self._records.append(record)
        except Exception as e:
            if self._strict:
                raise RuntimeError(f"Audit log failed: {e}")
            # Best-effort
            pass

    def get_records(self) -> List[AuditRecord]:
        with self._lock:
            return list(self._records)
