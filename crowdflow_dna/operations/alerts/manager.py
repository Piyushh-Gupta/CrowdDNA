import threading

class AlertManager:
    def __init__(self, throttler, rules):
        self.throttler = throttler
        self.rules = rules
        self.active_alerts = set()
        self.lock = threading.Lock()
        
    def trigger_alert(self, alert_id):
        # Deduplication
        with self.lock:
            if alert_id in self.active_alerts:
                return False # Duplicate
            self.active_alerts.add(alert_id)
            
        # Throttling
        if not self.throttler.is_throttled(alert_id):
            return True
        return False
