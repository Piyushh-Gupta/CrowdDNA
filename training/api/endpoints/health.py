from typing import Any
from training.api.routing import Router
from training.api.health.startup import check_startup
from training.api.health.readiness import check_readiness
from training.api.health.liveness import check_liveness

def health_handler(request: Any, context: Any):
    return {
        "startup": check_startup(),
        "readiness": check_readiness(),
        "liveness": check_liveness()
    }

Router.add_route("/api/v1/health", "GET", health_handler)
