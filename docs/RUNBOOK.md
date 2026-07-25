# Operational Runbook

## Maintenance Mode
To enable maintenance mode, toggle `MAINTENANCE_ACTIVE=true` in the configuration. The API will respond with 503 Retry-After.

## Health Verification
Check `/api/v1/health` for component status (DB, inference engine, queues).

## Backup & Restore
- **Backup**: Execute `scripts/backup.sh` to snapshot the database and generated datasets.
- **Restore**: Execute `scripts/restore.sh <timestamp>` to revert to a snapshot.

## Rollback
To rollback a deployment, use `scripts/rollback.sh <previous_tag>` to restart the container orchestrator with the last known good image.

## Incident Response
1. Acknowledge PagerDuty alert.
2. Review Grafana observability exports.
3. Drain traffic.
4. Scale up resources or rollback offending release.
