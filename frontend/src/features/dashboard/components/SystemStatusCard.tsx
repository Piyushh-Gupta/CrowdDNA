import { useState, useEffect, useCallback, type FC } from 'react';
import { apiClient } from '../../../api/client';
import type { HealthStatus, ReadyStatus, VersionInfo } from '../../../api/types';

export interface SystemStatusCardProps {
  onHealthLoaded?: (status: HealthStatus | null) => void;
}

export const SystemStatusCard: FC<SystemStatusCardProps> = ({ onHealthLoaded }) => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [ready, setReady] = useState<ReadyStatus | null>(null);
  const [version, setVersion] = useState<VersionInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [hRes, rRes, vRes] = await Promise.all([
        apiClient.getHealth().catch((err) => {
          console.warn('getHealth failed:', err);
          return null;
        }),
        apiClient.getReady().catch((err) => {
          console.warn('getReady failed:', err);
          return null;
        }),
        apiClient.getVersion().catch((err) => {
          console.warn('getVersion failed:', err);
          return null;
        }),
      ]);

      if (!hRes && !rRes) {
        throw new Error('API server is unreachable. Please verify backend service.');
      }

      setHealth(hRes);
      setReady(rRes);
      setVersion(vRes);
      setLastChecked(new Date());
      if (onHealthLoaded) onHealthLoaded(hRes);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to query system status';
      setError(msg);
      if (onHealthLoaded) onHealthLoaded(null);
    } finally {
      setLoading(false);
    }
  }, [onHealthLoaded]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const isHealthy = health?.status === 'healthy';
  const isReady = ready?.status === 'ready' || ready?.ready === true;
  const isModelLoaded = ready?.model_loaded ?? false;
  const device = ready?.device || 'cpu';
  const apiVersion = version?.api_version || health?.version || '1.0.0';
  const engineVersion = version?.engine_version || '1.0.0';

  return (
    <section className="glass-panel dash-card" aria-labelledby="system-status-heading">
      <div className="dash-card-header">
        <h2 id="system-status-heading" className="dash-card-title">
          <span aria-hidden="true">⚡</span> System Status
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {loading ? (
            <span className="badge-status badge-status-running" aria-live="polite">
              <span className="animate-spin" aria-hidden="true">⟳</span> Checking...
            </span>
          ) : error ? (
            <span className="badge-status badge-status-failed" aria-label="System status degraded">
              Offline
            </span>
          ) : (
            <span
              className={`badge-status ${isHealthy ? 'badge-status-completed' : 'badge-status-failed'}`}
              aria-label={`Service is ${health?.status || 'Unknown'}`}
            >
              <span aria-hidden="true">●</span> {health?.status || 'Active'}
            </span>
          )}
          <button
            type="button"
            className="glass-button"
            onClick={fetchStatus}
            disabled={loading}
            aria-label="Refresh system status"
            title="Refresh system status"
            style={{ padding: '4px 10px', fontSize: '12px' }}
          >
            {loading ? '...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="error-banner-panel" role="alert">
          <span className="error-banner-icon" aria-hidden="true">⚠️</span>
          <div className="error-banner-body">
            <span className="error-banner-title">API Connection Warning</span>
            <span className="error-banner-message">{error}</span>
          </div>
        </div>
      ) : (
        <div className="metrics-row">
          <div className="metric-cell">
            <span className="metric-cell-label">API Version</span>
            <span className="metric-cell-value" data-testid="sys-api-version">
              {loading && !version ? '—' : `v${apiVersion}`}
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-cell-label">Engine Version</span>
            <span className="metric-cell-value" data-testid="sys-engine-version">
              {loading && !version ? '—' : `v${engineVersion}`}
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-cell-label">Hardware Device</span>
            <span
              className="metric-cell-value"
              data-testid="sys-device"
              style={{ textTransform: 'uppercase' }}
            >
              {loading && !ready ? '—' : device}
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-cell-label">Readiness</span>
            <span
              className="metric-cell-value"
              data-testid="sys-model-status"
              style={{
                fontSize: '14px',
                color: isReady || isModelLoaded ? 'var(--risk-safe)' : 'var(--risk-congesting)',
              }}
            >
              {isModelLoaded ? 'Model Loaded' : isReady ? 'Ready (Standby)' : 'Initializing'}
            </span>
          </div>
        </div>
      )}

      {lastChecked && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'right' }}>
          Last checked: {lastChecked.toLocaleTimeString()}
        </div>
      )}
    </section>
  );
};
