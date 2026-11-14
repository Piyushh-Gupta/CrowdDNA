import type { FC } from 'react';
import type { RiskPredictionDTO, TimelineEntryDTO } from '../../../api/types';

export interface RegionRiskBreakdownProps {
  activeFrameEntry?: TimelineEntryDTO | null;
  frameIndex: number;
  fps?: number;
}

const getRiskColor = (label: string): { text: string; bg: string; fill: string } => {
  const l = label.toLowerCase();
  if (l.includes('critical')) {
    return {
      text: 'var(--risk-critical)',
      bg: 'var(--risk-critical-bg)',
      fill: 'var(--risk-critical)',
    };
  }
  if (l.includes('congesting')) {
    return {
      text: 'var(--risk-congesting)',
      bg: 'var(--risk-congesting-bg)',
      fill: 'var(--risk-congesting)',
    };
  }
  return {
    text: 'var(--risk-safe)',
    bg: 'var(--risk-safe-bg)',
    fill: 'var(--risk-safe)',
  };
};

export const RegionRiskBreakdown: FC<RegionRiskBreakdownProps> = ({
  activeFrameEntry,
  frameIndex,
  fps = 30,
}) => {
  const predictions: RiskPredictionDTO[] = activeFrameEntry?.predictions || [];
  const safeFps = fps > 0 ? fps : 30;
  const frameNum = activeFrameEntry?.frame_index ?? frameIndex;
  const timestamp = (frameNum / safeFps).toFixed(2);

  const safeCount = predictions.filter((p) => p.label.toLowerCase().includes('safe')).length;
  const congestingCount = predictions.filter((p) =>
    p.label.toLowerCase().includes('congesting')
  ).length;
  const criticalCount = predictions.filter((p) =>
    p.label.toLowerCase().includes('critical')
  ).length;

  return (
    <section className="glass-panel dash-card" aria-labelledby="region-breakdown-heading">
      <div className="dash-card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <h2 id="region-breakdown-heading" className="dash-card-title">
            <span aria-hidden="true">🗺️</span> Region-Level Risk & Confidence
          </h2>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Frame #{frameNum} ({timestamp}s)
          </span>
        </div>

        {predictions.length > 0 && (
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {criticalCount > 0 && (
              <span className="badge-risk badge-risk-critical">
                {criticalCount} Critical
              </span>
            )}
            {congestingCount > 0 && (
              <span className="badge-risk badge-risk-congesting">
                {congestingCount} Congesting
              </span>
            )}
            {safeCount > 0 && (
              <span className="badge-risk badge-risk-safe">
                {safeCount} Safe
              </span>
            )}
          </div>
        )}
      </div>

      {predictions.length === 0 ? (
        <div style={{ padding: 'var(--space-4)', textAlign: 'center', color: 'var(--text-muted)' }}>
          No spatial regions tracked for Frame #{frameNum}.
        </div>
      ) : (
        <div className="regions-grid" data-testid="regions-grid">
          {predictions.map((p) => {
            const confPct = Math.min(100, Math.max(0, p.confidence * 100));
            const color = getRiskColor(p.label);

            return (
              <div
                key={p.region_id}
                className="region-item-card"
                data-testid={`region-card-${p.region_id}`}
              >
                <div className="region-header">
                  <span className="region-id-label">Region #{p.region_id}</span>
                  <span
                    className={`badge-risk ${
                      p.label.toLowerCase().includes('critical')
                        ? 'badge-risk-critical'
                        : p.label.toLowerCase().includes('congesting')
                        ? 'badge-risk-congesting'
                        : 'badge-risk-safe'
                    }`}
                  >
                    {p.label}
                  </span>
                </div>

                <div className="confidence-bar-container">
                  <div className="confidence-bar-header">
                    <span>Model Confidence</span>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 600,
                        color: color.text,
                      }}
                    >
                      {confPct.toFixed(1)}%
                    </span>
                  </div>

                  <div
                    className="confidence-bar-track"
                    role="progressbar"
                    aria-valuenow={Math.round(confPct)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`Region ${p.region_id} confidence: ${confPct.toFixed(1)}%`}
                  >
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${confPct}%`,
                        backgroundColor: color.fill,
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};
