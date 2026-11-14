import type { FC } from 'react';
import { Link } from 'react-router-dom';

export interface EmptyStateProps {
  onLoadDemoJob: () => void;
  recentJobIds: string[];
  onSelectRecentJob: (jobId: string) => void;
}

export const EmptyState: FC<EmptyStateProps> = ({
  onLoadDemoJob,
  recentJobIds,
  onSelectRecentJob,
}) => {
  return (
    <div
      className="glass-panel empty-state-panel"
      role="region"
      aria-labelledby="empty-state-title"
      data-testid="dashboard-empty-state"
    >
      <div className="empty-state-icon" aria-hidden="true">
        📊
      </div>

      <div style={{ maxWidth: '520px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <h3 id="empty-state-title" className="text-h2" style={{ margin: 0 }}>
          No Inference Results Loaded
        </h3>
        <p className="text-body" style={{ margin: 0 }}>
          Inspect pedestrian flow metrics, regional risk classifications, and frame-by-frame
          trajectories by loading a completed inference job or submitting a new video stream.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
        <Link
          to="/inference"
          className="glass-button glass-button-primary"
          style={{ padding: '10px 20px', fontSize: '15px' }}
        >
          🚀 Submit Video in Inference
        </Link>
        <button
          type="button"
          className="glass-button"
          onClick={onLoadDemoJob}
          style={{ padding: '10px 20px', fontSize: '15px' }}
        >
          ✨ Load Sample Demonstration Job
        </button>
      </div>

      {recentJobIds.length > 0 && (
        <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Or inspect a recently processed job:
          </span>
          <div
            style={{
              display: 'flex',
              gap: '8px',
              flexWrap: 'wrap',
              justifyContent: 'center',
            }}
          >
            {recentJobIds.slice(0, 5).map((id) => (
              <button
                key={id}
                type="button"
                className="recent-chip"
                onClick={() => onSelectRecentJob(id)}
              >
                {id}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
