import type { FC } from 'react';
import type { VideoMetadataDTO } from '../../../api/types';

export interface VideoMetadataCardProps {
  metadata?: VideoMetadataDTO | null;
}

export const VideoMetadataCard: FC<VideoMetadataCardProps> = ({ metadata }) => {
  if (!metadata) {
    return (
      <section className="glass-panel dash-card" aria-labelledby="video-metadata-heading">
        <div className="dash-card-header">
          <h2 id="video-metadata-heading" className="dash-card-title">
            <span aria-hidden="true">🎬</span> Video Stream Metadata
          </h2>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Unavailable</span>
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          Video stream metadata is extracted upon inference completion. No metadata available for this job.
        </p>
      </section>
    );
  }

  const { width, height, fps, frame_count, duration_seconds, sample_rate } = metadata;
  const resolution = width && height ? `${width} × ${height}` : '—';
  const aspectRatio =
    width && height
      ? width / height >= 1.77
        ? '16:9'
        : width / height >= 1.3
        ? '4:3'
        : `${(width / height).toFixed(2)}:1`
      : '—';

  const formattedDuration =
    duration_seconds > 0
      ? duration_seconds < 60
        ? `${duration_seconds.toFixed(2)}s`
        : `${Math.floor(duration_seconds / 60)}m ${(duration_seconds % 60).toFixed(1)}s`
      : '0.00s';

  return (
    <section className="glass-panel dash-card" aria-labelledby="video-metadata-heading">
      <div className="dash-card-header">
        <h2 id="video-metadata-heading" className="dash-card-title">
          <span aria-hidden="true">🎬</span> Video Stream Metadata
        </h2>
        <span
          className="badge-risk badge-risk-safe"
          style={{ fontSize: '11px', textTransform: 'uppercase' }}
          aria-label="Metadata ingested"
        >
          Stream Ingested
        </span>
      </div>

      <div className="metrics-row">
        <div className="metric-cell">
          <span className="metric-cell-label">Resolution</span>
          <span className="metric-cell-value" data-testid="meta-resolution">
            {resolution}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{aspectRatio} Aspect</span>
        </div>

        <div className="metric-cell">
          <span className="metric-cell-label">Frame Rate (FPS)</span>
          <span className="metric-cell-value" data-testid="meta-fps">
            {typeof fps === 'number' ? `${fps.toFixed(1)}` : '—'}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Frames / sec</span>
        </div>

        <div className="metric-cell">
          <span className="metric-cell-label">Total Frames</span>
          <span className="metric-cell-value" data-testid="meta-total-frames">
            {frame_count != null ? frame_count.toLocaleString() : '—'}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Analyzed frames</span>
        </div>

        <div className="metric-cell">
          <span className="metric-cell-label">Duration</span>
          <span className="metric-cell-value" data-testid="meta-duration">
            {formattedDuration}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Sample Rate: {sample_rate ?? 1}x
          </span>
        </div>
      </div>
    </section>
  );
};
