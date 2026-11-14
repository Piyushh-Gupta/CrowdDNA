import type { FC } from 'react';
import { Link } from 'react-router-dom';
import type { JobResponse } from '../../../api/types';

export interface ErrorBannerProps {
  job: JobResponse;
  onRetry?: () => void;
}

export const ErrorBanner: FC<ErrorBannerProps> = ({ job, onRetry }) => {
  const isCancelled = job.status === 'cancelled';
  const isFailed = job.status === 'failed';

  if (!isCancelled && !isFailed) return null;

  return (
    <div
      className="error-banner-panel"
      style={{
        background: isCancelled ? 'rgba(161, 161, 170, 0.12)' : 'rgba(239, 68, 68, 0.12)',
        borderColor: isCancelled ? 'rgba(161, 161, 170, 0.35)' : 'rgba(239, 68, 68, 0.35)',
        color: isCancelled ? '#e4e4e7' : '#fca5a5',
      }}
      role="alert"
      data-testid="job-error-banner"
    >
      <span className="error-banner-icon" aria-hidden="true">
        {isCancelled ? '⏹️' : '⚠️'}
      </span>
      <div className="error-banner-body" style={{ flex: 1 }}>
        <span
          className="error-banner-title"
          style={{ color: isCancelled ? '#f4f4f5' : '#fecaca' }}
        >
          {isCancelled ? 'Inference Job Cancelled' : 'Inference Job Execution Failed'}
        </span>
        <span className="error-banner-message">
          {job.error ||
            (isCancelled
              ? 'This job was cancelled before execution was finished.'
              : 'The inference pipeline encountered an unrecoverable processing error.')}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '8px', alignSelf: 'center' }}>
        {onRetry && (
          <button
            type="button"
            className="glass-button"
            onClick={onRetry}
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            Retry Fetch
          </button>
        )}
        <Link
          to="/inference"
          className="glass-button glass-button-primary"
          style={{ padding: '6px 12px', fontSize: '12px' }}
        >
          New Inference Job
        </Link>
      </div>
    </div>
  );
};
