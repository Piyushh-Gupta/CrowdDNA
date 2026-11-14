import { useState, type FC, type FormEvent } from 'react';
import type { JobResponse, JobStatusEnum } from '../../../api/types';

export interface JobInspectorCardProps {
  currentJob: JobResponse | null;
  loading: boolean;
  jobIdInput: string;
  onJobIdInputChange: (val: string) => void;
  onLoadJob: (jobId: string) => void;
  onCancelJob: (jobId: string) => void;
  onLoadDemoJob: () => void;
  recentJobIds: string[];
  onSelectRecentJob: (jobId: string) => void;
  onClearJob: () => void;
}

const formatDate = (isoString?: string | null): string => {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return isoString;
  }
};

const calculateDuration = (
  startStr?: string | null,
  endStr?: string | null
): string | null => {
  if (!startStr || !endStr) return null;
  try {
    const start = new Date(startStr).getTime();
    const end = new Date(endStr).getTime();
    if (isNaN(start) || isNaN(end) || end < start) return null;
    const diffSec = (end - start) / 1000;
    if (diffSec < 60) return `${diffSec.toFixed(1)}s`;
    const min = Math.floor(diffSec / 60);
    const remSec = (diffSec % 60).toFixed(0);
    return `${min}m ${remSec}s`;
  } catch {
    return null;
  }
};

const getStatusBadgeClass = (status: JobStatusEnum): string => {
  switch (status) {
    case 'completed':
      return 'badge-status-completed';
    case 'running':
      return 'badge-status-running';
    case 'queued':
      return 'badge-status-queued';
    case 'failed':
      return 'badge-status-failed';
    case 'cancelled':
      return 'badge-status-cancelled';
    default:
      return 'badge-status-queued';
  }
};

export const JobInspectorCard: FC<JobInspectorCardProps> = ({
  currentJob,
  loading,
  jobIdInput,
  onJobIdInputChange,
  onLoadJob,
  onCancelJob,
  onLoadDemoJob,
  recentJobIds,
  onSelectRecentJob,
  onClearJob,
}) => {
  const [copied, setCopied] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = jobIdInput.trim();
    if (trimmed) {
      onLoadJob(trimmed);
    }
  };

  const handleCopyId = () => {
    if (currentJob?.job_id) {
      navigator.clipboard.writeText(currentJob.job_id).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  const isTerminal =
    currentJob?.status === 'completed' ||
    currentJob?.status === 'failed' ||
    currentJob?.status === 'cancelled';

  const duration = calculateDuration(currentJob?.started_at, currentJob?.completed_at);

  return (
    <section className="glass-panel dash-card" aria-labelledby="job-inspector-heading">
      <div className="dash-card-header">
        <h2 id="job-inspector-heading" className="dash-card-title">
          <span aria-hidden="true">🔍</span> Inference Job Inspector
        </h2>
        {currentJob && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              className={`badge-status ${getStatusBadgeClass(currentJob.status)}`}
              data-testid="job-status-badge"
              aria-label={`Job status: ${currentJob.status}`}
            >
              {currentJob.status === 'running' && (
                <span className="animate-spin" aria-hidden="true">⟳</span>
              )}
              {currentJob.status}
            </span>
            <button
              type="button"
              className="glass-button"
              onClick={onClearJob}
              style={{ padding: '3px 8px', fontSize: '11px' }}
              title="Inspect another job"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Search Input Bar */}
      <form onSubmit={handleSubmit} className="job-search-bar" role="search">
        <input
          type="text"
          className="glass-input job-input"
          placeholder="Enter Job ID (e.g. 8f3a9b1c-...)"
          value={jobIdInput}
          onChange={(e) => onJobIdInputChange(e.target.value)}
          aria-label="Job ID to inspect"
        />
        <button
          type="submit"
          className="glass-button glass-button-primary"
          disabled={loading || !jobIdInput.trim()}
          aria-label="Inspect Job"
        >
          {loading ? 'Loading...' : 'Inspect'}
        </button>
        <button
          type="button"
          className="glass-button"
          onClick={onLoadDemoJob}
          title="Load demonstration job with real metrics"
          aria-label="Load demo job"
        >
          Demo Job
        </button>
      </form>

      {/* Recent Jobs History */}
      {recentJobIds.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Recent Jobs:</span>
          <div className="recent-jobs-chips">
            {recentJobIds.map((id) => (
              <button
                key={id}
                type="button"
                className="recent-chip"
                onClick={() => onSelectRecentJob(id)}
                title={`Inspect ${id}`}
              >
                {id.length > 18 ? `${id.slice(0, 8)}...${id.slice(-6)}` : id}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Active Job Details */}
      {currentJob && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'var(--bg-surface-2)',
              padding: '8px 12px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>ID:</span>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '13px',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  wordBreak: 'break-all',
                }}
                data-testid="job-id-display"
              >
                {currentJob.job_id}
              </span>
            </div>
            <button
              type="button"
              className="glass-button"
              onClick={handleCopyId}
              style={{ padding: '3px 8px', fontSize: '11px', flexShrink: 0 }}
              title="Copy Job ID"
            >
              {copied ? '✓ Copied' : 'Copy'}
            </button>
          </div>

          <div className="metrics-row">
            <div className="metric-cell">
              <span className="metric-cell-label">Created At</span>
              <span
                className="metric-cell-value"
                style={{ fontSize: '12px' }}
                data-testid="job-created-at"
              >
                {formatDate(currentJob.created_at)}
              </span>
            </div>

            <div className="metric-cell">
              <span className="metric-cell-label">Started At</span>
              <span
                className="metric-cell-value"
                style={{ fontSize: '12px' }}
                data-testid="job-started-at"
              >
                {formatDate(currentJob.started_at)}
              </span>
            </div>

            <div className="metric-cell">
              <span className="metric-cell-label">Completed At</span>
              <span
                className="metric-cell-value"
                style={{ fontSize: '12px' }}
                data-testid="job-completed-at"
              >
                {formatDate(currentJob.completed_at)}
              </span>
            </div>

            {duration && (
              <div className="metric-cell">
                <span className="metric-cell-label">Duration</span>
                <span
                  className="metric-cell-value"
                  style={{ fontSize: '13px', color: 'var(--primary)' }}
                  data-testid="job-duration"
                >
                  {duration}
                </span>
              </div>
            )}
          </div>

          {/* Action Row for Running Jobs */}
          {!isTerminal && (
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingTop: '6px',
              }}
            >
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Job is currently {currentJob.status}...
              </span>
              <button
                type="button"
                className="glass-button"
                onClick={() => onCancelJob(currentJob.job_id)}
                style={{ color: 'var(--status-failed)', borderColor: 'rgba(244, 63, 94, 0.4)' }}
                aria-label="Cancel job"
              >
                Cancel Job
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
};
