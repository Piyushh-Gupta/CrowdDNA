import { useState, type FC } from 'react';
import { apiClient } from '../../../api/client';

export interface OutputVideoCardProps {
  outputVideoPath?: string | null;
  jobId?: string;
}

export const OutputVideoCard: FC<OutputVideoCardProps> = ({ outputVideoPath, jobId }) => {
  const [copied, setCopied] = useState(false);
  const [videoLoadFailed, setVideoLoadFailed] = useState(false);

  const handleCopy = () => {
    if (outputVideoPath) {
      navigator.clipboard.writeText(outputVideoPath).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  const isHttpUrl =
    outputVideoPath?.startsWith('http://') || outputVideoPath?.startsWith('https://');

  // Derive video streaming URL from backend endpoint or direct HTTP URL
  const videoStreamUrl = isHttpUrl
    ? outputVideoPath
    : jobId
    ? apiClient.getVideoUrl(jobId)
    : null;

  return (
    <section className="glass-panel dash-card" aria-labelledby="output-video-heading">
      <div className="dash-card-header">
        <h2 id="output-video-heading" className="dash-card-title">
          <span aria-hidden="true">🎥</span> Output Annotated Video
        </h2>
        {outputVideoPath ? (
          <span className="badge-risk badge-risk-safe" style={{ fontSize: '11px' }}>
            Generated
          </span>
        ) : (
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Not Available</span>
        )}
      </div>

      {outputVideoPath ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: 0 }}>
            Annotated bounding boxes, regional density heatmaps, and crowd danger classifications have been
            rendered to the inference output stream.
          </p>

          {/* Video Player */}
          {videoStreamUrl && (
            <div className="output-video-player-container" style={{ position: 'relative' }}>
              <video
                controls
                playsInline
                preload="metadata"
                src={videoStreamUrl}
                data-testid="output-video-player"
                style={{
                  width: '100%',
                  maxHeight: '420px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#0a0d14',
                  border: '1px solid var(--border-glass)',
                  display: videoLoadFailed ? 'none' : 'block',
                }}
                onError={() => setVideoLoadFailed(true)}
              >
                Your browser does not support the video tag.
              </video>

              {videoLoadFailed && (
                <div
                  className="video-fallback-banner"
                  data-testid="video-fallback-banner"
                  style={{
                    padding: 'var(--space-3)',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--bg-glass-subtle)',
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    border: '1px dashed var(--border-glass)',
                  }}
                >
                  ℹ️ In-browser video streaming is not supported for offline mock jobs, or video codec is unsupported. You can copy the local filesystem path below.
                </div>
              )}
            </div>
          )}

          {/* Secondary File Path & Actions */}
          <div className="output-video-box">
            <span
              style={{
                color: 'var(--text-primary)',
                wordBreak: 'break-all',
              }}
              data-testid="output-video-path"
            >
              {outputVideoPath}
            </span>
            <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
              <button
                type="button"
                className="glass-button"
                onClick={handleCopy}
                style={{ padding: '4px 10px', fontSize: '12px' }}
                aria-label="Copy output video file path"
              >
                {copied ? '✓ Copied' : 'Copy Path'}
              </button>
              {isHttpUrl && (
                <a
                  href={outputVideoPath}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="glass-button glass-button-primary"
                  style={{ padding: '4px 10px', fontSize: '12px' }}
                >
                  Open ↗
                </a>
              )}
            </div>
          </div>
        </div>
      ) : (
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: 0 }}>
          No annotated video output was produced for this job, or the job is still queued/running.
        </p>
      )}
    </section>
  );
};
