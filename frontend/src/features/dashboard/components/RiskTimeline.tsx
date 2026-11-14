import { useEffect, useState, useRef, useMemo, type FC } from 'react';
import type { TimelineEntryDTO } from '../../../api/types';
import { getFrameRisk, type RiskLevel } from './riskUtils';

export interface RiskTimelineProps {
  timeline: TimelineEntryDTO[];
  fps?: number;
  activeFrameIndex: number;
  onSelectFrame: (index: number | ((prevIndex: number) => number)) => void;
}

export const RiskTimeline: FC<RiskTimelineProps> = ({
  timeline,
  fps = 30,
  activeFrameIndex,
  onSelectFrame,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef<number | null>(null);

  const totalFrames = timeline.length;
  const safeFps = fps > 0 ? fps : 30;

  // Compute frame-by-frame risk categories and summary statistics
  const { frameRisks, safeCount, congestingCount, criticalCount } = useMemo(() => {
    let s = 0;
    let c = 0;
    let cr = 0;
    const risks: RiskLevel[] = [];

    for (let i = 0; i < timeline.length; i++) {
      const risk = getFrameRisk(timeline[i]);
      risks.push(risk);
      if (risk === 'Critical') cr++;
      else if (risk === 'Congesting') c++;
      else s++;
    }

    return {
      frameRisks: risks,
      safeCount: s,
      congestingCount: c,
      criticalCount: cr,
    };
  }, [timeline]);

  const safePct = totalFrames > 0 ? ((safeCount / totalFrames) * 100).toFixed(1) : '0';
  const congestingPct =
    totalFrames > 0 ? ((congestingCount / totalFrames) * 100).toFixed(1) : '0';
  const criticalPct =
    totalFrames > 0 ? ((criticalCount / totalFrames) * 100).toFixed(1) : '0';

  // Active frame calculations
  const clampedIndex = Math.max(0, Math.min(activeFrameIndex, Math.max(0, totalFrames - 1)));
  const activeEntry = timeline[clampedIndex];
  const activeRisk = frameRisks[clampedIndex] || 'Safe';
  const activeFrameNumber = activeEntry?.frame_index ?? clampedIndex;
  const activeTimestamp = (activeFrameNumber / safeFps).toFixed(2);

  // Playback loop
  useEffect(() => {
    if (isPlaying && totalFrames > 1) {
      playIntervalRef.current = window.setInterval(() => {
        onSelectFrame((prevIndex) => {
          if (prevIndex >= totalFrames - 1) {
            setIsPlaying(false);
            return 0;
          }
          return prevIndex + 1;
        });
      }, 1000 / Math.min(safeFps, 15)); // Smooth preview playback rate
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, totalFrames, safeFps, onSelectFrame]);

  const handleStep = (step: number) => {
    setIsPlaying(false);
    const next = Math.max(0, Math.min(clampedIndex + step, totalFrames - 1));
    onSelectFrame(next);
  };

  const togglePlay = () => {
    if (clampedIndex >= totalFrames - 1 && !isPlaying) {
      onSelectFrame(0);
    }
    setIsPlaying(!isPlaying);
  };

  if (totalFrames === 0) {
    return (
      <section className="glass-panel timeline-card" aria-labelledby="risk-timeline-heading">
        <div className="dash-card-header">
          <h2 id="risk-timeline-heading" className="dash-card-title">
            <span aria-hidden="true">📈</span> Risk Timeline Visualization
          </h2>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>No frames</span>
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          No timeline data is available for this job.
        </p>
      </section>
    );
  }

  // Playhead percentage for the scrubber strip
  const playheadPercent =
    totalFrames > 1 ? (clampedIndex / (totalFrames - 1)) * 100 : 0;

  return (
    <section
      className="glass-panel timeline-card"
      aria-labelledby="risk-timeline-heading"
      data-testid="risk-timeline-section"
    >
      <div className="dash-card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <h2 id="risk-timeline-heading" className="dash-card-title">
            <span aria-hidden="true">📈</span> Risk Timeline Visualization
          </h2>
          <span
            className={`badge-risk ${
              activeRisk === 'Critical'
                ? 'badge-risk-critical'
                : activeRisk === 'Congesting'
                ? 'badge-risk-congesting'
                : 'badge-risk-safe'
            }`}
            data-testid="active-frame-risk-badge"
          >
            Frame #{activeFrameNumber}: {activeRisk}
          </span>
        </div>

        <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {clampedIndex + 1}
          </span>{' '}
          / {totalFrames} frames ({activeTimestamp}s)
        </div>
      </div>

      {/* Aggregate Stats Summary Bar */}
      <div className="timeline-stats-summary" data-testid="timeline-stats-summary">
        <div className="stat-pill" title="Total video frames in inference timeline">
          <span className="stat-pill-label">Total Frames:</span>
          <span className="stat-pill-value">{totalFrames}</span>
        </div>

        <div
          className="stat-pill"
          style={{ borderColor: 'var(--risk-safe-border)' }}
          title="Frames with normal pedestrian flow"
        >
          <span
            aria-hidden="true"
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: 'var(--risk-safe)',
            }}
          />
          <span className="stat-pill-label">Safe:</span>
          <span
            className="stat-pill-value"
            style={{ color: 'var(--risk-safe)' }}
            data-testid="stat-safe-count"
          >
            {safeCount} ({safePct}%)
          </span>
        </div>

        <div
          className="stat-pill"
          style={{ borderColor: 'var(--risk-congesting-border)' }}
          title="Frames experiencing pedestrian congestion or flow bottlenecks"
        >
          <span
            aria-hidden="true"
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: 'var(--risk-congesting)',
            }}
          />
          <span className="stat-pill-label">Congesting:</span>
          <span
            className="stat-pill-value"
            style={{ color: 'var(--risk-congesting)' }}
            data-testid="stat-congesting-count"
          >
            {congestingCount} ({congestingPct}%)
          </span>
        </div>

        <div
          className="stat-pill"
          style={{ borderColor: 'var(--risk-critical-border)' }}
          title="Frames with high risk or dangerous crowd density"
        >
          <span
            aria-hidden="true"
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: 'var(--risk-critical)',
            }}
          />
          <span className="stat-pill-label">Critical:</span>
          <span
            className="stat-pill-value"
            style={{ color: 'var(--risk-critical)' }}
            data-testid="stat-critical-count"
          >
            {criticalCount} ({criticalPct}%)
          </span>
        </div>
      </div>

      {/* Interactive Timeline Heatmap Bar */}
      <div className="timeline-strip-container">
        <div
          className="timeline-strip"
          role="region"
          aria-label="Click to scrub frame timeline"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const fraction = Math.max(0, Math.min(1, clickX / rect.width));
            const targetIdx = Math.round(fraction * (totalFrames - 1));
            setIsPlaying(false);
            onSelectFrame(targetIdx);
          }}
        >
          {timeline.map((entry, idx) => {
            const r = frameRisks[idx];
            const isSelected = idx === clampedIndex;
            const segmentClass =
              r === 'Critical'
                ? 'timeline-segment-critical'
                : r === 'Congesting'
                ? 'timeline-segment-congesting'
                : 'timeline-segment-safe';

            return (
              <div
                key={entry.frame_index ?? idx}
                className={`timeline-segment ${segmentClass} ${
                  isSelected ? 'active-segment' : ''
                }`}
                title={`Frame #${entry.frame_index ?? idx}: ${r}`}
                data-testid={`timeline-tick-${idx}`}
              />
            );
          })}

          {/* Scrubber Playhead Line */}
          <div
            className="timeline-scrubber-playhead"
            style={{ left: `${playheadPercent}%` }}
            aria-hidden="true"
          />
        </div>
      </div>

      {/* Scrubber Controls */}
      <div className="timeline-controls">
        <div className="playback-controls">
          <button
            type="button"
            className="glass-button"
            onClick={() => {
              setIsPlaying(false);
              onSelectFrame(0);
            }}
            disabled={clampedIndex === 0}
            title="Jump to Start"
            aria-label="Jump to first frame"
            style={{ padding: '6px 10px' }}
          >
            ⏮
          </button>
          <button
            type="button"
            className="glass-button"
            onClick={() => handleStep(-1)}
            disabled={clampedIndex === 0}
            title="Previous Frame"
            aria-label="Step to previous frame"
            style={{ padding: '6px 12px' }}
          >
            ◀
          </button>
          <button
            type="button"
            className="glass-button glass-button-primary"
            onClick={togglePlay}
            title={isPlaying ? 'Pause' : 'Play Simulation'}
            aria-label={isPlaying ? 'Pause timeline playback' : 'Play timeline playback'}
            style={{ padding: '6px 16px', minWidth: '70px' }}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            type="button"
            className="glass-button"
            onClick={() => handleStep(1)}
            disabled={clampedIndex >= totalFrames - 1}
            title="Next Frame"
            aria-label="Step to next frame"
            style={{ padding: '6px 12px' }}
          >
            ▶
          </button>
          <button
            type="button"
            className="glass-button"
            onClick={() => {
              setIsPlaying(false);
              onSelectFrame(totalFrames - 1);
            }}
            disabled={clampedIndex >= totalFrames - 1}
            title="Jump to End"
            aria-label="Jump to last frame"
            style={{ padding: '6px 10px' }}
          >
            ⏭
          </button>
        </div>

        {/* Range Slider Scrubber */}
        <div className="scrubber-slider-wrapper">
          <input
            type="range"
            className="scrubber-slider"
            min={0}
            max={totalFrames - 1}
            value={clampedIndex}
            onChange={(e) => {
              setIsPlaying(false);
              onSelectFrame(Number(e.target.value));
            }}
            aria-label="Frame scrubber"
            aria-valuenow={clampedIndex + 1}
            aria-valuemin={1}
            aria-valuemax={totalFrames}
            aria-valuetext={`Frame ${clampedIndex + 1} of ${totalFrames}`}
            data-testid="timeline-scrubber-slider"
          />
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              color: 'var(--text-muted)',
              minWidth: '55px',
              textAlign: 'right',
            }}
          >
            {activeTimestamp}s
          </span>
        </div>
      </div>
    </section>
  );
};
