import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { apiClient } from '../../../api/client';
import type { JobResponse } from '../../../api/types';
import { SystemStatusCard } from '../components/SystemStatusCard';
import { JobInspectorCard } from '../components/JobInspectorCard';
import { VideoMetadataCard } from '../components/VideoMetadataCard';
import { RiskTimeline } from '../components/RiskTimeline';
import { RegionRiskBreakdown } from '../components/RegionRiskBreakdown';
import { OutputVideoCard } from '../components/OutputVideoCard';
import { EmptyState } from '../components/EmptyState';
import { ErrorBanner } from '../components/ErrorBanner';
import { SAMPLE_JOB_RESULT } from '../components/SampleJobData';
import '../dashboard.css';

const RECENT_JOBS_STORAGE_KEY = 'crowddna_recent_jobs';

export default function DashboardPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlJobId = searchParams.get('job_id') || '';

  const [jobIdInput, setJobIdInput] = useState<string>(urlJobId);
  const [currentJob, setCurrentJob] = useState<JobResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [jobFetchError, setJobFetchError] = useState<string | null>(null);
  const [activeFrameIndex, setActiveFrameIndex] = useState<number>(0);
  const [recentJobIds, setRecentJobIds] = useState<string[]>(() => {
    try {
      const stored = localStorage.getItem(RECENT_JOBS_STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  const pollingRef = useRef<number | null>(null);

  // Save recent job ID
  const recordRecentJob = useCallback((id: string) => {
    if (!id || id.startsWith('job-demo-')) return;
    setRecentJobIds((prev) => {
      const filtered = prev.filter((item) => item !== id);
      const updated = [id, ...filtered].slice(0, 8);
      try {
        localStorage.setItem(RECENT_JOBS_STORAGE_KEY, JSON.stringify(updated));
      } catch {
        // Ignore storage write issues
      }
      return updated;
    });
  }, []);

  // Fetch job by ID
  const fetchJob = useCallback(
    async (id: string, isSilentPoll = false) => {
      if (!id.trim()) return;
      if (!isSilentPoll) {
        setLoading(true);
        setJobFetchError(null);
      }

      // Check for demo mock job
      if (id === SAMPLE_JOB_RESULT.job_id) {
        setCurrentJob(SAMPLE_JOB_RESULT);
        setActiveFrameIndex(0);
        setLoading(false);
        return;
      }

      try {
        const data = await apiClient.getJob(id.trim());
        setCurrentJob(data);
        recordRecentJob(id.trim());
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Job retrieval failed';
        if (!isSilentPoll) {
          setJobFetchError(msg);
        }
      } finally {
        if (!isSilentPoll) {
          setLoading(false);
        }
      }
    },
    [recordRecentJob]
  );

  // Handle URL query parameter changes
  useEffect(() => {
    if (urlJobId) {
      setJobIdInput(urlJobId);
      fetchJob(urlJobId);
    } else {
      setCurrentJob(null);
    }
  }, [urlJobId, fetchJob]);

  // Polling for queued/running jobs
  useEffect(() => {
    if (currentJob && (currentJob.status === 'queued' || currentJob.status === 'running')) {
      pollingRef.current = window.setInterval(() => {
        fetchJob(currentJob.job_id, true);
      }, 3000);
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [currentJob, fetchJob]);

  const handleLoadJob = (id: string) => {
    setSearchParams({ job_id: id });
    setJobIdInput(id);
    fetchJob(id);
  };

  const handleCancelJob = async (id: string) => {
    try {
      const cancelled = await apiClient.cancelJob(id);
      setCurrentJob(cancelled);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to cancel job';
      setJobFetchError(msg);
    }
  };

  const handleLoadDemoJob = () => {
    setSearchParams({ job_id: SAMPLE_JOB_RESULT.job_id });
    setJobIdInput(SAMPLE_JOB_RESULT.job_id);
    setCurrentJob(SAMPLE_JOB_RESULT);
    setActiveFrameIndex(0);
    setJobFetchError(null);
  };

  const handleClearJob = () => {
    setSearchParams(new URLSearchParams());
    setJobIdInput('');
    setCurrentJob(null);
    setJobFetchError(null);
    setActiveFrameIndex(0);
  };

  const timeline = currentJob?.result?.timeline || [];
  const metadata = currentJob?.result?.metadata;
  const fps = metadata?.fps || 30;
  const activeFrameEntry = timeline[activeFrameIndex] || null;

  return (
    <div className="dashboard-container" data-testid="dashboard-page">
      {/* Page Header */}
      <header className="dashboard-header">
        <div className="dashboard-header-row">
          <div>
            <h1 className="dashboard-title">
              <span aria-hidden="true">🛰️</span> Inference Results Dashboard
            </h1>
            <p className="dashboard-subtitle">
              Live pedestrian telemetry, regional risk classification, and interactive frame-by-frame scrubber
            </p>
          </div>
        </div>
      </header>

      {/* Top Diagnostics & Inspector Grid */}
      <div className="dashboard-top-grid">
        <SystemStatusCard />

        <JobInspectorCard
          currentJob={currentJob}
          loading={loading}
          jobIdInput={jobIdInput}
          onJobIdInputChange={setJobIdInput}
          onLoadJob={handleLoadJob}
          onCancelJob={handleCancelJob}
          onLoadDemoJob={handleLoadDemoJob}
          recentJobIds={recentJobIds}
          onSelectRecentJob={handleLoadJob}
          onClearJob={handleClearJob}
        />
      </div>

      {/* Job Fetch Error Alert */}
      {jobFetchError && (
        <div className="error-banner-panel" role="alert" data-testid="fetch-error-alert">
          <span className="error-banner-icon" aria-hidden="true">⚠️</span>
          <div className="error-banner-body">
            <span className="error-banner-title">Unable to Load Job</span>
            <span className="error-banner-message">{jobFetchError}</span>
          </div>
        </div>
      )}

      {/* Job Execution Failure / Cancellation Banner */}
      {currentJob && (currentJob.status === 'failed' || currentJob.status === 'cancelled') && (
        <ErrorBanner job={currentJob} onRetry={() => fetchJob(currentJob.job_id)} />
      )}

      {/* Main Content Area: Either Loaded Job Output or Empty State */}
      {currentJob ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
          {/* Video Metadata Card */}
          <VideoMetadataCard metadata={metadata} />

          {/* Risk Timeline Visualization & Scrubber */}
          {currentJob.status === 'completed' && timeline.length > 0 && (
            <>
              <RiskTimeline
                timeline={timeline}
                fps={fps}
                activeFrameIndex={activeFrameIndex}
                onSelectFrame={setActiveFrameIndex}
              />

              <RegionRiskBreakdown
                activeFrameEntry={activeFrameEntry}
                frameIndex={activeFrameIndex}
                fps={fps}
              />

              <OutputVideoCard
                outputVideoPath={currentJob.result?.output_video_path}
                jobId={currentJob.job_id}
              />
            </>
          )}

          {currentJob.status === 'running' && (
            <div
              className="glass-panel"
              style={{ padding: 'var(--space-8)', textAlign: 'center' }}
            >
              <div className="animate-spin" style={{ fontSize: '32px', marginBottom: '12px' }}>
                ⟳
              </div>
              <h3 className="text-h3" style={{ margin: '0 0 6px' }}>
                Inference Job Running
              </h3>
              <p className="text-body" style={{ margin: 0 }}>
                Processing pedestrian tracking, metric calibration, and region risk estimation...
              </p>
            </div>
          )}
        </div>
      ) : (
        <EmptyState
          onLoadDemoJob={handleLoadDemoJob}
          recentJobIds={recentJobIds}
          onSelectRecentJob={handleLoadJob}
        />
      )}
    </div>
  );
}
