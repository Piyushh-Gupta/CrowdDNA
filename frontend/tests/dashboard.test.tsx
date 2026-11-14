import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import DashboardPage from '../src/features/dashboard/pages';
import { apiClient } from '../src/api/client';
import { SAMPLE_JOB_RESULT } from '../src/features/dashboard/components/SampleJobData';
import type { HealthStatus, ReadyStatus, VersionInfo, JobResponse } from '../src/api/types';

describe('DashboardPage Component Integration Tests', () => {
  const mockHealth: HealthStatus = {
    status: 'healthy',
    timestamp: '2026-11-14T12:00:00Z',
    version: '1.0.0',
  };

  const mockReady: ReadyStatus = {
    status: 'ready',
    ready: true,
    model_loaded: true,
    device: 'cuda:0',
  };

  const mockVersion: VersionInfo = {
    api_version: '1.0.0',
    engine_version: '1.0.0',
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();

    vi.spyOn(apiClient, 'getHealth').mockResolvedValue(mockHealth);
    vi.spyOn(apiClient, 'getReady').mockResolvedValue(mockReady);
    vi.spyOn(apiClient, 'getVersion').mockResolvedValue(mockVersion);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const renderDashboard = (initialEntries = ['/']) =>
    render(
      <MemoryRouter initialEntries={initialEntries}>
        <DashboardPage />
      </MemoryRouter>
    );

  it('renders initial dashboard page with system status diagnostics', async () => {
    renderDashboard();

    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
    expect(screen.getByText(/Inference Results Dashboard/i)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /system status/i })).toBeInTheDocument();

    // Verify system diagnostics are populated from API
    expect(await screen.findByTestId('sys-engine-version')).toHaveTextContent('v1.0.0');
    expect(screen.getByTestId('sys-device')).toHaveTextContent('cuda:0');
    expect(screen.getByTestId('sys-model-status')).toHaveTextContent('Model Loaded');
  });

  it('renders empty state initially when no job is inspected', async () => {
    renderDashboard();

    expect(screen.getByTestId('dashboard-empty-state')).toBeInTheDocument();
    expect(screen.getByText(/No Inference Results Loaded/i)).toBeInTheDocument();
    // Await async system status resolution to avoid act() warning
    await screen.findByTestId('sys-engine-version');
  });

  it('renders complete dashboard with real mock DTO results on Demo Job click', async () => {
    renderDashboard();

    const demoBtn = screen.getByRole('button', { name: /load demo job/i });
    fireEvent.click(demoBtn);

    // 1. Verify Job Inspector Card
    expect(await screen.findByTestId('job-id-display')).toHaveTextContent(
      SAMPLE_JOB_RESULT.job_id
    );
    expect(screen.getByTestId('job-status-badge')).toHaveTextContent('completed');
    expect(screen.getByTestId('job-created-at')).not.toHaveTextContent('—');
    expect(screen.getByTestId('job-started-at')).not.toHaveTextContent('—');
    expect(screen.getByTestId('job-completed-at')).not.toHaveTextContent('—');

    // 2. Verify Video Metadata Card
    expect(screen.getByTestId('meta-resolution')).toHaveTextContent('1920 × 1080');
    expect(screen.getByTestId('meta-fps')).toHaveTextContent('30.0');
    expect(screen.getByTestId('meta-total-frames')).toHaveTextContent('45');
    expect(screen.getByTestId('meta-duration')).toHaveTextContent('1.50s');

    // 3. Verify Risk Timeline Section
    expect(screen.getByTestId('risk-timeline-section')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-stats-summary')).toBeInTheDocument();
    expect(screen.getByTestId('stat-safe-count')).toBeInTheDocument();
    expect(screen.getByTestId('stat-congesting-count')).toBeInTheDocument();
    expect(screen.getByTestId('stat-critical-count')).toBeInTheDocument();
    expect(screen.getByTestId('active-frame-risk-badge')).toHaveTextContent('Frame #0: Safe');

    // 4. Verify Regional Risk Predictions Grid
    expect(screen.getByTestId('regions-grid')).toBeInTheDocument();
    expect(screen.getByTestId('region-card-1')).toBeInTheDocument();
    expect(screen.getByTestId('region-card-2')).toBeInTheDocument();
    expect(screen.getByTestId('region-card-3')).toBeInTheDocument();
    expect(screen.getByTestId('region-card-4')).toBeInTheDocument();

    // Verify Region 1 initial predictions (Safe at Frame 0)
    expect(screen.getByTestId('region-card-1')).toHaveTextContent(/Region/i);
    expect(screen.getByTestId('region-card-1')).toHaveTextContent('Safe');

    // 5. Verify Output Video Card and Video Player
    expect(screen.getByTestId('output-video-player')).toBeInTheDocument();
    expect(screen.getByTestId('output-video-player')).toHaveAttribute(
      'src',
      expect.stringContaining(`/jobs/${SAMPLE_JOB_RESULT.job_id}/video`)
    );
    expect(screen.getByTestId('output-video-path')).toHaveTextContent(
      'annotated_corridor_inference.mp4'
    );
  });

  it('navigates frame scrubber and updates regional risk predictions', async () => {
    renderDashboard();

    fireEvent.click(screen.getByRole('button', { name: /load demo job/i }));

    expect(await screen.findByTestId('active-frame-risk-badge')).toHaveTextContent(
      'Frame #0: Safe'
    );

    // Step forward 16 frames using Step Forward button
    const stepForwardBtn = screen.getByRole('button', { name: /step to next frame/i });
    for (let i = 0; i < 16; i++) {
      fireEvent.click(stepForwardBtn);
    }

    // At frame index 16, Region 1 transitions to Congesting in SAMPLE_JOB_RESULT
    expect(screen.getByTestId('active-frame-risk-badge')).toHaveTextContent(
      'Frame #16: Congesting'
    );
    expect(screen.getByTestId('region-card-1')).toHaveTextContent('Congesting');
  });

  it('loads real mock DTO job via Job ID search', async () => {
    const customJob: JobResponse = {
      job_id: 'job-real-dto-9001',
      status: 'completed',
      created_at: '2026-11-14T10:00:00Z',
      started_at: '2026-11-14T10:00:02Z',
      completed_at: '2026-11-14T10:00:12Z',
      result: {
        output_video_path: '/outputs/real_surveillance.mp4',
        metadata: {
          fps: 25.0,
          width: 1280,
          height: 720,
          frame_count: 50,
          duration_seconds: 2.0,
          sample_rate: 1,
        },
        timeline: [
          {
            frame_index: 0,
            predictions: [
              { region_id: 101, label: 'Critical', confidence: 0.98 },
              { region_id: 102, label: 'Congesting', confidence: 0.85 },
            ],
          },
        ],
      },
    };

    const getJobSpy = vi.spyOn(apiClient, 'getJob').mockResolvedValue(customJob);

    renderDashboard();

    const input = screen.getByLabelText(/job id to inspect/i);
    fireEvent.change(input, { target: { value: 'job-real-dto-9001' } });

    const inspectBtn = screen.getByRole('button', { name: /inspect job/i });
    fireEvent.click(inspectBtn);

    await waitFor(() => {
      expect(getJobSpy).toHaveBeenCalledWith('job-real-dto-9001');
    });

    expect(await screen.findByTestId('job-id-display')).toHaveTextContent(
      'job-real-dto-9001'
    );
    expect(screen.getByTestId('meta-resolution')).toHaveTextContent('1280 × 720');
    expect(screen.getByTestId('meta-fps')).toHaveTextContent('25.0');
    expect(screen.getByTestId('meta-total-frames')).toHaveTextContent('50');

    // Check custom region prediction
    expect(screen.getByTestId('region-card-101')).toHaveTextContent(/Region/i);
    expect(screen.getByTestId('region-card-101')).toHaveTextContent('Critical');
  });

  it('renders fetch error banner when job ID is not found', async () => {
    vi.spyOn(apiClient, 'getJob').mockRejectedValue(
      new Error("Job 'job-nonexistent' not found.")
    );

    renderDashboard();

    const input = screen.getByLabelText(/job id to inspect/i);
    fireEvent.change(input, { target: { value: 'job-nonexistent' } });

    fireEvent.click(screen.getByRole('button', { name: /inspect job/i }));

    expect(await screen.findByTestId('fetch-error-alert')).toBeInTheDocument();
    expect(screen.getByTestId('fetch-error-alert')).toHaveTextContent(
      "Job 'job-nonexistent' not found."
    );
  });

  it('clears active job and returns to empty state on Clear click', async () => {
    renderDashboard();

    fireEvent.click(screen.getByRole('button', { name: /load demo job/i }));
    expect(await screen.findByTestId('job-id-display')).toBeInTheDocument();

    const clearBtn = screen.getByRole('button', { name: /^clear$/i });
    fireEvent.click(clearBtn);

    await waitFor(() => {
      expect(screen.queryByTestId('job-id-display')).not.toBeInTheDocument();
    });
    expect(screen.getByTestId('dashboard-empty-state')).toBeInTheDocument();
  });
});
