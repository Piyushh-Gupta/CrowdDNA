import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import InferencePage, {
  validateVideoFile,
  ALLOWED_EXTENSIONS,
  MAX_UPLOAD_SIZE_BYTES,
} from '../src/features/inference/pages';
import { apiClient } from '../src/api/client';
import type { JobResponse } from '../src/api/types';

describe('Inference Video File Validation Unit Tests', () => {
  it('defines allowed video formats as .mp4 and .avi', () => {
    expect(ALLOWED_EXTENSIONS).toContain('.mp4');
    expect(ALLOWED_EXTENSIONS).toContain('.avi');
    expect(ALLOWED_EXTENSIONS.length).toBe(2);
  });

  it('accepts valid .mp4 video files', () => {
    const file = new File(['fake-mp4-data'], 'surveillance_cam01.mp4', {
      type: 'video/mp4',
    });
    const error = validateVideoFile(file);
    expect(error).toBeNull();
  });

  it('accepts valid uppercase .AVI video files', () => {
    const file = new File(['fake-avi-data'], 'STATION_CORRIDOR.AVI', {
      type: 'video/x-msvideo',
    });
    const error = validateVideoFile(file);
    expect(error).toBeNull();
  });

  it('rejects unsupported .txt files', () => {
    const file = new File(['plain text notes'], 'notes.txt', {
      type: 'text/plain',
    });
    const error = validateVideoFile(file);
    expect(error).not.toBeNull();
    expect(error).toContain('Unsupported format ".txt"');
  });

  it('rejects unsupported .png image files', () => {
    const file = new File(['image-bytes'], 'screenshot.png', {
      type: 'image/png',
    });
    const error = validateVideoFile(file);
    expect(error).not.toBeNull();
    expect(error).toContain('Unsupported format ".png"');
  });

  it('rejects empty 0-byte video files', () => {
    const file = new File([], 'empty.mp4', { type: 'video/mp4' });
    const error = validateVideoFile(file);
    expect(error).not.toBeNull();
    expect(error).toContain('empty (0 bytes)');
  });

  it('rejects video files exceeding max 200MB upload limit', () => {
    const largeFile = new File(['x'], 'huge.mp4', { type: 'video/mp4' });
    Object.defineProperty(largeFile, 'size', {
      value: MAX_UPLOAD_SIZE_BYTES + 1024,
      writable: false,
    });
    const error = validateVideoFile(largeFile);
    expect(error).not.toBeNull();
    expect(error).toContain('exceeds maximum allowed limit of 200 MB');
  });
});

describe('InferencePage Component Integration Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const renderComponent = () =>
    render(
      <MemoryRouter>
        <InferencePage />
      </MemoryRouter>
    );

  it('renders initial upload card with file input and disabled submit button', () => {
    renderComponent();

    expect(screen.getByRole('region', { name: /crowd inference engine/i })).toBeInTheDocument();
    expect(screen.getByTestId('video-file-input')).toBeInTheDocument();
    expect(screen.getByTestId('submit-inference-btn')).toBeDisabled();
    expect(screen.getByTestId('calibration-toggle')).not.toBeChecked();
  });

  it('rejects .txt file on upload and displays error alert', async () => {
    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    const invalidFile = new File(['dummy notes'], 'test_notes.txt', {
      type: 'text/plain',
    });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported format ".txt"');
    expect(screen.getByTestId('submit-inference-btn')).toBeDisabled();
    expect(screen.queryByTestId('selected-file-box')).not.toBeInTheDocument();
  });

  it('rejects .png file on upload and displays error alert', async () => {
    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    const invalidFile = new File(['dummy-bytes'], 'photo.png', {
      type: 'image/png',
    });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported format ".png"');
    expect(screen.getByTestId('submit-inference-btn')).toBeDisabled();
  });

  it('accepts .mp4 video file and enables submit button', async () => {
    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    const validFile = new File(['fake-mp4-stream'], 'crosswalk.mp4', {
      type: 'video/mp4',
    });

    fireEvent.change(fileInput, { target: { files: [validFile] } });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(await screen.findByTestId('selected-file-box')).toBeInTheDocument();
    expect(screen.getByTestId('selected-file-box')).toHaveTextContent('crosswalk.mp4');
    expect(screen.getByTestId('submit-inference-btn')).toBeEnabled();
  });

  it('submits valid video with calibration and renders queued job tracking state', async () => {
    const mockQueuedJob: JobResponse = {
      job_id: 'job-test-7711',
      status: 'queued',
      created_at: '2026-11-14T12:00:00Z',
    };

    const submitSpy = vi
      .spyOn(apiClient, 'submitInference')
      .mockResolvedValue(mockQueuedJob);

    // Mock getJob to return the same queued job during polling
    vi.spyOn(apiClient, 'getJob').mockResolvedValue(mockQueuedJob);

    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    const validFile = new File(['sample'], 'entrance.mp4', { type: 'video/mp4' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    const calibrationToggle = screen.getByTestId('calibration-toggle');
    fireEvent.click(calibrationToggle);
    expect(calibrationToggle).toBeChecked();

    const submitBtn = screen.getByTestId('submit-inference-btn');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(submitSpy).toHaveBeenCalledWith(validFile, true);
    });

    expect(await screen.findByTestId('job-tracking-card')).toBeInTheDocument();
    expect(screen.getByTestId('job-id-display')).toHaveTextContent('job-test-7711');
    expect(screen.getByTestId('job-status-badge')).toHaveTextContent('queued');
    expect(screen.getByTestId('timestamp-created')).toBeInTheDocument();
  });

  it('renders running job tracking state with cancel button and allows cancelling', async () => {
    const runningJob: JobResponse = {
      job_id: 'job-run-4422',
      status: 'running',
      created_at: '2026-11-14T12:00:00Z',
      started_at: '2026-11-14T12:00:02Z',
    };

    const cancelledJob: JobResponse = {
      job_id: 'job-run-4422',
      status: 'cancelled',
      created_at: '2026-11-14T12:00:00Z',
      started_at: '2026-11-14T12:00:02Z',
      completed_at: '2026-11-14T12:00:05Z',
    };

    vi.spyOn(apiClient, 'submitInference').mockResolvedValue(runningJob);
    vi.spyOn(apiClient, 'getJob').mockResolvedValue(runningJob);

    const cancelSpy = vi
      .spyOn(apiClient, 'cancelJob')
      .mockResolvedValue(cancelledJob);

    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    fireEvent.change(fileInput, {
      target: { files: [new File(['vid'], 'hallway.avi', { type: 'video/x-msvideo' })] },
    });
    fireEvent.click(screen.getByTestId('submit-inference-btn'));

    expect(await screen.findByTestId('job-tracking-card')).toBeInTheDocument();
    expect(screen.getByTestId('job-status-badge')).toHaveTextContent('running');
    expect(screen.getByTestId('cancel-job-btn')).toBeInTheDocument();

    // Cancel job
    fireEvent.click(screen.getByTestId('cancel-job-btn'));

    await waitFor(() => {
      expect(cancelSpy).toHaveBeenCalledWith('job-run-4422');
    });

    await waitFor(() => {
      expect(screen.getByTestId('job-status-badge')).toHaveTextContent('cancelled');
    });
  });

  it('renders completed job state with output details and dashboard link', async () => {
    const completedJob: JobResponse = {
      job_id: 'job-comp-9988',
      status: 'completed',
      created_at: '2026-11-14T12:00:00Z',
      completed_at: '2026-11-14T12:00:10Z',
      result: {
        output_video_path: '/outputs/annotated_output.mp4',
        timeline: [
          {
            frame_index: 0,
            predictions: [{ region_id: 1, label: 'Safe', confidence: 0.95 }],
          },
        ],
      },
    };

    vi.spyOn(apiClient, 'submitInference').mockResolvedValue(completedJob);

    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    fireEvent.change(fileInput, {
      target: { files: [new File(['vid'], 'plaza.mp4', { type: 'video/mp4' })] },
    });
    fireEvent.click(screen.getByTestId('submit-inference-btn'));

    expect(await screen.findByText('Inference Completed Successfully!')).toBeInTheDocument();
    expect(screen.getByTestId('job-status-badge')).toHaveTextContent('completed');
    expect(screen.getByRole('button', { name: /view results in dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /go to dashboard/i })).toBeInTheDocument();
  });

  it('renders failed job state when inference fails', async () => {
    const failedJob: JobResponse = {
      job_id: 'job-fail-1122',
      status: 'failed',
      created_at: '2026-11-14T12:00:00Z',
      error: 'Corrupted video stream at frame 12',
    };

    vi.spyOn(apiClient, 'submitInference').mockResolvedValue(failedJob);

    renderComponent();

    const fileInput = screen.getByTestId('video-file-input');
    fireEvent.change(fileInput, {
      target: { files: [new File(['vid'], 'corrupt.mp4', { type: 'video/mp4' })] },
    });
    fireEvent.click(screen.getByTestId('submit-inference-btn'));

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Corrupted video stream at frame 12');
    expect(screen.getByTestId('job-status-badge')).toHaveTextContent('failed');
  });
});
