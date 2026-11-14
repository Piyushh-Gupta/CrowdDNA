import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  apiClient,
  buildUrl,
  getHealth,
  getLive,
  getReady,
  getVersion,
  submitInference,
  getJob,
  getVideoUrl,
  cancelJob,
  pollJob,
} from '../src/api/client';
import { ApiError } from '../src/api/errors';
import { withRetry } from '../src/api/retry';
import type { JobResponse } from '../src/api/types';

describe('API Utils & Error Model', () => {
  it('creates ApiError with standard and compatibility fields', () => {
    const err = new ApiError('Validation failed', 'VALIDATION_ERROR', 422, [
      { loc: ['body', 'file'], msg: 'Field required', type: 'missing' },
    ]);
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe('Validation failed');
    expect(err.code).toBe('VALIDATION_ERROR');
    expect(err.statusCode).toBe(422);
    expect(err.status_code).toBe(422);
    expect(err.error).toBe('VALIDATION_ERROR');
    expect(Array.isArray(err.detail)).toBe(true);
  });

  it('retries idempotent requests upon server failure', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      if (attempts < 2) throw Object.assign(new Error('Server error'), { statusCode: 500 });
      return 'success';
    };
    const res = await withRetry(fn, 3, true);
    expect(res).toBe('success');
    expect(attempts).toBe(2);
  });

  it('does not retry 4xx client errors on non-idempotent operations', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      throw Object.assign(new Error('Bad Request'), { statusCode: 400 });
    };
    await expect(withRetry(fn, 3, false)).rejects.toThrow();
    expect(attempts).toBe(1);
  });

  it('buildUrl normalizes leading and trailing slashes correctly', () => {
    const url1 = buildUrl('/health');
    const url2 = buildUrl('health');
    expect(url1).toBe(url2);
    expect(url1.endsWith('/health')).toBe(true);
  });
});

describe('Typed System API', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('getHealth fetches /health and returns typed HealthStatus', async () => {
    const mockHealth = { status: 'healthy', timestamp: '2026-11-14T12:00:00Z', version: '1.0.0' };
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockHealth), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await getHealth();
    expect(result).toEqual(mockHealth);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/health$/),
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('getLive fetches /health/live and returns typed LiveStatus', async () => {
    const mockLive = { status: 'alive' };
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockLive), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await getLive();
    expect(result).toEqual(mockLive);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/health\/live$/),
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('getReady fetches /health/ready and returns typed ReadyStatus', async () => {
    const mockReady = { status: 'ready', ready: true, model_loaded: false, device: 'cpu' };
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockReady), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await getReady();
    expect(result).toEqual(mockReady);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/health\/ready$/),
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('getVersion fetches /version and returns typed VersionInfo', async () => {
    const mockVersion = { api_version: '1.0.0', engine_version: '1.0.0' };
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockVersion), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await getVersion();
    expect(result).toEqual(mockVersion);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/version$/),
      expect.objectContaining({ method: 'GET' })
    );
  });
});

describe('Typed Inference API', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('submitInference sends FormData with file and calibration, handling 202 Accepted', async () => {
    const mockJob: JobResponse = {
      job_id: 'job-12345',
      status: 'queued',
      created_at: '2026-11-14T12:00:00Z',
    };

    let capturedInit: RequestInit | undefined;
    globalThis.fetch = vi.fn().mockImplementation((_url, init) => {
      capturedInit = init;
      return Promise.resolve(
        new Response(JSON.stringify(mockJob), {
          status: 202,
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    const file = new File(['fake-video-content'], 'test.mp4', { type: 'video/mp4' });
    const result = await submitInference(file, true);

    expect(result).toEqual(mockJob);
    expect(capturedInit).toBeDefined();
    expect(capturedInit?.method).toBe('POST');
    expect(capturedInit?.body).toBeInstanceOf(FormData);

    const body = capturedInit?.body as FormData;
    expect(body.get('file')).toBeDefined();
    expect(body.get('enable_calibration')).toBe('true');

    // Headers should NOT manually set Content-Type so browser can compute boundary
    const headers = capturedInit?.headers as Headers;
    if (headers) {
      expect(headers.get('content-type')).toBeNull();
    }
  });

  it('getJob retrieves job status and payload', async () => {
    const mockJob: JobResponse = {
      job_id: 'job-xyz-789',
      status: 'running',
      created_at: '2026-11-14T12:00:00Z',
      started_at: '2026-11-14T12:00:02Z',
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockJob), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await getJob('job-xyz-789');
    expect(result).toEqual(mockJob);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/jobs\/job-xyz-789$/),
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('cancelJob calls DELETE /jobs/{jobId} and returns cancelled job', async () => {
    const mockCancelledJob: JobResponse = {
      job_id: 'job-xyz-789',
      status: 'cancelled',
      created_at: '2026-11-14T12:00:00Z',
      completed_at: '2026-11-14T12:00:05Z',
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockCancelledJob), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const result = await cancelJob('job-xyz-789');
    expect(result).toEqual(mockCancelledJob);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/jobs\/job-xyz-789$/),
      expect.objectContaining({ method: 'DELETE' })
    );
  });
});

describe('Polling & Lifecycle Management', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('pollJob polls until completed and calls onUpdate callback', async () => {
    const job1: JobResponse = { job_id: 'j1', status: 'queued', created_at: '' };
    const job2: JobResponse = { job_id: 'j1', status: 'running', created_at: '' };
    const job3: JobResponse = {
      job_id: 'j1',
      status: 'completed',
      created_at: '',
      result: { timeline: [], output_video_path: '/path/out.mp4' },
    };

    const updates: JobResponse[] = [];
    const getJobSpy = vi.spyOn(apiClient, 'getJob')
      .mockResolvedValueOnce(job1)
      .mockResolvedValueOnce(job2)
      .mockResolvedValueOnce(job3);

    const result = await pollJob('j1', 10, 5000, (job) => updates.push(job));

    expect(result.status).toBe('completed');
    expect(getJobSpy).toHaveBeenCalledTimes(3);
    expect(updates.map((u) => u.status)).toEqual(['queued', 'running', 'completed']);
  });

  it('pollJob resolves terminal failed status without endless polling', async () => {
    const failedJob: JobResponse = {
      job_id: 'j-fail',
      status: 'failed',
      created_at: '',
      error: 'Inference pipeline failure',
    };

    vi.spyOn(apiClient, 'getJob').mockResolvedValueOnce(failedJob);

    const result = await pollJob('j-fail', 10, 5000);
    expect(result.status).toBe('failed');
    expect(result.error).toBe('Inference pipeline failure');
  });

  it('pollJob throws ApiError when maxWaitMs is exceeded', async () => {
    const runningJob: JobResponse = { job_id: 'j-hang', status: 'running', created_at: '' };
    vi.spyOn(apiClient, 'getJob').mockResolvedValue(runningJob);

    await expect(pollJob('j-hang', 20, 50)).rejects.toThrow(ApiError);
  });

  it('pollJob terminates immediately when signal is aborted', async () => {
    const controller = new AbortController();
    controller.abort();

    await expect(pollJob('j-abort', 10, 5000, undefined, controller.signal)).rejects.toThrow();
  });
});

describe('Error Handling & Backend ErrorResponse Mapping', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('parses string detail in ErrorResponse payloads into structured ApiError', async () => {
    const errorPayload = {
      detail: "Unsupported file format '.txt'. Allowed formats: .avi, .mp4.",
      error: 'Bad Request',
      status_code: 400,
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(errorPayload), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    try {
      await getJob('invalid');
      expect.fail('Expected to throw ApiError');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.statusCode).toBe(400);
      expect(apiErr.code).toBe('Bad Request');
      expect(apiErr.message).toBe("Unsupported file format '.txt'. Allowed formats: .avi, .mp4.");
      expect(apiErr.detail).toBe("Unsupported file format '.txt'. Allowed formats: .avi, .mp4.");
    }
  });

  it('parses array detail in 422 validation ErrorResponse payloads', async () => {
    const validationPayload = {
      detail: [
        { loc: ['body', 'file'], msg: 'Missing file payload', type: 'missing' },
        { loc: ['body', 'enable_calibration'], msg: 'Must be boolean', type: 'type_error' },
      ],
      error: 'Validation Error',
      status_code: 422,
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(validationPayload), {
        status: 422,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    try {
      await getJob('validation-error');
      expect.fail('Expected to throw ApiError');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.statusCode).toBe(422);
      expect(apiErr.code).toBe('Validation Error');
      expect(apiErr.message).toContain('Missing file payload');
      expect(apiErr.message).toContain('Must be boolean');
      expect(Array.isArray(apiErr.detail)).toBe(true);
    }
  });

  it('handles non-JSON error responses gracefully', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('Resource Not Found', {
        status: 404,
        statusText: 'Not Found',
        headers: { 'Content-Type': 'text/plain' },
      })
    );

    try {
      await getHealth();
      expect.fail('Expected to throw ApiError');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.statusCode).toBe(404);
      expect(apiErr.message).toContain('Resource Not Found');
    }
  });

  it('constructs correct streaming video URL for a job', () => {
    const url = getVideoUrl('job-abc-123');
    expect(url).toContain('/jobs/job-abc-123/video');
    expect(apiClient.getVideoUrl('job-xyz-789')).toContain('/jobs/job-xyz-789/video');
  });
});
