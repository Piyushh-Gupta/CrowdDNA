/**
 * CrowdDNA Centralized API Client.
 * Provides strongly-typed client bindings for System and Inference endpoints,
 * with error mapping to ApiError and resilient polling mechanisms.
 */

import { env } from '../config/env';
import { ApiError } from './errors';
import { requestInterceptor, responseInterceptor } from './interceptors';
import { withRetry } from './retry';
import type {
  HealthStatus,
  JobResponse,
  LiveStatus,
  ReadyStatus,
  VersionInfo,
} from './types';

export * from './types';
export { ApiError } from './errors';

/**
 * Normalizes an API path against the configured base URL.
 * Handles trailing and leading slashes gracefully.
 */
export function buildUrl(path: string): string {
  const base = (env.API_BASE_URL || '').replace(/\/+$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${cleanPath}`;
}

/**
 * Perform an HTTP GET request with retry.
 */
export async function get<T = unknown>(path: string, signal?: AbortSignal): Promise<T> {
  const url = buildUrl(path);
  const config = requestInterceptor({ method: 'GET', signal });
  return withRetry(() => fetch(url, config).then((res) => responseInterceptor<T>(res)), 3, true);
}

/**
 * Perform an HTTP POST request.
 */
export async function post<T = unknown>(
  path: string,
  data?: unknown,
  idempotencyKey?: string,
  signal?: AbortSignal
): Promise<T> {
  const url = buildUrl(path);
  const headers: Record<string, string> = {};
  if (idempotencyKey) {
    headers['Idempotency-Key'] = idempotencyKey;
  }
  let body: BodyInit | undefined;
  if (data !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(data);
  }
  const config = requestInterceptor({
    method: 'POST',
    headers,
    body,
    signal,
  });
  return withRetry(
    () => fetch(url, config).then((res) => responseInterceptor<T>(res)),
    3,
    Boolean(idempotencyKey)
  );
}

/**
 * Perform an HTTP DELETE request.
 */
export async function del<T = unknown>(path: string, signal?: AbortSignal): Promise<T> {
  const url = buildUrl(path);
  const config = requestInterceptor({ method: 'DELETE', signal });
  return withRetry(() => fetch(url, config).then((res) => responseInterceptor<T>(res)), 1, false);
}

// ============================================================================
// System API
// ============================================================================

/**
 * Retrieve high-level application health status, timestamp, and version.
 */
export async function getHealth(signal?: AbortSignal): Promise<HealthStatus> {
  return apiClient.get<HealthStatus>('/health', signal);
}

/**
 * Kubernetes-style liveness probe confirming process responsiveness.
 */
export async function getLive(signal?: AbortSignal): Promise<LiveStatus> {
  return apiClient.get<LiveStatus>('/health/live', signal);
}

/**
 * Kubernetes-style readiness probe reporting service availability and compute device.
 */
export async function getReady(signal?: AbortSignal): Promise<ReadyStatus> {
  return apiClient.get<ReadyStatus>('/health/ready', signal);
}

/**
 * Retrieve API specification version and core CrowdDNA engine version.
 */
export async function getVersion(signal?: AbortSignal): Promise<VersionInfo> {
  return apiClient.get<VersionInfo>('/version', signal);
}

// ============================================================================
// Inference API
// ============================================================================

/**
 * Submit a video file for asynchronous crowd risk inference.
 * Sends multipart/form-data with 'file' and 'enable_calibration'.
 * Handles 202 Accepted response.
 */
export async function submitInference(
  file: File,
  enableCalibration: boolean = false,
  signal?: AbortSignal
): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('enable_calibration', String(Boolean(enableCalibration)));

  const url = buildUrl('/inference');
  // Note: Do not set Content-Type header manually; browser/fetch assigns multipart boundary
  const config = requestInterceptor({
    method: 'POST',
    body: formData,
    signal,
  });

  const response = await fetch(url, config);
  return responseInterceptor<JobResponse>(response);
}

/**
 * Retrieve the current lifecycle state and results of an inference job.
 */
export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  return apiClient.get<JobResponse>(`/jobs/${encodeURIComponent(jobId)}`, signal);
}

/**
 * Cancel an active or queued inference job (DELETE /jobs/{jobId}).
 */
export async function cancelJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  return apiClient.delete<JobResponse>(`/jobs/${encodeURIComponent(jobId)}`, signal);
}

/**
 * Construct URL for streaming the annotated output video of a completed job (GET /jobs/{jobId}/video).
 */
export function getVideoUrl(jobId: string): string {
  return buildUrl(`/jobs/${encodeURIComponent(jobId)}/video`);
}

/**
 * Poll an inference job until it reaches a terminal status ('completed', 'failed', 'cancelled')
 * or exceeds the specified max wait duration.
 *
 * @param jobId Unique identifier of the job to monitor.
 * @param intervalMs Polling check interval in milliseconds (default: 1000ms).
 * @param maxWaitMs Maximum wait duration in milliseconds before timeout (default: 60000ms).
 * @param onUpdate Optional callback invoked on every poll iteration with the latest JobResponse.
 * @param signal Optional AbortSignal to cancel polling early.
 * @returns Resolves with the terminal JobResponse.
 */
export async function pollJob(
  jobId: string,
  intervalMs: number = 1000,
  maxWaitMs: number = 60000,
  onUpdate?: (job: JobResponse) => void,
  signal?: AbortSignal
): Promise<JobResponse> {
  const startTime = Date.now();

  while (true) {
    if (signal?.aborted) {
      throw new DOMException('Polling aborted', 'AbortError');
    }

    const job = await apiClient.getJob(jobId, signal);
    if (onUpdate) {
      onUpdate(job);
    }

    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
      return job;
    }

    const elapsed = Date.now() - startTime;
    if (maxWaitMs > 0 && elapsed >= maxWaitMs) {
      throw new ApiError(`Job polling timed out after ${maxWaitMs}ms`, 'POLLING_TIMEOUT', 408);
    }

    const remainingTime = maxWaitMs > 0 ? maxWaitMs - elapsed : Infinity;
    const sleepDuration = Math.min(intervalMs, remainingTime);

    // Abortable sleep
    await new Promise<void>((resolve, reject) => {
      if (signal?.aborted) {
        return reject(new DOMException('Polling aborted', 'AbortError'));
      }
      const timer = setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      }, sleepDuration);

      const onAbort = () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        reject(new DOMException('Polling aborted', 'AbortError'));
      };

      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }
}

// Unified client object preserving existing apiClient contract
export const apiClient = {
  buildUrl,
  get,
  post,
  delete: del,
  getHealth,
  getLive,
  getReady,
  getVersion,
  submitInference,
  getJob,
  getVideoUrl,
  cancelJob,
  pollJob,
};

export default apiClient;
