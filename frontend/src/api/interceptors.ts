import { CONSTANTS } from '../config/constants';
import { ApiError } from './errors';
import type { ErrorDetail, ErrorResponse } from './types';

export const requestInterceptor = (config: RequestInit): RequestInit => {
  const token = typeof localStorage !== 'undefined' ? localStorage.getItem(CONSTANTS.TOKEN_KEY) : null;
  const headers = new Headers(config.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return { ...config, headers };
};

export const responseInterceptor = async <T = any>(response: Response): Promise<T> => {
  if (!response.ok) {
    if (response.status === 401) {
      // Mock token refresh handling
    }

    let payload: Partial<ErrorResponse> | null = null;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try {
        payload = (await response.json()) as Partial<ErrorResponse>;
      } catch {
        payload = null;
      }
    }

    if (payload && typeof payload === 'object') {
      const detail = payload.detail;
      let message = '';
      if (typeof detail === 'string' && detail.trim().length > 0) {
        message = detail;
      } else if (Array.isArray(detail) && detail.length > 0) {
        message = detail.map((d: ErrorDetail) => d.msg || JSON.stringify(d)).join('; ');
      } else if (typeof (payload as any).message === 'string') {
        message = (payload as any).message;
      } else {
        message = payload.error || response.statusText || `HTTP Error ${response.status}`;
      }

      const code = payload.error || (payload as any).code || 'HTTP_ERROR';
      const statusCode = typeof payload.status_code === 'number' ? payload.status_code : response.status;

      throw new ApiError(message, code, statusCode, detail);
    }

    // Non-JSON or empty response
    const text = await response.text().catch(() => '');
    const message = text || response.statusText || `HTTP Error ${response.status}`;
    throw new ApiError(message, response.statusText || 'HTTP_ERROR', response.status, text);
  }

  if (response.status === 204) {
    return undefined as unknown as T;
  }

  return response.json() as Promise<T>;
};
