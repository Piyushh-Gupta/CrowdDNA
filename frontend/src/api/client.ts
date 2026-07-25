
import { env } from '../config/env';
import { requestInterceptor, responseInterceptor } from './interceptors';
import { withRetry } from './retry';

export const apiClient = {
  get: (path: string, signal?: AbortSignal) => 
    withRetry(() => fetch(`${env.API_BASE_URL}${path}`, requestInterceptor({ method: 'GET', signal })).then(responseInterceptor), 3, true),
  
  post: (path: string, data: unknown, idempotencyKey: string, signal?: AbortSignal) => {
    const config = requestInterceptor({
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': idempotencyKey
      },
      body: JSON.stringify(data),
      signal
    });
    return withRetry(() => fetch(`${env.API_BASE_URL}${path}`, config).then(responseInterceptor), 3, true);
  }
};
