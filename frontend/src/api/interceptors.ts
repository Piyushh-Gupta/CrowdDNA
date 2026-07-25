
import { CONSTANTS } from '../config/constants';
import { ApiError } from './errors';

export const requestInterceptor = (config: RequestInit): RequestInit => {
  const token = localStorage.getItem(CONSTANTS.TOKEN_KEY);
  const headers = new Headers(config.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return { ...config, headers };
};

export const responseInterceptor = async (response: Response) => {
  if (!response.ok) {
    if (response.status === 401) {
       // Mock token refresh handling
    }
    const data = await response.json().catch(() => ({}));
    throw new ApiError(data.message || 'API Error', data.code || 'UNKNOWN', response.status);
  }
  return response.json();
};
