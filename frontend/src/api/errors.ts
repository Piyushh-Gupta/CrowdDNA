import type { ErrorDetail } from './types';

/**
 * Structured API error thrown when backend requests return non-2xx status codes.
 * Mirrors the FastAPI ErrorResponse schema ({ detail, error, status_code }).
 */
export class ApiError extends Error {
  readonly code: string;
  readonly statusCode: number;
  readonly detail?: string | ErrorDetail[];

  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    statusCode: number = 500,
    detail?: string | ErrorDetail[]
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.statusCode = statusCode;
    this.detail = detail ?? message;

    Object.setPrototypeOf(this, new.target.prototype);
  }

  /**
   * Alias for statusCode matching backend schema snake_case.
   */
  get status_code(): number {
    return this.statusCode;
  }

  /**
   * Alias for code matching backend schema error field.
   */
  get error(): string {
    return this.code;
  }
}
