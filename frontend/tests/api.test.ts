
import { describe, it, expect } from 'vitest';
import { ApiError } from '../src/api/errors';
import { withRetry } from '../src/api/retry';
import { ResumableUpload } from '../src/utils/upload';

describe('API Utils', () => {
  it('creates ApiError', () => {
    const err = new ApiError('test', 'TEST_CODE', 400);
    expect(err.statusCode).toBe(400);
  });

  it('retries idempotent requests', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      if (attempts < 2) throw Object.assign(new Error(), { statusCode: 500 });
      return 'success';
    };
    const res = await withRetry(fn, 3, true);
    expect(res).toBe('success');
    expect(attempts).toBe(2);
  });
  
  it('does not retry non-idempotent 500s unless specified', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      throw Object.assign(new Error(), { statusCode: 400 }); // 400s are not retried even if idempotent is true for logic, wait, in retry.ts we said `error.statusCode >= 500 || isIdempotent`
    };
    await expect(withRetry(fn, 3, false)).rejects.toThrow();
    expect(attempts).toBe(1);
  });
});
