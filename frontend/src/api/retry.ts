
export const withRetry = async <T>(fn: () => Promise<T>, retries = 3, isIdempotent = false): Promise<T> => {
  try {
    return await fn();
  } catch (error: any) {
    const statusCode = typeof error?.statusCode === 'number' ? error.statusCode : undefined;
    const isClientError = statusCode !== undefined && statusCode >= 400 && statusCode < 500 && statusCode !== 408 && statusCode !== 429;
    if (!isClientError && retries > 0 && (isIdempotent || (statusCode !== undefined && statusCode >= 500))) {
      await new Promise(res => setTimeout(res, 1000 * (4 - retries)));
      return withRetry(fn, retries - 1, isIdempotent);
    }
    throw error;
  }
};
