
export const withRetry = async <T>(fn: () => Promise<T>, retries = 3, isIdempotent = false): Promise<T> => {
  try {
    return await fn();
  } catch (error: any) {
    if (retries > 0 && (isIdempotent || error.statusCode >= 500)) {
      await new Promise(res => setTimeout(res, 1000 * (4 - retries)));
      return withRetry(fn, retries - 1, isIdempotent);
    }
    throw error;
  }
};
