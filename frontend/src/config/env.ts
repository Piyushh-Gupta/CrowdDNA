/**
 * Application environment configuration.
 * Resolves API base URL from Vite environment variables with fallback to
 * local development server or empty string when running in production/proxy mode.
 */
export const env = {
  API_BASE_URL:
    import.meta.env.VITE_API_BASE_URL ||
    (import.meta.env.PROD ? '' : 'http://localhost:8000'),
  IS_PRODUCTION: Boolean(import.meta.env.PROD),
};
