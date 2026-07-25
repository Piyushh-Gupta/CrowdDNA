import { lazy } from 'react';

const Fallback = () => <div>Not Found</div>;

export const Dashboard = lazy(() => import('../features/dashboard/pages/index').catch(() => ({ default: Fallback })));
export const Workflows = lazy(() => import('../features/workflows/pages/index').catch(() => ({ default: Fallback })));
export const Inference = lazy(() => import('../features/inference/pages/index').catch(() => ({ default: Fallback })));
export const Explainability = lazy(() => import('../features/explainability/pages/index').catch(() => ({ default: Fallback })));
export const Experiments = lazy(() => import('../features/experiments/pages/index').catch(() => ({ default: Fallback })));
export const Observability = lazy(() => import('../features/observability/pages/index').catch(() => ({ default: Fallback })));
export const Reproducibility = lazy(() => import('../features/reproducibility/pages/index').catch(() => ({ default: Fallback })));
export const Security = lazy(() => import('../features/security/pages/index').catch(() => ({ default: Fallback })));
export const Settings = lazy(() => import('../features/settings/pages/index').catch(() => ({ default: Fallback })));
