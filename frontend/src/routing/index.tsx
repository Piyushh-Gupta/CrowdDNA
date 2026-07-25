
import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { ProtectedRoute } from './ProtectedRoute';
import * as Lazy from './lazy';
import { featureFlags } from '../config/featureFlags';

export class GlobalErrorBoundary extends React.Component<{children: React.ReactNode}, {hasError: boolean}> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) return <div role="alert">Something went wrong.</div>;
    return this.props.children;
  }
}

const NotFound = () => <div>404 Not Found</div>;
const Login = () => <div>Login Page</div>;

export const AppRouter = () => (
  <BrowserRouter>
    <GlobalErrorBoundary>
      <Suspense fallback={<div aria-busy="true">Loading...</div>}>
        <div className="app-layout">
          <nav className="sidebar" aria-label="Main Navigation">
            <h2>CrowdDNA</h2>
            <ul>
              <li><Link to="/">Dashboard</Link></li>
              <li><Link to="/workflows">Workflows</Link></li>
              <li><Link to="/inference">Inference</Link></li>
              {featureFlags.enableExplainability && <li><Link to="/explainability">Explainability</Link></li>}
            </ul>
          </nav>
          <main className="content">
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<Lazy.Dashboard />} />
              <Route path="/workflows" element={<ProtectedRoute><Lazy.Workflows /></ProtectedRoute>} />
              <Route path="/inference" element={<ProtectedRoute><Lazy.Inference /></ProtectedRoute>} />
              {featureFlags.enableExplainability && <Route path="/explainability" element={<ProtectedRoute><Lazy.Explainability /></ProtectedRoute>} />}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
        </div>
      </Suspense>
    </GlobalErrorBoundary>
  </BrowserRouter>
);
