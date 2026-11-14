import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { ProtectedRoute } from '../src/routing/ProtectedRoute';
import InferencePage from '../src/features/inference/pages/index';
import { AuthProvider } from '../src/providers/AuthProvider';

describe('Routing & Unauthenticated Route Access', () => {
  it('allows direct access to /inference without login token', async () => {
    // Ensure localStorage has no token
    localStorage.clear();

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/inference']}>
          <Routes>
            <Route path="/login" element={<div>Login Page</div>} />
            <Route path="/inference" element={<InferencePage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    // Verify Inference Page is rendered directly without redirecting to Login
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /crowd inference & risk prediction/i })).toBeDefined();
    });
    expect(screen.queryByText('Login Page')).toBeNull();
  });

  it('ProtectedRoute with allowAnonymous allows direct access without token', async () => {
    localStorage.clear();

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/inference']}>
          <Routes>
            <Route path="/login" element={<div>Login Page</div>} />
            <Route
              path="/inference"
              element={
                <ProtectedRoute allowAnonymous>
                  <InferencePage />
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /crowd inference & risk prediction/i })).toBeDefined();
    });
    expect(screen.queryByText('Login Page')).toBeNull();
  });

  it('ProtectedRoute without allowAnonymous redirects unauthenticated users to /login', async () => {
    localStorage.clear();

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/workflows']}>
          <Routes>
            <Route path="/login" element={<div>Login Page</div>} />
            <Route
              path="/workflows"
              element={
                <ProtectedRoute>
                  <div>Workflows Content</div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Login Page')).toBeDefined();
    });
    expect(screen.queryByText('Workflows Content')).toBeNull();
  });
});
