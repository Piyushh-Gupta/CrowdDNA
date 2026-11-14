import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../providers/AuthProvider';

export interface ProtectedRouteProps {
  children: React.ReactNode;
  claim?: string;
  allowAnonymous?: boolean;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  claim,
  allowAnonymous = false,
}) => {
  const { token, permissions } = useAuth();

  // If anonymous access is allowed, bypass token verification
  if (allowAnonymous) {
    return <>{children}</>;
  }

  if (!token) return <Navigate to="/login" replace />;
  if (claim && !permissions.includes(claim)) return <div>403 Forbidden</div>;

  return <>{children}</>;
};
