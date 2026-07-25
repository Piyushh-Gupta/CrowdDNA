
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../providers/AuthProvider';

export const ProtectedRoute: React.FC<{ children: React.ReactNode; claim?: string }> = ({ children, claim }) => {
  const { token, permissions } = useAuth();
  
  if (!token) return <Navigate to="/login" replace />;
  if (claim && !permissions.includes(claim)) return <div>403 Forbidden</div>;
  
  return <>{children}</>;
};
