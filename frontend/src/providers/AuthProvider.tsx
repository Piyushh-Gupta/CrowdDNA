
import React, { createContext, useContext, useState } from 'react';
import { CONSTANTS } from '../config/constants';

interface AuthContextType {
  token: string | null;
  permissions: string[];
  login: (token: string, permissions: string[]) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem(CONSTANTS.TOKEN_KEY));
  const [permissions, setPermissions] = useState<string[]>([]);

  const login = (newToken: string, perms: string[]) => {
    localStorage.setItem(CONSTANTS.TOKEN_KEY, newToken);
    setToken(newToken);
    setPermissions(perms);
  };

  const logout = () => {
    localStorage.removeItem(CONSTANTS.TOKEN_KEY);
    setToken(null);
    setPermissions([]);
  };

  return <AuthContext.Provider value={{ token, permissions, login, logout }}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext)!;
