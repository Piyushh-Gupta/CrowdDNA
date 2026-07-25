
import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext<{ dark: boolean; toggle: () => void } | null>(null);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [dark, setDark] = useState(true);
  
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  return <ThemeContext.Provider value={{ dark, toggle: () => setDark(!dark) }}>{children}</ThemeContext.Provider>;
};

export const useTheme = () => useContext(ThemeContext)!;
