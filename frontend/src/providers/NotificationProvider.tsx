
import React, { createContext, useState, useCallback, useMemo } from 'react';

interface NotificationContextType {
  notify: (msg: string) => void;
}
export const NotificationContext = createContext<NotificationContextType | null>(null);

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<string[]>([]);
  
  const notify = useCallback((msg: string) => {
    setNotifications(prev => [...prev, msg]);
    setTimeout(() => setNotifications(prev => prev.slice(1)), 5000);
  }, []);
  
  const value = useMemo(() => ({ notify }), [notify]);
  
  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="fixed bottom-4 right-4" role="status" aria-live="polite">
        {notifications.map((n, i) => <div key={i} className="bg-gray-800 text-white p-2 mb-2 rounded shadow">{n}</div>)}
      </div>
    </NotificationContext.Provider>
  );
};
