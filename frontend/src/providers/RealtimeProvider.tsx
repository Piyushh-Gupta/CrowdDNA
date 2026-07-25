
import React, { createContext, useEffect, useState } from 'react';
import { RealtimeClient } from '../api/streaming';

export const RealtimeContext = createContext<RealtimeClient | null>(null);

export const RealtimeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [client] = useState(() => new RealtimeClient('/api/v1/stream'));
  
  useEffect(() => {
    client.connect();
    return () => client.disconnect();
  }, [client]);

  return <RealtimeContext.Provider value={client}>{children}</RealtimeContext.Provider>;
};
