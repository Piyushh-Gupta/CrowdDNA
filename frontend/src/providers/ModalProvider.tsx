
import React, { createContext, useState, useEffect, useCallback, useMemo } from 'react';

interface ModalContextType {
  setModal: (node: React.ReactNode | null) => void;
}
export const ModalContext = createContext<ModalContextType | null>(null);

export const ModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [modal, setModal] = useState<React.ReactNode | null>(null);
  
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape' && modal) setModal(null);
  }, [modal]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const value = useMemo(() => ({ setModal }), []);

  return (
    <ModalContext.Provider value={value}>
      {children}
      {modal && <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
        <div className="bg-white dark:bg-gray-800 p-4 rounded shadow-lg" role="dialog" aria-modal="true" tabIndex={-1}>
          {modal}
          <button onClick={() => setModal(null)} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded focus:outline-none focus:ring">Close</button>
        </div>
      </div>}
    </ModalContext.Provider>
  );
};
