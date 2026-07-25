
import { AuthProvider } from './providers/AuthProvider';
import { QueryProvider } from './providers/QueryProvider';
import { ThemeProvider } from './providers/ThemeProvider';
import { NotificationProvider } from './providers/NotificationProvider';
import { ModalProvider } from './providers/ModalProvider';
import { RealtimeProvider } from './providers/RealtimeProvider';
import { AppRouter } from './routing/index';
import './theme/tokens.css';
import './theme/variables.css';
import './theme/typography.css';
import './theme/spacing.css';
import './theme/animations.css';
import './theme/glass.css';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <QueryProvider>
          <NotificationProvider>
            <ModalProvider>
              <RealtimeProvider>
                <AppRouter />
              </RealtimeProvider>
            </ModalProvider>
          </NotificationProvider>
        </QueryProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
export default App;
