
export class RealtimeClient {
  private url: string;
  private es: EventSource | null = null;
  private listeners: Map<string, Function[]> = new Map();
  private pollingTimer: number | null = null;
  private isVisible: boolean = true;

  constructor(url: string) {
    this.url = url;
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        this.isVisible = document.visibilityState === 'visible';
        if (this.isVisible && !this.es) this.connect();
      });
    }
  }

  connect() {
    if (!this.isVisible) return;
    this.es = new EventSource(this.url);
    this.es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const handlers = this.listeners.get(data.type) || [];
      handlers.forEach(fn => fn(data.payload));
    };
    this.es.onerror = () => {
      this.disconnect();
      this.startPollingFallback();
    };
  }

  startPollingFallback() {
    if (this.pollingTimer) return;
    this.pollingTimer = window.setInterval(() => {
      if (!this.isVisible) return;
      // Polling implementation here
    }, 5000);
  }

  on(type: string, handler: Function) {
    const handlers = this.listeners.get(type) || [];
    this.listeners.set(type, [...handlers, handler]);
  }

  off(type: string, handler: Function) {
    const handlers = this.listeners.get(type) || [];
    this.listeners.set(type, handlers.filter(h => h !== handler));
  }

  disconnect() {
    if (this.es) {
      this.es.close();
      this.es = null;
    }
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
  }
}
