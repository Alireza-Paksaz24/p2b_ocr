import { useEffect, useState } from 'react';

interface SSEOptions {
  onMessage: (data: any) => void;
  onError?: (error: Event) => void;
}

export function useSSE(url: string, { onMessage, onError }: SSEOptions) {
  const [readyState, setReadyState] = useState<number>(EventSource.CONNECTING);

  useEffect(() => {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
    const fullUrl = `${baseUrl}${url}`;
    const eventSource = new EventSource(fullUrl);

    eventSource.onopen = () => setReadyState(EventSource.OPEN);
    eventSource.onerror = (err) => {
      setReadyState(EventSource.CLOSED);
      onError?.(err);
      eventSource.close();
    };

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        onMessage(parsed);
      } catch {
        // ignore non-JSON
      }
    };

    return () => {
      eventSource.close();
    };
  }, [url, onMessage, onError]);

  return readyState;
}