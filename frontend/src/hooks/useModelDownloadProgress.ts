import { useSSE } from './useSSE';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import type { DownloadProgress } from '@/lib/api';

export function useModelDownloadProgress(modelId: string) {
  const queryClient = useQueryClient();

  // Use SSE to receive updates, then update query cache
  useSSE(`/models/${modelId}/download/progress`, {
    onMessage: (data: DownloadProgress) => {
      queryClient.setQueryData(queryKeys.modelDownload(modelId), data);
    },
  });

  // Also initial poll (or use query data from SSE)
  return useQuery<DownloadProgress>({
    queryKey: queryKeys.modelDownload(modelId),
    queryFn: () => api.getModelDownloadStatus(modelId),
    refetchInterval: false, // SSE handles updates
    initialData: { status: 'queued', progress: 0, message: '', model_id: modelId },
  });
}