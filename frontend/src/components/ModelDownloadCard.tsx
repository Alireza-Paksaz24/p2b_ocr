import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Progress } from './ui/Progress';
import { useModelDownloadProgress } from '@/hooks/useModelDownloadProgress';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import type { Model } from '@/lib/api';
import { toast } from 'sonner';

interface Props {
  model: Model;
}

export function ModelDownloadCard({ model }: Props) {
  const queryClient = useQueryClient();
  const downloadMutation = useMutation({
    mutationFn: () => api.downloadModel(model.id),
    onSuccess: () => {
      toast.success(`Download started for ${model.name}`);
      queryClient.invalidateQueries({ queryKey: queryKeys.models });
    },
    onError: (error: Error) => {
      toast.error(`Download failed: ${error.message}`);
    },
  });

  const { data: progress } = useModelDownloadProgress(model.id);

  const isDownloading = progress?.status === 'downloading' || progress?.status === 'queued';
  const isDone = model.downloaded || progress?.status === 'done';

  return (
    <Card className="flex flex-col justify-between">
      <div>
        <h3 className="text-lg font-semibold">{model.name}</h3>
        <p className="text-sm text-gray-600 mt-1">{model.description}</p>
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs bg-gray-100 px-2 py-0.5 rounded">{model.type}</span>
          <span className={`text-xs px-2 py-0.5 rounded ${model.downloaded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {model.downloaded ? 'Downloaded' : 'Not downloaded'}
          </span>
        </div>
      </div>
      <div className="mt-4">
        {isDone ? (
          <span className="text-sm text-green-600 font-medium">Ready</span>
        ) : (
          <>
            {isDownloading && progress && (
              <div className="mb-2">
                <Progress value={progress.progress} />
                <p className="text-xs text-gray-500 mt-1">{progress.message}</p>
              </div>
            )}
            <Button
              variant="primary"
              size="sm"
              disabled={isDownloading}
              onClick={() => downloadMutation.mutate()}
              className="w-full"
            >
              {isDownloading ? 'Downloading...' : 'Download'}
            </Button>
          </>
        )}
      </div>
    </Card>
  );
}