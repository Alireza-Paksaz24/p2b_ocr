import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import { useSSE } from '@/hooks/useSSE';
import { Progress } from '@/components/ui/Progress';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { toast } from 'sonner';
import { useState, useEffect, useMemo } from 'react';
import { marked } from 'marked';

export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const {
    data: job,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: queryKeys.job(jobId!),
    queryFn: () => api.getJob(jobId!),
    enabled: !!jobId,
  });

  // Real‑time progress via SSE
  useSSE(`/ocr/jobs/${jobId}/progress`, {
    onMessage: (data) => {
      // Update query cache for the job
      queryClient.setQueryData(queryKeys.job(jobId!), (old: any) => ({
        ...old,
        status: data.status,
        progress: data.progress,
        message: data.message,
      }));
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteJob(jobId!),
    onSuccess: () => {
      toast.success('Job deleted');
      queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
      navigate('/jobs');
    },
    onError: (err: Error) => toast.error(`Delete failed: ${err.message}`),
  });

  // Markdown preview
  const markdownContent = useMemo(() => {
    if (job?.pages?.length) {
      // Combine pages content, assume page.content is markdown
      return job.pages.map((p) => p.content).join('\n\n');
    }
    return '';
  }, [job]);

  const htmlPreview = useMemo(() => {
    if (!markdownContent) return '';
    return marked.parse(markdownContent);
  }, [markdownContent]);

  if (isLoading) return <p>Loading job details...</p>;
  if (error || !job) return <p className="text-red-600">Error loading job: {error?.message || 'Not found'}</p>;

  const isCompleted = job.status === 'completed';
  const isFailed = job.status === 'failed';

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Job Detail</h1>
        <Button variant="danger" size="sm" onClick={() => setDeleteDialogOpen(true)}>
          Delete Job
        </Button>
      </div>

      <Card className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <span className="text-sm text-gray-500">Job ID</span>
            <p className="font-mono">{job.job_id}</p>
          </div>
          <div>
            <span className="text-sm text-gray-500">Model</span>
            <p>{job.model_id}</p>
          </div>
          <div>
            <span className="text-sm text-gray-500">Status</span>
            <p className="capitalize">{job.status}</p>
          </div>
          <div>
            <span className="text-sm text-gray-500">Output Formats</span>
            <p>{job.output_formats}</p>
          </div>
          <div>
            <span className="text-sm text-gray-500">Input Files</span>
            <ul className="list-disc list-inside">
              {job.input_files?.map((f, i) => <li key={i}>{f}</li>) ?? 'None'}
            </ul>
          </div>
        </div>

        {job.status !== 'completed' && job.status !== 'failed' && (
          <div className="mt-4">
            <Progress value={job.progress * 100} />
            <p className="text-sm text-gray-600 mt-1">{job.message || 'Processing...'}</p>
          </div>
        )}
      </Card>

      {isFailed && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <h3 className="text-red-800 font-semibold">Job Failed</h3>
          <p className="text-red-600 mt-1">{job.error || 'Unknown error'}</p>
        </Card>
      )}

      {isCompleted && (
        <>
          <Card className="mb-6">
            <h2 className="text-lg font-semibold mb-3">Download Results</h2>
            <div className="flex flex-wrap gap-3">
              {Object.entries(job.result_paths ?? {}).map(([format, path]) => (
                <a
                  key={format}
                  href={`/api/ocr/jobs/${jobId}/download/${path.split('/').pop()}`}
                  className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                  download
                >
                  Download {format.toUpperCase()}
                </a>
              ))}
            </div>
          </Card>

          {htmlPreview && (
            <Card>
              <h2 className="text-lg font-semibold mb-3">Markdown Preview</h2>
              <div
                className="prose prose-sm max-w-none border rounded p-4 bg-gray-50"
                dangerouslySetInnerHTML={{ __html: htmlPreview }}
              />
            </Card>
          )}
        </>
      )}

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        title="Delete Job"
        description="Are you sure you want to delete this job and all its files? This action cannot be undone."
      >
        <div className="flex justify-end gap-3">
          <Button variant="secondary" onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="danger"
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </div>
      </Dialog>
    </div>
  );
}