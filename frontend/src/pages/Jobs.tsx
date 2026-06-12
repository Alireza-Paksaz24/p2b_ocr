import { useQuery } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import { Link } from 'react-router-dom';
import { Progress } from '@/components/ui/Progress';
// import { formatDistanceToNow } from 'date-fns'; // optional, we'll use simple formatting

export default function Jobs() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.jobs,
    queryFn: () => api.getJobs(50),
  });

  if (isLoading) return <p>Loading jobs...</p>;
  if (error) return <p className="text-red-600">Error loading jobs: {error.message}</p>;

  const jobs = data?.jobs ?? [];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">All Jobs</h1>
      <div className="overflow-x-auto bg-white shadow rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Job ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Progress</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {jobs.map((job) => (
              <tr key={job.job_id} className="hover:bg-gray-50">
                <td className="px-4 py-3 text-sm font-mono">
                  <Link to={`/jobs/${job.job_id}`} className="text-blue-600 hover:underline">
                    {job.job_id.substring(0, 8)}...
                  </Link>
                </td>
                <td className="px-4 py-3 text-sm">{job.model_id}</td>
                <td className="px-4 py-3 text-sm capitalize">{job.status}</td>
                <td className="px-4 py-3 text-sm">
                  {job.status === 'pending' || job.status === 'processing' ? (
                    <Progress value={job.progress * 100} />
                  ) : (
                    <span>{job.status}</span>
                  )}
                </td>
                <td className="px-4 py-3 text-sm text-gray-500">
                  {new Date(job.created_at).toLocaleString()}
                </td>
                <td className="px-4 py-3 text-sm">
                  <Link to={`/jobs/${job.job_id}`} className="text-blue-600 hover:underline">
                    View
                  </Link>
                </td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-gray-500">
                  No jobs found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}