import { useQuery } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import { Card } from '@/components/ui/Card';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
  });

  const { data: jobsData, isLoading: jobsLoading } = useQuery({
    queryKey: queryKeys.jobs,
    queryFn: () => api.getJobs(10),
  });

  const { data: modelsData } = useQuery({
    queryKey: queryKeys.models,
    queryFn: api.getModels,
  });

  const recentJobs = jobsData?.jobs?.slice(0, 5) ?? [];
  const models = modelsData?.models ?? [];
  const downloadedCount = models.filter((m) => m.downloaded).length;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card>
          <h2 className="text-sm font-medium text-gray-500">Health</h2>
          {healthLoading ? (
            <p className="text-lg">Loading...</p>
          ) : (
            <div className="mt-2">
              <p className={`text-lg font-semibold ${health?.status === 'ok' ? 'text-green-600' : 'text-red-600'}`}>
                {health?.status ?? 'Unknown'}
              </p>
              <p className="text-sm text-gray-600 mt-1">
                DB enabled: {health?.db_enabled ? 'Yes' : 'No'}
              </p>
            </div>
          )}
        </Card>

        <Card>
          <h2 className="text-sm font-medium text-gray-500">Models Ready</h2>
          <p className="text-2xl font-semibold mt-2">{downloadedCount} / {models.length}</p>
        </Card>

        <Card>
          <h2 className="text-sm font-medium text-gray-500">Recent Jobs</h2>
          {jobsLoading ? (
            <p className="text-sm">Loading...</p>
          ) : (
            <p className="text-2xl font-semibold mt-2">{recentJobs.length}</p>
          )}
        </Card>
      </div>

      <Card>
        <h2 className="text-lg font-semibold mb-3">Latest Jobs</h2>
        {recentJobs.length === 0 ? (
          <p className="text-gray-500">No jobs submitted yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {recentJobs.map((job) => (
                  <tr key={job.job_id} className="hover:bg-gray-50">
                    <td className="px-4 py-2 text-sm">
                      <Link to={`/jobs/${job.job_id}`} className="text-blue-600 hover:underline">
                        {job.job_id.substring(0, 8)}...
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-sm">{job.model_id}</td>
                    <td className="px-4 py-2 text-sm capitalize">{job.status}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{new Date(job.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}