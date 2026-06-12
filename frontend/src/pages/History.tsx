import { useQuery } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import { Link } from 'react-router-dom';

export default function History() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.history,
    queryFn: api.getHistory,
  });

  if (isLoading) return <p>Loading history...</p>;
  if (error) return <p className="text-red-600">Error loading history: {error.message}</p>;

  const history = data?.history ?? [];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Job History</h1>
      {history.length === 0 ? (
        <p className="text-gray-500">No historical jobs found. History may be disabled on the server.</p>
      ) : (
        <div className="overflow-x-auto bg-white shadow rounded-lg">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Job ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Completed</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Input Files</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {history.map((entry) => (
                <tr key={entry.job_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-mono">
                    <Link to={`/jobs/${entry.job_id}`} className="text-blue-600 hover:underline">
                      {entry.job_id.substring(0, 8)}...
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-sm">{entry.model_id}</td>
                  <td className="px-4 py-3 text-sm capitalize">{entry.status}</td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {new Date(entry.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {entry.completed_at ? new Date(entry.completed_at).toLocaleString() : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {entry.input_files?.join(', ') || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}