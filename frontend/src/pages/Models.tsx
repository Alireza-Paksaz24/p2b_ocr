import { useQuery } from '@tanstack/react-query';
import { api, queryKeys } from '@/lib/api';
import { ModelDownloadCard } from '@/components/ModelDownloadCard';

export default function Models() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.models,
    queryFn: api.getModels,
  });

  if (isLoading) return <p>Loading models...</p>;
  if (error) return <p className="text-red-600">Error loading models: {error.message}</p>;

  const models = data?.models ?? [];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Model Management</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelDownloadCard key={model.id} model={model} />
        ))}
      </div>
    </div>
  );
}